# Docker images

SageMaker용 Dockerfile은 이 디렉터리에서 관리한다. 빌드 컨텍스트는 프로젝트
루트여야 하므로, 아래 명령은 `sentiment-service/`에서 실행한다.

## Images

| Dockerfile | Image tag | Runtime | Precision policy |
|------------|-----------|---------|------------------|
| `docker/Dockerfile.sagemaker-cpu` | `sentiment-service:cpu` | PyTorch CPU wheel | 기본 `fp32` |
| `docker/Dockerfile.sagemaker-cu128-fp32` | `sentiment-service:cu128-fp32` | PyTorch CUDA 12.8 wheel | `DEFAULT_PRECISION=fp32`, `ALLOWED_PRECISIONS=fp32` |
| `docker/Dockerfile.sagemaker-cu128-fp16` | `sentiment-service:cu128-fp16` | PyTorch CUDA 12.8 wheel | `DEFAULT_PRECISION=fp16`, `ALLOWED_PRECISIONS=fp16` |

## Build

```bash
docker build -f docker/Dockerfile.sagemaker-cpu -t sentiment-service:cpu .
docker build -f docker/Dockerfile.sagemaker-cu128-fp32 -t sentiment-service:cu128-fp32 .
docker build -f docker/Dockerfile.sagemaker-cu128-fp16 -t sentiment-service:cu128-fp16 .
```

## Local run

CPU image:

```bash
docker run --rm -p 8080:8080 sentiment-service:cpu
```

CUDA fp32 image:

```bash
docker run --rm --gpus all -p 8080:8080 sentiment-service:cu128-fp32
```

CUDA fp16 image:

```bash
docker run --rm --gpus all -p 8080:8080 sentiment-service:cu128-fp16
```

## Precision behavior

Each container loads only `DEFAULT_PRECISION` at startup. Requests with a
precision outside `ALLOWED_PRECISIONS` return 400, which prevents fp32 and fp16
model copies from being loaded into one process.

For CPU int8 experiments, override the CPU container environment:

```bash
docker run --rm -p 8080:8080 \
  -e ALLOWED_PRECISIONS=fp32,int8 \
  sentiment-service:cpu
```

Then request `{"precision": "int8"}` explicitly.

## SageMaker

The images expose port `8080` and provide `/ping` and `/invocations`, matching
SageMaker inference container expectations.

Use `/invocations` for both single and batch payloads. `text` can be either a
string or a list of strings. SageMaker-style `instances` batch payloads use the same bounded
`INFERENCE_BATCH_SIZE` model-forward path as `/analyze/batch`.

```json
{"text": ["The battery life is terrible", "The screen is great"], "aspect": "overall"}
```

```json
{"instances": ["The battery life is terrible", "The screen is great"], "aspect": "overall"}
```
