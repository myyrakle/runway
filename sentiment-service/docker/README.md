# Docker images

SageMaker용 Dockerfile은 이 디렉터리에서 관리한다. 빌드 컨텍스트는 프로젝트
루트여야 하므로, 아래 명령은 `sentiment-service/`에서 실행한다.

## Images

| Dockerfile | Image tag | Runtime | Precision policy |
|------------|-----------|---------|------------------|
| `docker/Dockerfile.sagemaker-cpu` | `sentiment-service:cpu` | PyTorch CPU wheel | 기본 `fp32` |
| `docker/Dockerfile.sagemaker-cpu-int8` | `sentiment-service:cpu-int8` | PyTorch CPU wheel | `DEFAULT_PRECISION=int8`, `ALLOWED_PRECISIONS=int8` |
| `docker/Dockerfile.sagemaker-cu128-fp32` | `sentiment-service:cu128-fp32` | PyTorch CUDA 12.8 wheel | `DEFAULT_PRECISION=fp32`, `ALLOWED_PRECISIONS=fp32` |
| `docker/Dockerfile.sagemaker-cu128-fp16` | `sentiment-service:cu128-fp16` | PyTorch CUDA 12.8 wheel | `DEFAULT_PRECISION=fp16`, `ALLOWED_PRECISIONS=fp16` |
| `docker/Dockerfile.sagemaker-cu128-onnx-cuda` | `sentiment-service:cu128-onnx-cuda` | ONNX Runtime CUDA EP | `INFERENCE_BACKEND=onnx-cuda` |
| `docker/Dockerfile.sagemaker-cu128-ort-trt` | `sentiment-service:cu128-ort-trt` | ONNX Runtime TensorRT EP | `INFERENCE_BACKEND=ort-trt` |
| `docker/Dockerfile.sagemaker-cu128-tensorrt` | `sentiment-service:cu128-tensorrt` | native TensorRT engine | `INFERENCE_BACKEND=tensorrt` |

## Build

```bash
docker build -f docker/Dockerfile.sagemaker-cpu -t sentiment-service:cpu .
docker build -f docker/Dockerfile.sagemaker-cpu-int8 -t sentiment-service:cpu-int8 .
docker build -f docker/Dockerfile.sagemaker-cu128-fp32 -t sentiment-service:cu128-fp32 .
docker build -f docker/Dockerfile.sagemaker-cu128-fp16 -t sentiment-service:cu128-fp16 .
docker build -f docker/Dockerfile.sagemaker-cu128-onnx-cuda -t sentiment-service:cu128-onnx-cuda .
docker build -f docker/Dockerfile.sagemaker-cu128-ort-trt -t sentiment-service:cu128-ort-trt .
docker build -f docker/Dockerfile.sagemaker-cu128-tensorrt -t sentiment-service:cu128-tensorrt .
```

## Local run

CPU image:

```bash
docker run --rm -p 8080:8080 sentiment-service:cpu
```

CPU int8 image:

```bash
docker run --rm -p 8080:8080 sentiment-service:cpu-int8
```

CUDA fp32 image:

```bash
docker run --rm --gpus all -p 8080:8080 sentiment-service:cu128-fp32
```

CUDA fp16 image:

```bash
docker run --rm --gpus all -p 8080:8080 sentiment-service:cu128-fp16
```

ONNX Runtime CUDA image:

```bash
docker run --rm --gpus all -p 8080:8080 sentiment-service:cu128-onnx-cuda
```

ONNX Runtime TensorRT EP image:

```bash
docker run --rm --gpus all -p 8080:8080 sentiment-service:cu128-ort-trt
```

Native TensorRT image:

```bash
docker run --rm --gpus all -p 8080:8080 sentiment-service:cu128-tensorrt
```

## Precision behavior

Each container loads only `DEFAULT_PRECISION` at startup. Requests with a
precision outside `ALLOWED_PRECISIONS` return 400, which prevents fp32 and fp16
model copies from being loaded into one process.

For CPU int8 experiments, prefer the dedicated int8 image. It defaults to
`INFERENCE_BATCH_SIZE=16`, `OMP_NUM_THREADS=4`, and `MKL_NUM_THREADS=4`.
Override those values for the instance CPU shape:

```bash
docker run --rm -p 8080:8080 \
  -e INFERENCE_BATCH_SIZE=8 \
  -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 \
  sentiment-service:cpu-int8
```

The generic CPU image can still allow int8 by overriding
`ALLOWED_PRECISIONS=fp32,int8`, but that can load multiple model variants in one
process. The dedicated int8 image avoids that.

## SageMaker

The images expose port `8080` and provide `/ping` and `/invocations`, matching
SageMaker inference container expectations.

Use `/invocations` for both single and batch payloads. Use `text` for one item
and `texts` for a batch. SageMaker-style `instances` batch payloads use the same bounded
`INFERENCE_BATCH_SIZE` model-forward path as `/analyze/batch`.

```json
{"texts": ["The battery life is terrible", "The screen is great"], "aspect": "overall"}
```

```json
{"instances": ["The battery life is terrible", "The screen is great"], "aspect": "overall"}
```

Batch inference defaults:

- `INFERENCE_BATCH_SIZE=64`: maximum items per model forward pass.
- `MAX_BATCH_TOKENS=0`: disabled by default; set a positive approximate padded-token budget per model forward.
- `SORT_BATCH_BY_LENGTH=1`: sort request items by approximate length before chunking, then restore response order.
- `PAD_TO_MULTIPLE_OF=0`: auto mode; CUDA fp16 uses tokenizer `pad_to_multiple_of=8`.

## Accelerated backends

The accelerated GPU backends require exported artifacts:

- `onnx-cuda`: loads `ONNX_MODEL_PATH` with ONNX Runtime `CUDAExecutionProvider`.
- `ort-trt`: loads `ONNX_MODEL_PATH` with ONNX Runtime `TensorrtExecutionProvider`, then falls back to CUDA EP.
- `tensorrt`: loads `TRT_ENGINE_PATH` as a native TensorRT engine.

The Dockerfiles export/build artifacts during image build by default. For faster
iteration, set `EXPORT_ONNX=0` or `BUILD_ENGINE=0` and mount prebuilt artifacts.
TensorRT images use `nvcr.io/nvidia/tensorrt:25.05-py3` by default; override
`BASE_IMAGE` if your CUDA/TensorRT runtime must match a different deployment
environment.
