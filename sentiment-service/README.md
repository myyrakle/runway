# sentiment-service

DeBERTa v3 ABSA (`yangheng/deberta-v3-base-absa-v1.1`) 기반 Aspect-Based 감성분석을 FastAPI로 서빙하는 독립 서비스 (threshold 0.6, ABSA 형식).

## Precision 변형

컨테이너는 `DEFAULT_PRECISION` 하나만 startup에 로드한다. `ALLOWED_PRECISIONS`에
없는 precision 요청은 400으로 거절되므로, fp32/fp16 모델이 한 프로세스 메모리에
동시에 상주하지 않게 배포 이미지를 나눠 운영한다.

| precision | 설명 | 제약 |
|-----------|------|------|
| `fp32` (기본) | 원본 정밀도 | — |
| `fp16` | half precision | CUDA 전용 (CPU면 400) |
| `int8` | 동적 양자화 (Linear) | CPU에서 실행 |

## 실행

```bash
cd sentiment-service
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001
```

GPU가 있으면 자동으로 `cuda`, 없으면 `cpu`로 로드된다.
Linux에서는 `torch==2.8.0+cu128` wheel을 사용하도록 고정되어 있으므로 NVIDIA Driver 570 / CUDA 12.8 호환 환경을 기준으로 한다.

로컬에서 `uv`로 fp16 전용 모드를 실행하려면 CUDA GPU가 있는 환경에서 precision
환경변수를 고정한다.

```bash
DEFAULT_PRECISION=fp16 \
ALLOWED_PRECISIONS=fp16 \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001
```

fp16은 CUDA 전용이므로 `torch.cuda.is_available()`가 `False`이면 startup에서
실패한다.

SageMaker Dockerfile과 빌드/실행 명령은 [docker/README.md](docker/README.md)에
정리되어 있다. CUDA 이미지는 precision별로 분리되어 있고, CPU 이미지는 PyTorch
CPU wheel을 사용한다.

```bash
docker build -f docker/Dockerfile.sagemaker-cpu -t sentiment-service:cpu .
docker build -f docker/Dockerfile.sagemaker-cu128-fp32 -t sentiment-service:cu128-fp32 .
docker build -f docker/Dockerfile.sagemaker-cu128-fp16 -t sentiment-service:cu128-fp16 .
docker build -f docker/Dockerfile.sagemaker-cu128-onnx-cuda -t sentiment-service:cu128-onnx-cuda .
docker build -f docker/Dockerfile.sagemaker-cu128-ort-trt -t sentiment-service:cu128-ort-trt .
docker build -f docker/Dockerfile.sagemaker-cu128-tensorrt -t sentiment-service:cu128-tensorrt .
```

CUDA 로딩 확인:

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_NAME` | `yangheng/deberta-v3-base-absa-v1.1` | HF repo id |
| `INFERENCE_BACKEND` | `pytorch` | `pytorch`, `onnx-cuda`, `ort-trt`, `tensorrt` |
| `ONNX_MODEL_PATH` | `artifacts/model.onnx` | ONNX backend artifact path |
| `TRT_ENGINE_PATH` | `artifacts/model.plan` | native TensorRT engine path |
| `ORT_TRT_CACHE_PATH` | `artifacts/ort_trt_cache` | ONNX Runtime TensorRT EP engine cache |
| `NEGATIVE_THRESHOLD` | `0.6` | 부정 판정 임계값 |
| `MAX_LENGTH` | `512` | 토큰 truncation 길이 |
| `MAX_BATCH_ITEMS` | `1024` | `/analyze/batch`, `/invocations` 배치 요청의 최대 텍스트 수 |
| `INFERENCE_BATCH_SIZE` | `64` | 한 번의 model forward에 넣는 내부 추론 청크 크기 |
| `MAX_BATCH_TOKENS` | `0` | 한 번의 model forward에 넣을 approximate padded token 예산. `0`이면 비활성화 |
| `SORT_BATCH_BY_LENGTH` | `1` | 배치를 길이순으로 묶어 chunk 내 padding 낭비 감소 |
| `PAD_TO_MULTIPLE_OF` | `0` | tokenizer padding 배수. `0`이면 CUDA fp16에서 자동으로 `8` 사용 |
| `DEFAULT_PRECISION` | `fp32` | startup에 로드할 precision |
| `ALLOWED_PRECISIONS` | `DEFAULT_PRECISION` | 이 컨테이너에서 허용할 precision 목록 |

## API

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 상태 + 모델/디바이스 |
| POST | `/invocations` | SageMaker 추론 진입점, 단일/배치 모두 지원 |
| POST | `/analyze` | 단일 텍스트 |
| POST | `/analyze/batch` | 텍스트 배치 |
| POST | `/benchmark` | 추론 속도 측정 (avg/min/max ms) |

배치 요청은 `MAX_BATCH_ITEMS`까지 받되, 실제 모델 forward는
`INFERENCE_BATCH_SIZE` 단위로 나눠 실행한다. 길이가 큰 배치 요청 하나가
GPU/CPU 메모리를 한 번에 밀어붙이지 않도록 하기 위한 안전장치다. 단, 여러
개의 `/analyze` 단건 요청을 서버 내부에서 자동으로 모아 micro-batching하는
큐는 아직 없다. 처리량이 중요하면 호출 측에서 `/invocations` 또는
`/analyze/batch`에 배치 payload를 보내야 한다. 두 엔드포인트는 같은 배치 추론
경로를 사용한다.

GPU fp16 배치에서는 기본적으로 tokenizer padding을 8의 배수로 맞춰 Tensor Core
친화적인 shape를 만들고, batch item을 길이순으로 chunking해서 dynamic padding
낭비를 줄인다. 응답 순서는 입력 순서대로 복원된다.
`MAX_BATCH_TOKENS`를 양수로 설정하면 개수 기준 `INFERENCE_BATCH_SIZE`와 함께
대략적인 padded token 예산도 넘지 않도록 chunk를 더 작게 나눈다.

### 배치 크기 가이드

최적 `INFERENCE_BATCH_SIZE`는 GPU, precision, 입력 길이 분포에 따라 달라진다.
이 저장소의 [sample/texts.json](sample/texts.json)처럼 긴 텍스트가 섞인 입력과
NVIDIA T4 + fp16 조합에서는 아래 값을 기준으로 시작한다.

| 목적 | 권장값 |
|------|--------|
| 안정 우선 | `INFERENCE_BATCH_SIZE=16` |
| T4 fp16 기본 추천 | `INFERENCE_BATCH_SIZE=32` |
| 처리량 실험 | `INFERENCE_BATCH_SIZE=64` |

T4 fp16 로컬 실행 예:

```bash
DEFAULT_PRECISION=fp16 \
ALLOWED_PRECISIONS=fp16 \
INFERENCE_BATCH_SIZE=32 \
MAX_BATCH_TOKENS=16384 \
SORT_BATCH_BY_LENGTH=1 \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001
```

`sample/texts.json`으로 실제 배치 요청:

```bash
curl -X POST localhost:8001/invocations \
  -H 'Content-Type: application/json' \
  -d "{\"texts\": $(cat sample/texts.json), \"aspect\": \"overall\"}"
```

튜닝은 `INFERENCE_BATCH_SIZE=16`, `32`, `64`를 비교한다. `per_item_ms`가 더 이상
줄지 않거나 OOM/latency 급증이 생기면 직전 값을 사용한다. `/benchmark`는 같은
텍스트를 반복하므로 `sample/texts.json`의 실제 길이 분포를 완전히 대변하지
않는다.

CPU int8 모드는 GPU가 없을 때의 fallback 성격이다. dedicated Dockerfile은
`DEFAULT_PRECISION=int8`, `ALLOWED_PRECISIONS=int8`, `INFERENCE_BATCH_SIZE=16`,
`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`를 기본값으로 둔다. 로컬 `uv` 실행도 CPU
코어 수에 맞춰 thread 값을 명시하는 편이 좋다.

```bash
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
DEFAULT_PRECISION=int8 \
ALLOWED_PRECISIONS=int8 \
INFERENCE_BATCH_SIZE=16 \
MAX_BATCH_TOKENS=4096 \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### ONNX/TensorRT 실행

PyTorch fp16 baseline을 먼저 측정한 뒤 아래 순서로 비교한다.

```bash
make onnx-export            # fp32 ONNX (artifacts/model.onnx)
make onnx-cuda-sample       # onnx-cuda, fp32 그래프
make onnx-export-fp16       # fp16 ONNX (artifacts/model.fp16.onnx)
make onnx-cuda-sample-fp16  # onnx-cuda, fp16 그래프 (Tensor Core)
make ort-trt-sample         # ONNX Runtime TensorRT EP (fp16)
make trt-build              # native TensorRT fp16 engine
make trt-sample             # native TensorRT (fp16)
```

`*-sample` 타깃은 `--warmup`(기본 2회) 이후를 측정하므로 ORT/cuDNN 첫 실행의
알고리즘 탐색 비용이 빠진다.

> onnx-cuda 백엔드는 ONNX 그래프 자체의 정밀도를 그대로 실행한다. fp32 ONNX를
> `--precision fp16`으로 돌려도 실제 연산은 fp32다. fp16 이득을 보려면
> `onnx-export-fp16`으로 변환한 그래프(`onnxconverter_common.float16`,
> `keep_io_types`로 int 입력·fp32 logits 유지)를 써야 한다. 반면 `ort-trt`와
> native `tensorrt`는 엔진 빌드 시 fp16을 켜므로 fp32 ONNX에서 출발해도 fp16으로
> 실행된다. fp16 변환은 정밀도 손실이 생길 수 있으니 `sample/`로 정확도를 확인한다.

서비스(FastAPI)로 띄울 때는 backend별 make 타깃을 쓰면 artifact path·precision·
CUDA 로더 경로가 한 번에 설정된다.

```bash
make serve-onnx-cuda        # fp32 ONNX
make serve-onnx-cuda-fp16   # fp16 ONNX (onnx-export-fp16 먼저)
make serve-ort-trt          # ort-trt (fp16)
make serve-tensorrt         # native TensorRT (trt-build 먼저)
```

또는 직접 artifact path와 backend를 지정한다.

```bash
INFERENCE_BACKEND=onnx-cuda \
ONNX_MODEL_PATH=artifacts/model.onnx \
DEFAULT_PRECISION=fp16 \
ALLOWED_PRECISIONS=fp16 \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001
```

```bash
INFERENCE_BACKEND=ort-trt \
ONNX_MODEL_PATH=artifacts/model.onnx \
ORT_TRT_CACHE_PATH=artifacts/ort_trt_cache \
DEFAULT_PRECISION=fp16 \
ALLOWED_PRECISIONS=fp16 \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001
```

```bash
INFERENCE_BACKEND=tensorrt \
TRT_ENGINE_PATH=artifacts/model.plan \
DEFAULT_PRECISION=fp16 \
ALLOWED_PRECISIONS=fp16 \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001
```

ONNX Runtime TensorRT EP와 native TensorRT는 CUDA/TensorRT 런타임 버전과 GPU
아키텍처 영향을 크게 받는다. T4에서 만든 TensorRT engine은 같은 계열 배포
환경에서 재사용하는 것을 권장한다.

```bash
curl -X POST localhost:8001/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text": "The battery life is terrible", "aspect": "battery"}'
```

SageMaker `/invocations` 배치 요청:

```bash
curl -X POST localhost:8001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["The battery life is terrible", "The screen is great"], "aspect": "overall"}'
```

SageMaker-style `instances` payload도 같은 배치 경로를 사용한다.

```bash
curl -X POST localhost:8001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"instances": ["The battery life is terrible", "The screen is great"], "aspect": "overall"}'
```

```json
{
  "sentiment": "negative",
  "confidence": 0.98,
  "is_negative": true,
  "probs": {"negative": 0.98, "neutral": 0.01, "positive": 0.01}
}
```

### 벤치마크

워밍업(`warmup`) 후 `iterations`회 반복 측정. `precision`은 컨테이너의
`ALLOWED_PRECISIONS` 안에서만 선택할 수 있고, `mode`로 단건(`single`) vs
배치(`batch`, `batch_size`개 동시)를 고른다. 디폴트 바디로 그냥 호출해도 된다.

```bash
# fp16, 64개 배치 인코딩 처리량
curl -X POST localhost:8001/benchmark \
  -H 'Content-Type: application/json' \
  -d '{"precision": "fp16", "mode": "batch", "batch_size": 64, "iterations": 50, "warmup": 5}'
```

```json
{
  "model": "yangheng/deberta-v3-base-absa-v1.1",
  "precision": "fp16",
  "device": "cuda",
  "mode": "batch",
  "batch_size": 64,
  "iterations": 50,
  "warmup": 5,
  "avg_ms": 41.2,
  "min_ms": 38.9,
  "max_ms": 55.1,
  "total_ms": 2060.0,
  "per_item_ms": 0.64
}
```

`mode=single`이면 `batch_size`는 1, `per_item_ms`는 null이다. fp32/fp16 변형
비교는 각각의 Docker 이미지로 띄운 endpoint를 비교한다.

Swagger: `http://localhost:8001/docs`
