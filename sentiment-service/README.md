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

SageMaker Dockerfile과 빌드/실행 명령은 [docker/README.md](docker/README.md)에
정리되어 있다. CUDA 이미지는 precision별로 분리되어 있고, CPU 이미지는 PyTorch
CPU wheel을 사용한다.

```bash
docker build -f docker/Dockerfile.sagemaker-cpu -t sentiment-service:cpu .
docker build -f docker/Dockerfile.sagemaker-cu128-fp32 -t sentiment-service:cu128-fp32 .
docker build -f docker/Dockerfile.sagemaker-cu128-fp16 -t sentiment-service:cu128-fp16 .
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
| `NEGATIVE_THRESHOLD` | `0.6` | 부정 판정 임계값 |
| `MAX_LENGTH` | `512` | 토큰 truncation 길이 |
| `MAX_BATCH_ITEMS` | `1024` | `/analyze/batch`, `/invocations` 배치 요청의 최대 텍스트 수 |
| `INFERENCE_BATCH_SIZE` | `64` | 한 번의 model forward에 넣는 내부 추론 청크 크기 |
| `DEFAULT_PRECISION` | `fp32` | startup에 로드할 precision |
| `ALLOWED_PRECISIONS` | `DEFAULT_PRECISION` | 이 컨테이너에서 허용할 precision 목록 |

## API

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 상태 + 모델/디바이스 |
| POST | `/analyze` | 단일 텍스트 |
| POST | `/analyze/batch` | 텍스트 배치 |
| POST | `/benchmark` | 추론 속도 측정 (avg/min/max ms) |

배치 요청은 `MAX_BATCH_ITEMS`까지 받되, 실제 모델 forward는
`INFERENCE_BATCH_SIZE` 단위로 나눠 실행한다. 길이가 큰 배치 요청 하나가
GPU/CPU 메모리를 한 번에 밀어붙이지 않도록 하기 위한 안전장치다. 단, 여러
개의 `/analyze` 단건 요청을 서버 내부에서 자동으로 모아 micro-batching하는
큐는 아직 없다. 처리량이 중요하면 호출 측에서 `/analyze/batch`를 사용해야 한다.

```bash
curl -X POST localhost:8001/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text": "The battery life is terrible", "aspect": "battery"}'
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
