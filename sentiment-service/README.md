# sentiment-service

DeBERTa v3 ABSA (`yangheng/deberta-v3-base-absa-v1.1`) 기반 Aspect-Based 감성분석을 FastAPI로 서빙하는 독립 서비스 (threshold 0.6, ABSA 형식).

## Precision 변형

추론/벤치 엔드포인트에서 `precision`으로 모델 변형을 선택한다. 변형은 첫 요청 시 lazy 로드되어 캐시된다 (변형마다 메모리에 별도 상주).

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

## API

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 상태 + 모델/디바이스 |
| POST | `/analyze` | 단일 텍스트 |
| POST | `/analyze/batch` | 텍스트 배치 |
| POST | `/benchmark` | 추론 속도 측정 (avg/min/max ms) |

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

워밍업(`warmup`) 후 `iterations`회 반복 측정. `precision`으로 변형을, `mode`로 단건(`single`) vs 배치(`batch`, `batch_size`개 동시)를 고른다. 디폴트 바디로 그냥 호출해도 된다.

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

`mode=single`이면 `batch_size`는 1, `per_item_ms`는 null이다. 변형 비교는 `precision`만 바꿔 같은 요청을 반복 호출하면 된다.

Swagger: `http://localhost:8001/docs`
