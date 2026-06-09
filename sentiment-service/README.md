# sentiment-service

DeBERTa v3 ABSA (`yangheng/deberta-v3-base-absa-v1.1`) 기반 Aspect-Based 감성분석을 FastAPI로 서빙하는 독립 서비스 (fp32, threshold 0.6, ABSA 형식).

## 실행

```bash
cd sentiment-service
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001
```

GPU가 있으면 자동으로 `cuda`, 없으면 `cpu`로 로드된다.

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

Swagger: `http://localhost:8001/docs`
