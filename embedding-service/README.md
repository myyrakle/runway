# embedding-service

Multilingual E5 Large (`intfloat/multilingual-e5-large`, 1024-dim) 텍스트 임베딩을 FastAPI로 서빙하는 독립 서비스 (E5 prefix + L2 정규화).

## Precision 변형

추론/벤치 엔드포인트에서 `precision`으로 모델 변형을 선택한다. 변형은 첫 요청 시 lazy 로드되어 캐시된다.

| precision | 설명 | 제약 |
|-----------|------|------|
| `fp32` (기본) | 원본 정밀도 | — |
| `fp16` | half precision | CUDA 전용 (CPU면 400) |

## 실행

```bash
cd embedding-service
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8002
```

GPU 자동 감지(`cuda`/`cpu`).

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_NAME` | `intfloat/multilingual-e5-large` | HF repo id |
| `EMBEDDING_DIM` | `1024` | 출력 차원 |
| `DEFAULT_BATCH_SIZE` | `256` | encode 배치 |
| `DEFAULT_PREFIX` | `query: ` | 기본 E5 prefix |

## E5 prefix

E5는 모든 입력에 task prefix가 필요하다. 요청마다 `prefix`로 지정:
- `"query: "` — 짧은 쿼리/라벨
- `"passage: "` — 긴 문서/인용

## API

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 상태 + 모델/디바이스/차원 |
| POST | `/embed` | 텍스트 배치 → 정규화 벡터 |
| POST | `/benchmark` | 인코딩 속도 측정 (avg/min/max ms) |

```bash
curl -X POST localhost:8002/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["삼성 갤럭시 카메라 성능"], "prefix": "query: "}'
```

```json
{"embeddings": [[0.01, -0.03, ...]], "dim": 1024, "count": 1}
```

### 벤치마크

워밍업(`warmup`) 후 `iterations`회 반복 인코딩 측정. `precision`으로 변형을, `texts`로 배치 크기를 바꿔 처리량을 본다. `per_text_ms`는 텍스트당 평균(avg_ms / 텍스트 수).

```bash
curl -X POST localhost:8002/benchmark \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["문장1", "문장2"], "precision": "fp16", "iterations": 50, "warmup": 5}'
```

```json
{
  "model": "intfloat/multilingual-e5-large",
  "precision": "fp16",
  "device": "cuda",
  "iterations": 50,
  "warmup": 5,
  "texts_per_iteration": 2,
  "avg_ms": 14.3,
  "min_ms": 12.8,
  "max_ms": 25.1,
  "total_ms": 715.0,
  "per_text_ms": 7.15
}
```

Swagger: `http://localhost:8002/docs`
