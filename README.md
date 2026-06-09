# runway

Self-hosted AI 모델을 독립 FastAPI 추론 서비스로 서빙하는 모노레포.
각 서비스는 **완전 독립** (자체 `pyproject.toml` + `uv.lock`)이며 따로 배포/실행한다.

| 서비스 | 모델 | 용도 | 포트 |
|--------|------|------|------|
| [sentiment-service](sentiment-service/) | `yangheng/deberta-v3-base-absa-v1.1` (DeBERTa v3 ABSA) | Aspect-Based 감성분석 (negative/neutral/positive) | 8001 |
| [embedding-service](embedding-service/) | `intfloat/multilingual-e5-large` (1024-dim) | 다국어 텍스트 임베딩 | 8002 |

## 공통 실행 패턴

```bash
cd <service>
uv sync                       # uv가 호환 Python(3.12) 자동 설치
uv run uvicorn app.main:app --host 0.0.0.0 --port <port>
```

GPU가 있으면 자동으로 `cuda`, 없으면 `cpu`로 로드된다. 모델은 startup(lifespan) 시 1회 로드되어 메모리에 상주한다.

각 서비스의 상세 API/환경변수는 해당 디렉토리 README 참고.

## License

See [LICENSE](LICENSE).
