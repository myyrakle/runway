# sentiment-service-rs

DeBERTa-v3 ABSA 감성 분석 서비스를 Rust로 재구현한 슬림 네이티브 TensorRT 서버입니다.
Python `sentiment-service`와 HTTP 호환(`/ping`, `/health`, `/invocations`, `/analyze`,
`/analyze/batch`)이지만, **런타임 이미지에 PyTorch·transformers·onnxruntime·Python이
전혀 없습니다** — Rust 바이너리, TensorRT 런타임 `.so`, CUDA/cuDNN, 미리 빌드한 엔진,
그리고 `tokenizer.json`만 들어갑니다.

배경: 네이티브 TensorRT 경로에서 모델 forward는 전부 `.plan` 엔진 안에 있습니다.
PyTorch는 그저 CUDA 메모리 할당자 + 스트림 핸들 + 3-클래스 softmax 역할이었고,
`transformers`는 토크나이저 용도뿐이었는데, 이 둘이 이미지에 수 GB를 끌고 들어옵니다.
이 프로젝트는 둘 다 들어냅니다.

## 아키텍처

```
HTTP (axum/tokio)
  └─ routes.rs        ping/health/invocations/analyze/analyze_batch
       └─ inference.rs  길이정렬 → 청킹(INFERENCE_BATCH_SIZE / MAX_BATCH_TOKENS)
                        → 토크나이즈 → backend.infer → softmax → 원순서 복원
            ├─ tokenizer.rs   HF `tokenizers` 크레이트, (text, aspect) 페어 인코딩
            └─ engine/         Backend 트레잇
                 ├─ trt.rs      FFI → cpp/trt_shim.cpp  (feature = "trt")
                 └─ stub.rs     CPU stub (GPU 불필요; 균등 logits) — 기본 빌드
cpp/trt_shim.cpp   libnvinfer + cudart 위의 C ABI: 엔진 deserialize, H2D, enqueueV3,
                   D2H. 추론 전용(빌더 없음)이라 TensorRT lean runtime 사용 가능.
                   int32↔int64 입력 / fp16↔fp32 출력 변환을 여기서 처리.
```

TensorRT 실행 컨텍스트는 thread-safe가 아니므로 각 forward를 `Mutex`로 직렬화합니다
(Python `TensorRTRunner`와 동일한 제약). 처리량 오버랩(CUDA 스트림 / 동적 배칭)은
의도적으로 후속 작업으로 남겨뒀습니다 — "다음 단계" 참고.

## `trt` feature 플래그

- **기본 빌드** (`cargo build`): CPU `stub` 백엔드 사용. CUDA/TensorRT 불필요 —
  어디서나 컴파일·실행되므로 HTTP/토크나이저/배칭 레이어를 노트북/CI에서 개발·테스트
  가능. 예측값은 무의미(균등 분포).
- **`--features trt`**: `build.rs`가 `cpp/trt_shim.cpp`를 컴파일하고 `libnvinfer` +
  `cudart`를 링크. 실제 엔진을 로드. Docker 이미지가 이 모드로 빌드.
- `STUB=1`: `trt` 빌드에서도 stub을 강제(GPU 없는 호스트에서 API 스모크 테스트용).

## 빌드 & 실행

이 프로젝트는 **자립형**입니다 — 엔진 export/build 스크립트(`scripts/`)와 그 Python
의존성이 vendoring돼 있어, 형제 `sentiment-service` 디렉터리 없이 단독으로 빌드됩니다.
모든 작업은 `make`로 감싸뒀습니다 (`make help` 참고).

### 로컬 (stub, GPU 불필요)

```bash
make tokenizer        # artifacts/tokenizer.json 생성 (python+transformers 필요)
make run              # CPU stub 백엔드로 :8080 기동 (STUB=1)

curl -s localhost:8080/health
curl -s -X POST localhost:8080/invocations -H 'content-type: application/json' \
  -d '{"texts":["great screen","awful battery"],"aspect":"overall"}'
```

`make check` / `build` / `fmt` / `clippy` / `test` / `clean` 도 제공합니다.

### Docker (실제 TensorRT 이미지)

빌드 컨텍스트는 **이 디렉터리(`sentiment-service-rs/`)** 입니다. BuildKit CDI GPU 설정
필요 (`sudo nvidia-ctk cdi generate ...` + daemon.json `{"features":{"cdi":true}}`).

```bash
make docker-build     # = docker buildx build --allow device=... -f docker/Dockerfile -t sentiment-rs:trt --load .
make docker-run       # = docker run --rm --gpus all -p 8080:8080 sentiment-rs:trt

# 더 큰 배치 엔진:
make docker-build INFERENCE_BATCH_SIZE=64 MAX_BATCH_TOKENS=32768
```

GPU 호스트에서 `--features trt`로 직접 빌드·실행도 가능: `make build-trt` / `make run-trt`
(엔진 `.plan`과 `tokenizer.json`이 `artifacts/`에 있어야 함, CUDA/TensorRT dev libs 필요).

`.plan`은 빌드 GPU 아키텍처 + TensorRT 버전에 고정됩니다: 빌드 호스트 GPU와 추론 GPU가
일치해야 합니다. 런타임 베이스의 Ubuntu/glibc는 TensorRT 빌더 이미지(`TRT_IMAGE`)의
것과 일치해야 합니다(기본값은 둘 다 Ubuntu 24.04).

## 설정 (환경변수)

| 변수 | 기본값 | 의미 |
|---|---|---|
| `TRT_ENGINE_PATH` | `artifacts/model.plan` | 직렬화된 엔진 |
| `TOKENIZER_PATH` | `artifacts/tokenizer.json` | fast 토크나이저 |
| `MODEL_NAME` | `yangheng/deberta-v3-base-absa-v1.1` | `/health`에 표시 |
| `MAX_LENGTH` | `512` | truncation 길이 |
| `INFERENCE_BATCH_SIZE` | `64` | forward 1회당 최대 텍스트 수 |
| `MAX_BATCH_TOKENS` | `0` | 토큰 버짓 청킹 (0 = 비활성) |
| `SORT_BATCH_BY_LENGTH` | `1` | 청킹 전 길이 정렬 |
| `PAD_TO_MULTIPLE_OF` | `0` (→ 8) | 시퀀스 길이를 배수로 패딩 |
| `NEGATIVE_THRESHOLD` | `0.6` | `is_negative` 판정 임계값 |
| `MAX_BATCH_ITEMS` | `1024` | 공개 배치 크기 상한 |
| `PORT` | `8080` | 리스닝 포트 |
| `STUB` | `0` | CPU stub 백엔드 강제 |

## 이미지 크기 추가 절감 레버

현재 런타임 베이스는 `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04` + 복사한
`libnvinfer`입니다. 더 줄이려면:

1. **엔진 빌드 시 cuDNN/cuBLAS 제외** — TensorRT tactic sources에서 `CUDNN`/`CUBLAS`를
   제외해 빌드(`sentiment-service/scripts/build_trt_engine.py`의
   `config.set_tactic_sources(...)`), 그 후 런타임 베이스를
   `nvidia/cuda:12.8.0-runtime`(cuDNN 없음)으로 변경 → ~1GB+ 절감.
2. **TensorRT lean runtime** — `libnvinfer_lean` 링크(`--features trt` + `TRT_LEAN=1`)
   하고 lean `.so`만 복사 → 전체 빌더/런타임 제거.
3. **distroless / scratch** — Rust 바이너리 + 정확히 필요한 `.so` 세트(`libcudart`,
   `libnvinfer_lean`, 그리고 1번을 안 했다면 cuDNN/cuBLAS)만 distroless 베이스에 복사.
   바이너리는 glibc 링크이므로 빌더의 glibc와 맞춰야 함.

## 검증 상태

- ✅ 기본(stub) 빌드 컴파일 통과; HTTP + 토크나이저 + 배칭을 CPU에서 end-to-end 검증
  완료(전체 엔드포인트, 배치 순서 보존, 검증 에러).
- ⚠️ `--features trt` 경로(C++ shim, FFI, 실제 엔진)는 여기서 **컴파일·실행되지
  않았습니다** — 개발 환경에 CUDA/TensorRT/GPU가 없었습니다. **GPU 호스트에서 빌드·검증**
  해야 하며, 동일 입력에 대해 Python 서비스의 `tensorrt` 백엔드 출력과 비교하는 것을
  권장합니다.

## 다음 단계 (처리량)

- 전역 forward `Mutex`를 단일 GPU 워커 + 요청 큐로 교체해 **동적 배칭**(동시 요청을 하나의
  GPU 배치로 병합) 가능하게.
- 다중 실행 컨텍스트 + CUDA 스트림으로 submit/execute 오버랩.
- 위 항목들은 "동시 배치 요청 2~3개" 처리량 문제를 이미지 크기와는 별개로 해결합니다.
