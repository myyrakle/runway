PY ?= uv run python
HOST ?= 0.0.0.0
PORT ?= 8001
VENV ?= .venv

MODEL_NAME ?= yangheng/deberta-v3-base-absa-v1.1
MAX_LENGTH ?= 512
PRECISION ?= fp16
BATCH_SIZE ?= 32
MAX_BATCH_TOKENS ?= 16384
# Pad sequence length to a multiple so the native-TRT CUDA-graph path sees only a
# few distinct shapes (one captured graph each). 0 = dynamic per-chunk padding.
PAD_MULTIPLE ?= 64
ONNX_MODEL ?= artifacts/model.onnx
ONNX_MODEL_FP16 ?= artifacts/model.fp16.onnx
TRT_ENGINE ?= artifacts/model.plan
ORT_TRT_CACHE ?= artifacts/ort_trt_cache
SAMPLE_TEXTS ?= sample/texts.json

# Load test (Locust). LOADTEST_HOST targets the running server; USERS = concurrent
# users (closed-loop, ≈ in-flight requests); LOAD_BATCH = texts per request.
LOADTEST_HOST ?= http://localhost:$(PORT)
LOADTEST_SAMPLE ?= /home/myyrakle/Download/chunk-0.json
USERS ?= 100
LOAD_BATCH ?= 100
RUN_TIME ?= 2m

# torch's cu128 wheel bundles the CUDA runtime as nvidia-* site-packages
# (libcusparseLt.so.0 etc.), and tensorrt-cu12 ships libnvinfer in tensorrt_libs/.
# Point the dynamic loader at both so `import torch` and the ort-trt EP (which
# dlopens libnvinfer) work without a system CUDA/TensorRT install. Resolved in the
# recipe shell so it sees .venv after `uv sync`; run `uv sync --extra onnx`
# (or `--extra tensorrt`) first.
WITH_CUDA = LD_LIBRARY_PATH="$$(ls -d $(VENV)/lib/python*/site-packages/nvidia/*/lib $(VENV)/lib/python*/site-packages/tensorrt_libs 2>/dev/null | tr '\n' ':')$$LD_LIBRARY_PATH"

.PHONY: serve-fp16 serve-onnx-cuda serve-ort-trt serve-tensorrt serve-tensorrt-cudagraph
.PHONY: onnx-export onnx-export-fp16 onnx-cuda-sample ort-trt-sample trt-build trt-sample trt-sample-cudagraph
.PHONY: docker-build-onnx-cuda docker-build-ort-trt docker-build-tensorrt docker-build-tensorrt-cudagraph
.PHONY: docker-build-tensorrt-fwdcompat docker-build-tensorrt-fwdcompat-cudagraph
.PHONY: loadtest loadtest-headless

# Serve the FastAPI app (uvicorn). Each target sets the backend + artifact paths
# the app reads at startup, plus the CUDA loader path. Hit http://HOST:PORT/docs.
serve-fp16:
	$(WITH_CUDA) DEFAULT_PRECISION=fp16 ALLOWED_PRECISIONS=fp16 INFERENCE_BATCH_SIZE=$(BATCH_SIZE) MAX_BATCH_TOKENS=$(MAX_BATCH_TOKENS) uv run uvicorn app.main:app --host $(HOST) --port $(PORT)

serve-onnx-cuda:
	$(WITH_CUDA) INFERENCE_BACKEND=onnx-cuda ONNX_MODEL_PATH=$(ONNX_MODEL_FP16) DEFAULT_PRECISION=fp16 ALLOWED_PRECISIONS=fp16 INFERENCE_BATCH_SIZE=$(BATCH_SIZE) MAX_BATCH_TOKENS=$(MAX_BATCH_TOKENS) uv run uvicorn app.main:app --host $(HOST) --port $(PORT)

serve-ort-trt:
	$(WITH_CUDA) INFERENCE_BACKEND=ort-trt ONNX_MODEL_PATH=$(ONNX_MODEL) ORT_TRT_CACHE_PATH=$(ORT_TRT_CACHE) DEFAULT_PRECISION=$(PRECISION) ALLOWED_PRECISIONS=$(PRECISION) INFERENCE_BATCH_SIZE=$(BATCH_SIZE) MAX_BATCH_TOKENS=$(MAX_BATCH_TOKENS) uv run uvicorn app.main:app --host $(HOST) --port $(PORT)

serve-tensorrt:
	$(WITH_CUDA) INFERENCE_BACKEND=tensorrt TRT_ENGINE_PATH=$(TRT_ENGINE) DEFAULT_PRECISION=$(PRECISION) ALLOWED_PRECISIONS=$(PRECISION) INFERENCE_BATCH_SIZE=$(BATCH_SIZE) MAX_BATCH_TOKENS=$(MAX_BATCH_TOKENS) uv run uvicorn app.main:app --host $(HOST) --port $(PORT)

onnx-export:
	$(WITH_CUDA) $(PY) scripts/export_onnx.py --model-name $(MODEL_NAME) --output $(ONNX_MODEL) --max-length $(MAX_LENGTH)

onnx-export-fp16:
	$(WITH_CUDA) $(PY) scripts/export_onnx.py --model-name $(MODEL_NAME) --output $(ONNX_MODEL_FP16) --max-length $(MAX_LENGTH) --fp16

onnx-cuda-sample:
	$(WITH_CUDA) $(PY) scripts/run_accelerated.py --backend onnx-cuda --precision fp16 --texts $(SAMPLE_TEXTS) --onnx $(ONNX_MODEL_FP16) --batch-size $(BATCH_SIZE) --max-batch-tokens $(MAX_BATCH_TOKENS)

ort-trt-sample:
	$(WITH_CUDA) $(PY) scripts/run_accelerated.py --backend ort-trt --precision $(PRECISION) --texts $(SAMPLE_TEXTS) --onnx $(ONNX_MODEL) --batch-size $(BATCH_SIZE) --max-batch-tokens $(MAX_BATCH_TOKENS)

trt-build:
	$(WITH_CUDA) $(PY) scripts/build_trt_engine.py --onnx $(ONNX_MODEL) --output $(TRT_ENGINE) --opt-batch $(BATCH_SIZE) --max-batch $(BATCH_SIZE) --max-seq $(MAX_LENGTH) --fp16

trt-sample:
	$(WITH_CUDA) $(PY) scripts/run_accelerated.py --backend tensorrt --precision $(PRECISION) --texts $(SAMPLE_TEXTS) --engine $(TRT_ENGINE) --batch-size $(BATCH_SIZE) --max-batch-tokens $(MAX_BATCH_TOKENS)

# Native TensorRT with per-shape CUDA-graph capture (near-constant latency, lower
# jitter). PAD_TO_MULTIPLE_OF buckets sequence lengths so only a few shapes — and
# thus a few captured graphs — appear. Compare its timing against `make trt-sample`.
trt-sample-cudagraph:
	$(WITH_CUDA) PAD_TO_MULTIPLE_OF=$(PAD_MULTIPLE) $(PY) scripts/run_accelerated.py --backend tensorrt --precision $(PRECISION) --texts $(SAMPLE_TEXTS) --engine $(TRT_ENGINE) --batch-size $(BATCH_SIZE) --max-batch-tokens $(MAX_BATCH_TOKENS) --cuda-graph

# Serve the FastAPI app on the native-TRT CUDA-graph path.
serve-tensorrt-cudagraph:
	$(WITH_CUDA) INFERENCE_BACKEND=tensorrt TRT_ENGINE_PATH=$(TRT_ENGINE) TRT_CUDA_GRAPH=1 PAD_TO_MULTIPLE_OF=$(PAD_MULTIPLE) DEFAULT_PRECISION=$(PRECISION) ALLOWED_PRECISIONS=$(PRECISION) INFERENCE_BATCH_SIZE=$(BATCH_SIZE) MAX_BATCH_TOKENS=$(MAX_BATCH_TOKENS) uv run uvicorn app.main:app --host $(HOST) --port $(PORT)

# Locust load test against a running server. `make loadtest` opens the web UI
# (http://localhost:8089); `make loadtest-headless` runs USERS for RUN_TIME and
# writes a CSV report. Install first: `uv sync --extra loadtest`.
loadtest:
	SAMPLE_FILE=$(LOADTEST_SAMPLE) BATCH_SIZE=$(LOAD_BATCH) PRECISION=$(PRECISION) \
		uv run locust -f loadtest/locustfile.py --host $(LOADTEST_HOST) -u $(USERS) -r $(USERS)

loadtest-headless:
	SAMPLE_FILE=$(LOADTEST_SAMPLE) BATCH_SIZE=$(LOAD_BATCH) PRECISION=$(PRECISION) \
		uv run locust -f loadtest/locustfile.py --host $(LOADTEST_HOST) -u $(USERS) -r $(USERS) \
		-t $(RUN_TIME) --headless --csv loadtest/report

docker-build-onnx-cuda:
	docker build -f docker/Dockerfile.sagemaker-cu128-onnx-cuda -t sentiment-service:cu128-onnx-cuda .

docker-build-ort-trt:
	docker build -f docker/Dockerfile.sagemaker-cu128-ort-trt -t sentiment-service:cu128-ort-trt .

# The TensorRT engine is built at image-build time and needs a live GPU. BuildKit does
# not honor the daemon's `default-runtime: nvidia`, so the GPU is injected into the
# engine-build RUN via CDI (`RUN --device=nvidia.com/gpu=all` in the Dockerfile).
# Build-host requirements:
#   - Docker Engine 28+ / BuildKit 0.20+ / Buildx 0.21+
#   - CDI spec generated:  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
#   - CDI enabled in daemon:  /etc/docker/daemon.json -> {"features":{"cdi":true}}
# `--allow device=nvidia.com/gpu=all` grants the device entitlement to the build.
# NOTE: the .plan engine is locked to the BUILD GPU's architecture — build on the same
# GPU family as the inference host (T4/sm75 for g4dn, L4/sm89 for g6).
docker-build-tensorrt:
	docker buildx build --allow device=nvidia.com/gpu=all -f docker/Dockerfile.sagemaker-cu128-tensorrt -t sentiment-service:cu128-tensorrt --load .

docker-build-tensorrt-cudagraph:
	docker buildx build --allow device=nvidia.com/gpu=all -f docker/Dockerfile.sagemaker-cu128-tensorrt --build-arg CUDA_GRAPH=1 --build-arg PAD_MULTIPLE=$(PAD_MULTIPLE) -t sentiment-service:cu128-tensorrt-cudagraph --load .

# Forward-compatible variant: bakes cuda-compat-12-8 into the runtime so the cu128 image
# runs on an OLD host driver (R470 / "CUDA 11.4", e.g. SageMaker ml.g4dn). Build the same
# way (CDI GPU into the engine step). MUST be built on the same GPU arch as the target
# (T4/sm75 for g4dn) — forward compat fixes the driver only, not the engine's arch lock.
docker-build-tensorrt-fwdcompat:
	docker buildx build --allow device=nvidia.com/gpu=all -f docker/Dockerfile.sagemaker-cu128-tensorrt-fwdcompat -t sentiment-service:cu128-tensorrt-fwdcompat --load .

docker-build-tensorrt-fwdcompat-cudagraph:
	docker buildx build --allow device=nvidia.com/gpu=all -f docker/Dockerfile.sagemaker-cu128-tensorrt-fwdcompat --build-arg CUDA_GRAPH=1 --build-arg PAD_MULTIPLE=$(PAD_MULTIPLE) -t sentiment-service:cu128-tensorrt-fwdcompat-cudagraph --load .
