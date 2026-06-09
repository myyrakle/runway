PY ?= uv run python
HOST ?= 0.0.0.0
PORT ?= 8001
VENV ?= .venv

MODEL_NAME ?= yangheng/deberta-v3-base-absa-v1.1
MAX_LENGTH ?= 512
PRECISION ?= fp16
BATCH_SIZE ?= 32
MAX_BATCH_TOKENS ?= 16384
ONNX_MODEL ?= artifacts/model.onnx
ONNX_MODEL_FP16 ?= artifacts/model.fp16.onnx
TRT_ENGINE ?= artifacts/model.plan
ORT_TRT_CACHE ?= artifacts/ort_trt_cache
SAMPLE_TEXTS ?= sample/texts.json

# torch's cu128 wheel bundles the CUDA runtime as nvidia-* site-packages
# (libcusparseLt.so.0 etc.), and tensorrt-cu12 ships libnvinfer in tensorrt_libs/.
# Point the dynamic loader at both so `import torch` and the ort-trt EP (which
# dlopens libnvinfer) work without a system CUDA/TensorRT install. Resolved in the
# recipe shell so it sees .venv after `uv sync`; run `uv sync --extra onnx`
# (or `--extra tensorrt`) first.
WITH_CUDA = LD_LIBRARY_PATH="$$(ls -d $(VENV)/lib/python*/site-packages/nvidia/*/lib $(VENV)/lib/python*/site-packages/tensorrt_libs 2>/dev/null | tr '\n' ':')$$LD_LIBRARY_PATH"

.PHONY: serve-fp16 serve-onnx-cuda serve-ort-trt serve-tensorrt
.PHONY: onnx-export onnx-export-fp16 onnx-cuda-sample ort-trt-sample trt-build trt-sample
.PHONY: docker-build-onnx-cuda docker-build-ort-trt docker-build-tensorrt

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

docker-build-onnx-cuda:
	docker build -f docker/Dockerfile.sagemaker-cu128-onnx-cuda -t sentiment-service:cu128-onnx-cuda .

docker-build-ort-trt:
	docker build -f docker/Dockerfile.sagemaker-cu128-ort-trt -t sentiment-service:cu128-ort-trt .

docker-build-tensorrt:
	docker build -f docker/Dockerfile.sagemaker-cu128-tensorrt -t sentiment-service:cu128-tensorrt .
