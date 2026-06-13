# sentiment-service-rs — self-contained build/run.
# Local dev uses the CPU stub backend (no GPU needed). The real TensorRT path and the
# Docker image build the engine + binary on a GPU host. Override any var on the CLI,
# e.g.  make docker-build INFERENCE_BATCH_SIZE=64 MAX_BATCH_TOKENS=32768

IMAGE                ?= sentiment-rs:trt
MODEL_NAME           ?= yangheng/deberta-v3-base-absa-v1.1
INFERENCE_BATCH_SIZE ?= 32
MAX_SEQ              ?= 512
MAX_BATCH_TOKENS     ?= 16384
PORT                 ?= 8080
TOKENIZER            ?= artifacts/tokenizer.json
# TENSORRT_PIP pins the TensorRT version that builds AND serves the engine — set to the
# SAME pip wheel as the Python deployment (its uv.lock: tensorrt-cu12 10.16.1.11) so the
# baked kernels / inference speed match. TRT_IMAGE only supplies headers+toolchain to
# compile the shim; CUDA_RUNTIME_IMAGE supplies cudart/glibc (same Ubuntu as TRT_IMAGE).
TENSORRT_PIP         ?= tensorrt-cu12==10.16.1.11
TRT_IMAGE            ?= nvcr.io/nvidia/tensorrt:25.05-py3
CUDA_RUNTIME_IMAGE   ?= nvidia/cuda:12.8.0-base-ubuntu24.04
# Forward-compat image (old-driver hosts, e.g. R470/ml.g4dn): see docker/Dockerfile.fwdcompat
IMAGE_FWDCOMPAT      ?= sentiment-rs:trt-fwdcompat
CUDA_COMPAT_IMAGE    ?= nvidia/cuda:12.8.0-base-ubuntu24.04
ENGINE               ?= artifacts/model.plan

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "local (no GPU — CPU stub backend):"
	@echo "  make check        cargo check"
	@echo "  make build        cargo build --release (stub)"
	@echo "  make tokenizer    generate $(TOKENIZER) (needs python+transformers)"
	@echo "  make run          run the stub server on :$(PORT)"
	@echo "  make fmt | clippy | test | clean"
	@echo "local (GPU + CUDA/TensorRT dev libs):"
	@echo "  make build-trt    cargo build --release --features trt"
	@echo "  make run-trt      run the real TensorRT server (needs $(ENGINE) + $(TOKENIZER))"
	@echo "docker (self-contained; context = this dir):"
	@echo "  make docker-build           multi-stage GPU build -> $(IMAGE)"
	@echo "  make docker-run             run $(IMAGE) with --gpus all on :$(PORT)"
	@echo "  make docker-build-fwdcompat old-driver (R470/g4dn) variant -> $(IMAGE_FWDCOMPAT)"
	@echo "  make docker-run-fwdcompat   run the fwdcompat image"

# ---------------- local: CPU stub (no GPU/TensorRT) ----------------
.PHONY: check build fmt clippy test clean
check:
	cargo check
build:
	cargo build --release
fmt:
	cargo fmt
clippy:
	cargo clippy --all-targets -- -D warnings
test:
	cargo test
clean:
	cargo clean

.PHONY: tokenizer
tokenizer:
	@mkdir -p $(dir $(TOKENIZER))
	python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$(MODEL_NAME)').backend_tokenizer.save('$(TOKENIZER)')"
	@echo "wrote $(TOKENIZER)"

.PHONY: run
run:
	TOKENIZER_PATH=$(TOKENIZER) STUB=1 PORT=$(PORT) cargo run

# ---------------- local: real TensorRT (GPU + headers/libs) ----------------
# Override CUDA_PATH / TENSORRT_DIR if your install differs from build.rs defaults.
.PHONY: build-trt run-trt
build-trt:
	cargo build --release --features trt
run-trt:
	TOKENIZER_PATH=$(TOKENIZER) TRT_ENGINE_PATH=$(ENGINE) \
	INFERENCE_BATCH_SIZE=$(INFERENCE_BATCH_SIZE) MAX_BATCH_TOKENS=$(MAX_BATCH_TOKENS) \
	MAX_LENGTH=$(MAX_SEQ) PORT=$(PORT) \
	cargo run --release --features trt

# ---------------- docker (self-contained; context = this directory) ----------------
.PHONY: docker-build docker-build-fwdcompat docker-run docker-run-fwdcompat
docker-build:
	docker buildx build --allow device=nvidia.com/gpu=all \
	  --build-arg TENSORRT_PIP=$(TENSORRT_PIP) \
	  --build-arg TRT_IMAGE=$(TRT_IMAGE) \
	  --build-arg CUDA_RUNTIME_IMAGE=$(CUDA_RUNTIME_IMAGE) \
	  --build-arg MODEL_NAME=$(MODEL_NAME) \
	  --build-arg INFERENCE_BATCH_SIZE=$(INFERENCE_BATCH_SIZE) \
	  --build-arg MAX_SEQ=$(MAX_SEQ) \
	  --build-arg MAX_BATCH_TOKENS=$(MAX_BATCH_TOKENS) \
	  -f docker/Dockerfile -t $(IMAGE) --load .

# Old-driver hosts (R470 / SageMaker ml.g4dn): bakes CUDA forward-compat libcuda.
docker-build-fwdcompat:
	docker buildx build --allow device=nvidia.com/gpu=all \
	  --build-arg TENSORRT_PIP=$(TENSORRT_PIP) \
	  --build-arg TRT_IMAGE=$(TRT_IMAGE) \
	  --build-arg CUDA_RUNTIME_IMAGE=$(CUDA_RUNTIME_IMAGE) \
	  --build-arg CUDA_COMPAT_IMAGE=$(CUDA_COMPAT_IMAGE) \
	  --build-arg MODEL_NAME=$(MODEL_NAME) \
	  --build-arg INFERENCE_BATCH_SIZE=$(INFERENCE_BATCH_SIZE) \
	  --build-arg MAX_SEQ=$(MAX_SEQ) \
	  --build-arg MAX_BATCH_TOKENS=$(MAX_BATCH_TOKENS) \
	  -f docker/Dockerfile.fwdcompat -t $(IMAGE_FWDCOMPAT) --load .
docker-run:
	docker run --rm --gpus all -p $(PORT):8080 $(IMAGE)
docker-run-fwdcompat:
	docker run --rm --gpus all -p $(PORT):8080 $(IMAGE_FWDCOMPAT)
