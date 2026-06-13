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
# TensorRT version is set by these two images (must stay glibc/CUDA-compatible with
# each other). Match TRT_IMAGE's bundled TensorRT to your Python deployment's version
# so the engine kernels match. Verify with:
#   docker run --rm $(TRT_IMAGE) python -c "import tensorrt; print(tensorrt.__version__)"
TRT_IMAGE            ?= nvcr.io/nvidia/tensorrt:25.05-py3
CUDA_RUNTIME_IMAGE   ?= nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04
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
	@echo "  make docker-build multi-stage GPU build -> $(IMAGE)"
	@echo "  make docker-run   run $(IMAGE) with --gpus all on :$(PORT)"

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
.PHONY: docker-build docker-run
docker-build:
	docker buildx build --allow device=nvidia.com/gpu=all \
	  --build-arg TRT_IMAGE=$(TRT_IMAGE) \
	  --build-arg CUDA_RUNTIME_IMAGE=$(CUDA_RUNTIME_IMAGE) \
	  --build-arg MODEL_NAME=$(MODEL_NAME) \
	  --build-arg INFERENCE_BATCH_SIZE=$(INFERENCE_BATCH_SIZE) \
	  --build-arg MAX_SEQ=$(MAX_SEQ) \
	  --build-arg MAX_BATCH_TOKENS=$(MAX_BATCH_TOKENS) \
	  -f docker/Dockerfile -t $(IMAGE) --load .
docker-run:
	docker run --rm --gpus all -p $(PORT):8080 $(IMAGE)
