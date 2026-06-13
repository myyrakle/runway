# syntax=docker/dockerfile:1-labs
# Slim native-TensorRT sentiment service (Rust). Three stages:
#
#   1. engine-builder  — reuses the Python service's export/build tooling to produce
#                        model.plan + tokenizer.json (needs a GPU at build time).
#   2. rust-builder    — compiles the Rust binary with `--features trt` against the
#                        TensorRT dev headers/libs.
#   3. runtime         — slim image: the static-ish Rust binary + the TensorRT runtime
#                        .so + cudart/cudnn + the .plan + tokenizer.json. No Python,
#                        no PyTorch, no transformers, no onnxruntime.
#
# BUILD CONTEXT = the runway repo root (so both `sentiment-service/` and
# `sentiment-service-rs/` are visible):
#   docker buildx build --allow device=nvidia.com/gpu=all \
#     -f sentiment-service-rs/docker/Dockerfile -t sentiment-rs:trt --load .
#
# Requires the same BuildKit CDI GPU setup as the Python tensorrt image (see
# sentiment-service/docker/Dockerfile.sagemaker-cu128-tensorrt for the host prereqs).
#
# The .plan is locked to the build GPU architecture + TensorRT version, so the build
# host GPU and the inference GPU must match.

ARG MODEL_NAME=yangheng/deberta-v3-base-absa-v1.1
ARG TRT_IMAGE=nvcr.io/nvidia/tensorrt:25.05-py3
# IMPORTANT: the runtime base's Ubuntu/glibc must match TRT_IMAGE's, or the Rust binary
# (compiled in rust-builder against TRT_IMAGE's glibc) fails to load. nvcr TensorRT
# 25.x is Ubuntu 24.04 (glibc 2.39) — keep these aligned if you change TRT_IMAGE.
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04
# Engine + chunking batch. INFERENCE_BATCH_SIZE is used for BOTH the engine's
# opt/max batch (build) AND the runtime chunk size, so they always match. To serve a
# bigger batch, rebuild with e.g. --build-arg INFERENCE_BATCH_SIZE=64 and bump
# MAX_BATCH_TOKENS to >= 64*MAX_SEQ (or 0 to disable token-budget chunking).
ARG INFERENCE_BATCH_SIZE=32
ARG MAX_SEQ=512
ARG MAX_BATCH_TOKENS=16384

# ============================== 1. engine-builder ==============================
# Mirrors sentiment-service's tensorrt build: fp32 ONNX export, TRT builds fp16.
#
# CRITICAL: build the engine with the SAME TensorRT that the runtime links, or
# deserializeCudaEngine fails with a serialization-version mismatch. So this stage runs
# on TRT_IMAGE (the same image rust-builder/runtime take libnvinfer from) and uses its
# BUNDLED tensorrt — we deliberately do NOT pip-install tensorrt-cu12 (a different
# version, and a 4.3GB wheel). We add only torch + transformers + onnx for the export.
FROM ${TRT_IMAGE} AS engine-builder
ARG MODEL_NAME
ARG INFERENCE_BATCH_SIZE
ARG MAX_SEQ
WORKDIR /build

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/build/.cache/huggingface \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    ONNX_MODEL_PATH=/build/artifacts/model.onnx \
    TRT_ENGINE_PATH=/build/artifacts/model.plan \
    TOKENIZER_OUT=/build/artifacts/tokenizer.json

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Reuse the Python service's pyproject + export/build scripts (build context = repo root).
COPY sentiment-service/pyproject.toml sentiment-service/README.md ./sentiment-service/
COPY sentiment-service/app ./sentiment-service/app
COPY sentiment-service/scripts ./sentiment-service/scripts
# Base deps (torch/transformers/...) + onnx for the export. NOT the [tensorrt] extra:
# that pip-installs a different tensorrt-cu12 version and onnxruntime-gpu, neither of
# which we want here. build_trt_engine.py imports the image's bundled `tensorrt`.
RUN python -m pip install --no-cache-dir -e "./sentiment-service" \
    && python -m pip install --no-cache-dir "onnx>=1.16.0"

# Put torch's bundled CUDA libs on the loader path (the image's TensorRT libs are
# already registered by the base image).
RUN python -c "import glob,os,site; dirs=[p for r in (site.getsitepackages()+[site.getusersitepackages()]) for p in glob.glob(os.path.join(r,'nvidia','*','lib'))]; open('/etc/ld.so.conf.d/torch-cuda-libs.conf','w').write('\n'.join(dirs)+'\n')" \
    && ldconfig

ARG BUILD_ENGINE=1
# GPU is mounted into THIS step only (BuildKit CDI). Exports fp32 ONNX, lets TRT do
# fp16, then dumps the fast tokenizer to a standalone tokenizer.json the Rust runtime
# loads (no transformers needed at serve time).
RUN --device=nvidia.com/gpu=all \
    if [ "$BUILD_ENGINE" = "1" ]; then \
      ldconfig && \
      export LD_LIBRARY_PATH="$(dirname "$(find / -name 'libcuda.so*' 2>/dev/null | head -n1)"):${LD_LIBRARY_PATH}" && \
      mkdir -p /build/artifacts && \
      python sentiment-service/scripts/export_onnx.py --model-name "$MODEL_NAME" --output "$ONNX_MODEL_PATH" --max-length "$MAX_SEQ" && \
      python sentiment-service/scripts/build_trt_engine.py --onnx "$ONNX_MODEL_PATH" --output "$TRT_ENGINE_PATH" --opt-batch "$INFERENCE_BATCH_SIZE" --max-batch "$INFERENCE_BATCH_SIZE" --max-seq "$MAX_SEQ" --fp16 && \
      python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL_NAME').backend_tokenizer.save('$TOKENIZER_OUT')" && \
      rm -f "$ONNX_MODEL_PATH"; \
    fi

# ============================== 2. rust-builder ==============================
# Build inside the TensorRT image so NvInfer.h + libnvinfer + CUDA are present.
FROM ${TRT_IMAGE} AS rust-builder
WORKDIR /src

# Rust toolchain (the TRT image is Ubuntu-based; rustup is the simplest install).
ENV PATH=/root/.cargo/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && rm -rf /var/lib/apt/lists/*

# Header/lib locations inside the nvcr TensorRT image.
ENV CUDA_PATH=/usr/local/cuda \
    TENSORRT_DIR=/usr

COPY sentiment-service-rs/Cargo.toml sentiment-service-rs/build.rs ./
COPY sentiment-service-rs/cpp ./cpp
COPY sentiment-service-rs/src ./src
# Build the real TensorRT-linked binary.
RUN cargo build --release --features trt \
    && cp target/release/sentiment-service-rs /usr/local/bin/sentiment-service-rs \
    && strip /usr/local/bin/sentiment-service-rs || true

# Collect exactly the TensorRT runtime .so the binary needs (inference-only: nvinfer,
# NOT the builder resource). Resolved from the binary's own NEEDED list.
RUN mkdir -p /trt-libs \
    && for so in $(ldd /usr/local/bin/sentiment-service-rs | awk '/libnvinfer/ {print $3}'); do cp -v "$so"* /trt-libs/ 2>/dev/null || true; done \
    && cp -v /usr/lib/x86_64-linux-gnu/libnvinfer.so* /trt-libs/ 2>/dev/null || true

# ============================== 3. runtime ==============================
FROM ${CUDA_RUNTIME_IMAGE} AS runtime
ARG MODEL_NAME
# Inherit the build-time batch/seq values so the runtime chunker matches the engine
# the builder produced (no hardcoded drift).
ARG INFERENCE_BATCH_SIZE
ARG MAX_BATCH_TOKENS
ARG MAX_SEQ
WORKDIR /opt/program

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MODEL_NAME=${MODEL_NAME} \
    TRT_ENGINE_PATH=/opt/program/artifacts/model.plan \
    TOKENIZER_PATH=/opt/program/artifacts/tokenizer.json \
    INFERENCE_BATCH_SIZE=${INFERENCE_BATCH_SIZE} \
    MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS} \
    MAX_LENGTH=${MAX_SEQ} \
    PAD_TO_MULTIPLE_OF=8 \
    RUST_LOG=info

# TensorRT runtime libs (from rust-builder) + the binary + artifacts.
COPY --from=rust-builder /trt-libs/ /usr/local/lib/trt/
COPY --from=rust-builder /usr/local/bin/sentiment-service-rs /usr/local/bin/sentiment-service-rs
COPY --from=engine-builder /build/artifacts/model.plan /opt/program/artifacts/model.plan
COPY --from=engine-builder /build/artifacts/tokenizer.json /opt/program/artifacts/tokenizer.json

# Register the TRT libs with the loader.
RUN echo "/usr/local/lib/trt" > /etc/ld.so.conf.d/trt.conf && ldconfig

EXPOSE 8080
CMD ["sentiment-service-rs"]
