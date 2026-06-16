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
# BUILD CONTEXT = this directory (sentiment-service-rs/). Self-contained: the engine
# export/build scripts and their Python deps are vendored under scripts/, so there is
# NO dependency on the sibling sentiment-service project.
#   cd sentiment-service-rs
#   docker buildx build --allow device=nvidia.com/gpu=all \
#     -f docker/Dockerfile -t sentiment-rs:trt --load .
# (or just `make docker-build`)
#
# Requires BuildKit CDI GPU setup on the build host:
#   sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
#   /etc/docker/daemon.json: {"features": {"cdi": true}}
#
# The .plan is locked to the build GPU architecture + TensorRT version, so the build
# host GPU and the inference GPU must match.

ARG MODEL_NAME=yangheng/deberta-v3-base-absa-v1.1
# TENSORRT_PIP sets the TensorRT VERSION that builds the engine AND that the runtime
# loads it with — pinned to the EXACT pip wheel the Python service uses (see its
# uv.lock: tensorrt-cu12 10.16.1.11) so the engine's baked kernels (=> inference speed)
# match Python. Engine kernels are chosen at build time by this TRT version, so this is
# the knob that matters for latency parity.
ARG TENSORRT_PIP=tensorrt-cu12==10.16.1.11
# TRT_IMAGE only provides the C++ headers (NvInfer.h) + CUDA + cargo toolchain to COMPILE
# the shim in rust-builder. The shim uses only ABI-stable TensorRT 10.x API, so linking
# against this image's TRT and running against TENSORRT_PIP's (same major 10) is safe.
ARG TRT_IMAGE=nvcr.io/nvidia/tensorrt:25.05-py3
# Runtime base: minimal CUDA base (NO cudart/cudnn/cublas — those come from the
# TENSORRT_PIP wheel set copied in, so we don't ship them twice). Just glibc + the
# driver-injection setup. Must be the same Ubuntu as TRT_IMAGE (the Rust binary is
# compiled against TRT_IMAGE's glibc — nvcr 25.x = Ubuntu 24.04). libcuda.so is injected
# by `docker run --gpus` at run time.
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.8.0-base-ubuntu24.04
# Engine + chunking batch. INFERENCE_BATCH_SIZE is used for BOTH the engine's
# opt/max batch (build) AND the runtime chunk size, so they always match. To serve a
# bigger batch, rebuild with e.g. --build-arg INFERENCE_BATCH_SIZE=64 and bump
# MAX_BATCH_TOKENS to >= 64*MAX_SEQ (or 0 to disable token-budget chunking).
ARG INFERENCE_BATCH_SIZE=64
ARG MAX_SEQ=512
# >= INFERENCE_BATCH_SIZE * MAX_SEQ so a full 64-item chunk runs in ONE forward pass
# instead of being re-chunked by the token budget. Worst-case forward 64x512 (~3.5GB).
ARG MAX_BATCH_TOKENS=32768
# Engine optimization seq length. TensorRT tunes kernels for the OPT shape. Set to 512:
# the real workload truncates ~74% of inputs to MAX_LENGTH (p50 token len ~641), so most
# forwards run at seq=512 and that's the case worth tuning. (Python used the script
# default 256 — mis-tuned for a length that rarely occurs.) max-seq=512 is the upper bound.
ARG OPT_SEQ=512

# ============================== 1. engine-builder ==============================
# Identical to the Python service's tensorrt builder: python:3.10-slim + pip torch +
# pip TENSORRT_PIP. Building on the SAME pip TensorRT wheel as Python is what makes the
# baked engine (and thus inference speed) match. Exports fp32 ONNX, TRT builds fp16,
# then dumps a standalone tokenizer.json. Its pip TensorRT + CUDA libs are also what the
# runtime stage serves with (copied below).
FROM python:3.10-slim AS engine-builder
ARG MODEL_NAME
ARG INFERENCE_BATCH_SIZE
ARG MAX_SEQ
ARG OPT_SEQ
ARG TENSORRT_PIP
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

# Export deps (transformers/onnx/sentencepiece/...) + the pinned pip TensorRT (engine build).
COPY scripts ./scripts
RUN python -m pip install --no-cache-dir -r scripts/requirements-build.txt \
    && python -m pip install --no-cache-dir "${TENSORRT_PIP}"

# torch's bundled CUDA libs + the pip tensorrt_libs on the loader path so the TRT builder
# resolves its .so during the engine build.
RUN python -c "import glob,os,site; dirs=[p for r in (site.getsitepackages()+[site.getusersitepackages()]) for p in glob.glob(os.path.join(r,'nvidia','*','lib'))+glob.glob(os.path.join(r,'tensorrt_libs'))]; open('/etc/ld.so.conf.d/cuda-trt-libs.conf','w').write('\n'.join(dirs)+'\n')" \
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
      python scripts/export_onnx.py --model-name "$MODEL_NAME" --output "$ONNX_MODEL_PATH" --max-length "$MAX_SEQ" && \
      python scripts/build_trt_engine.py --onnx "$ONNX_MODEL_PATH" --output "$TRT_ENGINE_PATH" --opt-batch "$INFERENCE_BATCH_SIZE" --max-batch "$INFERENCE_BATCH_SIZE" --opt-seq "$OPT_SEQ" --max-seq "$MAX_SEQ" --fp16 && \
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

COPY Cargo.toml Cargo.lock build.rs ./
COPY cpp ./cpp
COPY src ./src
# Build the binary. It links against this image's libnvinfer (DT_NEEDED libnvinfer.so.10)
# but RUNS against TENSORRT_PIP's libnvinfer.so.10 — ABI-compatible within TensorRT major 10.
RUN cargo build --release --features trt \
    && cp target/release/sentiment-service-rs /usr/local/bin/sentiment-service-rs \
    && strip /usr/local/bin/sentiment-service-rs || true

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

# Copy ONLY the .so TensorRT inference needs — NOT torch's extra CUDA libs
# (cufft/cusolver/cusparse/nccl/nvjitlink/cusparselt) that bloated the image. From the
# pip 10.16 install: libnvinfer (+plugin) and the CUDA math libs TRT links/dlopens
# (cuDNN, cuBLAS, cudart, nvrtc). cuDNN's sub-libs are dlopen'd lazily, so we copy each
# whole lib/ dir, not individual files. The base image is cudnn-less to avoid duplication.
COPY --from=engine-builder /usr/local/lib/python3.10/site-packages/tensorrt_libs /opt/program/libs/tensorrt_libs
# cuDNN intentionally NOT copied: DeBERTa has no convolutions, so this engine selects no
# cuDNN tactics (verified — the engine loads and serves without it). Drops ~1.5-2.5GB.
# If a future model/engine needs cuDNN, re-add the nvidia/cudnn/lib copy + ld path entry.
COPY --from=engine-builder /usr/local/lib/python3.10/site-packages/nvidia/cublas/lib /opt/program/libs/cublas
COPY --from=engine-builder /usr/local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib /opt/program/libs/cuda_runtime
# cuda_nvrtc (runtime kernel JIT) also omitted: verified this engine infers correctly
# without it (DeBERTa fp16 uses no JIT-compiled tactics). Drops ~0.1GB.
COPY --from=rust-builder /usr/local/bin/sentiment-service-rs /usr/local/bin/sentiment-service-rs
COPY --from=engine-builder /build/artifacts/model.plan /opt/program/artifacts/model.plan
COPY --from=engine-builder /build/artifacts/tokenizer.json /opt/program/artifacts/tokenizer.json

# Put the copied TensorRT/CUDA .so dirs on LD_LIBRARY_PATH directly. The ldconfig cache
# below is built at image-build time, but the NVIDIA container runtime can re-run ldconfig
# at container start (regenerating /etc/ld.so.cache from the base image's conf), which
# drops these app dirs — so the binary failed at runtime with `libnvinfer.so.10: cannot
# open shared object file` despite the build-time ldd check passing. An explicit
# LD_LIBRARY_PATH resolves the libs regardless of cache state. (${LD_LIBRARY_PATH} keeps
# the base image's injected /usr/local/nvidia/lib* dirs.)
ENV LD_LIBRARY_PATH=/opt/program/libs/tensorrt_libs:/opt/program/libs/cublas:/opt/program/libs/cuda_runtime:${LD_LIBRARY_PATH}

# Drop the parts of tensorrt_libs that inference never loads (build-only tactic DB ~1GB,
# the ONNX parser, and the lean/dispatch/vc alternative runtimes), register the copied
# .so dirs, and fail the build if a NON-driver lib is unresolved (libcuda/libnvidia-* are
# injected by --gpus at run time, so they're expected-missing here).
RUN cd /opt/program/libs/tensorrt_libs \
    && rm -f libnvinfer_builder_resource* libnvonnxparser* libnvinfer_lean* libnvinfer_dispatch* libnvinfer_vc_plugin* 2>/dev/null || true; \
    { for d in /opt/program/libs/tensorrt_libs /opt/program/libs/cublas /opt/program/libs/cuda_runtime; do echo "$d"; done; } > /etc/ld.so.conf.d/trt.conf \
    && ldconfig \
    && missing="$(ldd /usr/local/bin/sentiment-service-rs | grep 'not found' | grep -vE 'libcuda|libnvidia' || true)" \
    && if [ -n "$missing" ]; then echo "MISSING NON-DRIVER LIBS:"; echo "$missing"; exit 1; fi

EXPOSE 8080
# ENTRYPOINT (not CMD) so SageMaker's `docker run <image> serve` works: the `serve` arg
# would REPLACE a CMD (exec'ing a nonexistent `serve` binary); as ENTRYPOINT it is an
# argv the binary harmlessly ignores (main() reads config from env, not args).
ENTRYPOINT ["sentiment-service-rs"]
