"""DeBERTa ABSA model wrapper.

ABSA input format [CLS] text [SEP] aspect [SEP], 3-class label map.
Supports precision variants: fp32 (default), fp16 (CUDA), int8 (dynamic quant, CPU).
"""
from __future__ import annotations

import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import (
    DEVICE,
    INFERENCE_BATCH_SIZE,
    INFERENCE_BACKEND,
    MAX_BATCH_TOKENS,
    MAX_LENGTH,
    MODEL_NAME,
    NEGATIVE_THRESHOLD,
    ONNX_MODEL_PATH,
    PAD_TO_MULTIPLE_OF,
    SORT_BATCH_BY_LENGTH,
    TOKENIZER_SOURCE,
    TRT_ENGINE_PATH,
    TRT_CUDA_GRAPH,
    ORT_TRT_CACHE_PATH,
)

PRECISIONS = ("fp32", "fp16", "int8")


class DeBERTaABSA:
    """DeBERTa-based Aspect-Based Sentiment Analysis model."""

    # ABSA labels: 0=negative, 1=neutral, 2=positive
    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    def __init__(self, precision: str = "fp32") -> None:
        if precision not in PRECISIONS:
            raise ValueError(f"Unknown precision {precision!r}, expected {PRECISIONS}")
        self.precision = precision

        self.backend = INFERENCE_BACKEND
        print(
            f"[Model] Loading DeBERTa ABSA ({MODEL_NAME}) "
            f"precision={precision} backend={self.backend}..."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SOURCE)

        if self.backend != "pytorch":
            if self.backend in {"onnx-cuda", "ort-trt", "tensorrt"}:
                self.device = "cuda"
            else:
                self.device = DEVICE
            # ONNX Runtime takes CPU numpy and handles the host->device copy
            # itself; only the native TensorRT runner needs CUDA input tensors.
            self._inputs_to_cuda = self.backend == "tensorrt"
            self.model = self._load_accelerated_runner()
            self.pad_to_multiple_of = self._resolve_pad_to_multiple_of()
            print(
                f"[Model] Loaded precision={precision} backend={self.backend} "
                f"on {self.device}"
            )
            return

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()

        if precision == "fp16":
            if DEVICE != "cuda":
                raise ValueError("fp16 precision requires a CUDA device")
            self.device = "cuda"
            model = model.half().to(self.device)
        elif precision == "int8":
            # Dynamic quantization targets CPU Linear layers.
            self.device = "cpu"
            model = model.to(self.device)
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:  # fp32
            self.device = DEVICE
            model = model.to(self.device)

        self._inputs_to_cuda = self.device == "cuda"
        self.model = model
        self.pad_to_multiple_of = self._resolve_pad_to_multiple_of()
        print(f"[Model] Loaded precision={precision} backend=pytorch on {self.device}")

    def _load_accelerated_runner(self):
        if self.backend == "onnx-cuda":
            return OnnxRuntimeRunner(
                ONNX_MODEL_PATH,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        if self.backend == "ort-trt":
            return OnnxRuntimeRunner(
                ONNX_MODEL_PATH,
                providers=[
                    (
                        "TensorrtExecutionProvider",
                        {
                            "trt_fp16_enable": self.precision == "fp16",
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": ORT_TRT_CACHE_PATH,
                        },
                    ),
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )
        if self.backend == "tensorrt":
            return TensorRTRunner(TRT_ENGINE_PATH, cuda_graph=TRT_CUDA_GRAPH)
        raise ValueError(f"Unsupported backend {self.backend!r}")

    def _resolve_pad_to_multiple_of(self) -> int | None:
        if PAD_TO_MULTIPLE_OF > 0:
            return PAD_TO_MULTIPLE_OF
        if self.precision == "fp16" and self.device == "cuda":
            return 8
        return None

    def _tokenize(self, texts, aspects):
        kwargs = {
            "return_tensors": "pt",
            "truncation": True,
            "max_length": MAX_LENGTH,
            "padding": True,
        }
        pad_to_multiple_of = getattr(
            self, "pad_to_multiple_of", self._resolve_pad_to_multiple_of()
        )
        if pad_to_multiple_of is not None:
            kwargs["pad_to_multiple_of"] = pad_to_multiple_of

        inputs = self.tokenizer(texts, aspects, **kwargs)
        # ONNX backends keep inputs on CPU (the runner feeds numpy) to avoid a
        # wasteful host<->device round trip; pytorch/tensorrt need CUDA tensors.
        if getattr(self, "_inputs_to_cuda", self.device == "cuda"):
            return {k: v.to("cuda", non_blocking=True) for k, v in inputs.items()}
        return {k: v.to("cpu") for k, v in inputs.items()}

    def analyze(self, text: str, aspect: str = "overall") -> dict:
        """Single-text sentiment analysis (delegates to the batch chunk path)."""
        return self._analyze_batch_chunk([text], aspect)[0]

    def analyze_batch(self, texts: list[str], aspect: str = "overall") -> list[dict]:
        """Batch sentiment analysis."""
        if not texts:
            return []

        indexed_texts = list(enumerate(texts))
        if SORT_BATCH_BY_LENGTH:
            indexed_texts.sort(key=lambda item: len(item[1]) + len(aspect))

        ordered_results: list[dict | None] = [None] * len(texts)
        for chunk in self._iter_inference_chunks(indexed_texts, aspect):
            chunk_indices = [index for index, _ in chunk]
            chunk_texts = [text for _, text in chunk]
            chunk_results = self._analyze_batch_chunk(chunk_texts, aspect)
            for index, result in zip(chunk_indices, chunk_results):
                ordered_results[index] = result
        return [result for result in ordered_results if result is not None]

    def _iter_inference_chunks(
        self,
        indexed_texts: list[tuple[int, str]],
        aspect: str,
    ):
        if MAX_BATCH_TOKENS <= 0:
            for start in range(0, len(indexed_texts), INFERENCE_BATCH_SIZE):
                yield indexed_texts[start:start + INFERENCE_BATCH_SIZE]
            return

        chunk: list[tuple[int, str]] = []
        chunk_max_tokens = 0
        for item in indexed_texts:
            item_tokens = self._estimate_sequence_tokens(item[1], aspect)
            next_max_tokens = max(chunk_max_tokens, item_tokens)
            next_size = len(chunk) + 1
            would_exceed_size = next_size > INFERENCE_BATCH_SIZE
            would_exceed_tokens = next_size * next_max_tokens > MAX_BATCH_TOKENS

            if chunk and (would_exceed_size or would_exceed_tokens):
                yield chunk
                chunk = []
                chunk_max_tokens = 0

            chunk.append(item)
            chunk_max_tokens = max(chunk_max_tokens, item_tokens)

        if chunk:
            yield chunk

    def _estimate_sequence_tokens(self, text: str, aspect: str) -> int:
        # Cheap upper-ish estimate for chunking before tokenizer padding. The
        # tokenizer still performs exact truncation/padding for the model input.
        return min(MAX_LENGTH, len(text) + len(aspect) + 3)

    def _analyze_batch_chunk(self, texts: list[str], aspect: str) -> list[dict]:
        """Run one bounded model forward pass."""
        inputs = self._tokenize(texts, [aspect] * len(texts))

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = torch.softmax(logits, dim=-1)

        # One bulk device->host copy + C-level conversion to plain Python floats,
        # instead of ~5 per-row `.item()` boundary crossings (5k+ calls at batch=1k).
        # argmax is then a trivial 3-way scan over the already-host floats.
        probs_list = probs.to("cpu").tolist()

        label_map = self.label_map
        threshold = NEGATIVE_THRESHOLD
        results = []
        for prob in probs_list:
            neg, neu, pos = prob
            label_idx = prob.index(max(prob))  # first-max index, matches argmax
            confidence = prob[label_idx]
            results.append({
                "sentiment": label_map[label_idx],
                "confidence": confidence,
                "is_negative": label_idx == 0 and confidence >= threshold,
                "probs": {"negative": neg, "neutral": neu, "positive": pos},
            })

        return results


class OnnxRuntimeRunner:
    def __init__(self, model_path: str, providers) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime-gpu is required for ONNX accelerated backends. "
                "Install with `uv sync --extra onnx`."
            ) from exc

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # DeBERTa's disentangled attention forces some ops onto the CPU EP, so ORT
        # logs a Memcpy/constant-fold warning per node. They are informational and
        # very noisy; keep errors only. Set ORT_LOG_SEVERITY=1 to see them again.
        sess_options.log_severity_level = int(os.environ.get("ORT_LOG_SEVERITY", "3"))
        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        self.input_names = {input_meta.name for input_meta in self.session.get_inputs()}

    def __call__(self, **inputs):
        ort_inputs = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in inputs.items()
            if name in self.input_names
        }
        logits = self.session.run(["logits"], ort_inputs)[0]
        # fp16 graphs return fp16 logits; upcast so the CPU softmax downstream is safe.
        return torch.from_numpy(logits).float()


class _CapturedGraph:
    """One CUDA graph captured for a fixed set of input shapes."""

    __slots__ = ("graph", "static_inputs", "static_output")

    def __init__(self, graph, static_inputs, static_output) -> None:
        self.graph = graph
        self.static_inputs = static_inputs
        self.static_output = static_output


class TensorRTRunner:
    def __init__(self, engine_path: str, cuda_graph: bool = False) -> None:
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT is required for native TensorRT backend. "
                "Install with `uv sync --extra tensorrt` in a CUDA/TensorRT image."
            ) from exc

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as engine_file:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine at {engine_path}")
        self.context = self.engine.create_execution_context()
        self.tensor_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        # Partition IO once instead of re-scanning every call.
        self.input_names = [
            name for name in self.tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        if "logits" in self.tensor_names:
            self.output_name = "logits"
        else:
            self.output_name = next(
                name for name in self.tensor_names
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
            )
        # Per-call host work (re-setting shapes/addresses, allocating the output)
        # is what makes native-TRT latency jitter relative to ORT-TRT's C++ glue.
        # Cache so each is touched only when it actually changes.
        self._shape_cache: dict[str, tuple[int, ...]] = {}
        self._addr_cache: dict[str, int] = {}
        self._output_cache: dict[tuple[int, ...], "torch.Tensor"] = {}

        self.cuda_graph = cuda_graph
        # One captured graph per distinct input-shape signature. A shape that fails
        # to capture is marked None so we fall back to eager for it without retrying.
        self._graphs: dict[tuple, "_CapturedGraph | None"] = {}
        if cuda_graph:
            print("[Model] TensorRT CUDA graph capture enabled (per input shape)")

    def __call__(self, **inputs):
        contiguous = {name: inputs[name].contiguous() for name in self.input_names}
        if self.cuda_graph:
            key = tuple((name, tuple(contiguous[name].shape)) for name in self.input_names)
            entry = self._graphs.get(key, "missing")
            if entry == "missing":
                entry = self._capture(contiguous)
                self._graphs[key] = entry
            if entry is not None:
                for name in self.input_names:
                    entry.static_inputs[name].copy_(contiguous[name])
                entry.graph.replay()
                return entry.static_output
        return self._run_eager(contiguous)

    def _bind_shapes(self, contiguous):
        context = self.context
        for name in self.input_names:
            shape = tuple(contiguous[name].shape)
            if self._shape_cache.get(name) != shape:
                context.set_input_shape(name, shape)
                self._shape_cache[name] = shape

    def _run_eager(self, contiguous):
        context = self.context
        self._bind_shapes(contiguous)
        for name in self.input_names:
            addr = contiguous[name].data_ptr()
            if self._addr_cache.get(name) != addr:
                context.set_tensor_address(name, addr)
                self._addr_cache[name] = addr

        output_shape = tuple(context.get_tensor_shape(self.output_name))
        # Reuse the device buffer for a given output shape (batch size) instead of
        # allocating per call. Safe because the caller syncs (softmax(...).cpu())
        # before the next forward pass reuses it.
        logits = self._output_cache.get(output_shape)
        if logits is None:
            logits = torch.empty(output_shape, dtype=torch.float32, device="cuda")
            self._output_cache[output_shape] = logits
        out_addr = logits.data_ptr()
        if self._addr_cache.get(self.output_name) != out_addr:
            context.set_tensor_address(self.output_name, out_addr)
            self._addr_cache[self.output_name] = out_addr

        stream = torch.cuda.current_stream().cuda_stream
        ok = context.execute_async_v3(stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return logits

    def _capture(self, contiguous):
        """Capture a CUDA graph for this shape, binding fixed static IO buffers.

        Returns None (and logs) if capture fails, so the caller drops to the eager
        path for this shape rather than crashing the request.
        """
        context = self.context
        try:
            # Static buffers the graph reads from / writes to on every replay.
            static_inputs = {name: contiguous[name].clone() for name in self.input_names}
            self._bind_shapes(static_inputs)
            for name in self.input_names:
                context.set_tensor_address(name, static_inputs[name].data_ptr())
            output_shape = tuple(context.get_tensor_shape(self.output_name))
            static_output = torch.empty(output_shape, dtype=torch.float32, device="cuda")
            context.set_tensor_address(self.output_name, static_output.data_ptr())
            # The eager path's address cache no longer reflects the context bindings.
            self._addr_cache.clear()

            # Warm up on a side stream so any lazy TRT setup / autotuning happens
            # before capture (capturing an allocating kernel corrupts the graph).
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    if not context.execute_async_v3(side.cuda_stream):
                        raise RuntimeError("TensorRT warmup execute_async_v3 failed")
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            return _CapturedGraph(graph, static_inputs, static_output)
        except Exception as exc:  # capture is best-effort; never break inference
            print(
                f"[Model] CUDA graph capture failed for shape "
                f"{[tuple(contiguous[n].shape) for n in self.input_names]}: {exc}. "
                "Falling back to eager TensorRT for this shape."
            )
            self._shape_cache.clear()
            self._addr_cache.clear()
            return None
