"""DeBERTa ABSA model wrapper.

ABSA input format [CLS] text [SEP] aspect [SEP], 3-class label map.
Supports precision variants: fp32 (default), fp16 (CUDA), int8 (dynamic quant, CPU).
"""
from __future__ import annotations

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
    TRT_ENGINE_PATH,
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
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
            return TensorRTRunner(TRT_ENGINE_PATH)
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
        """Single-text sentiment analysis."""
        inputs = self._tokenize(text, aspect)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = torch.softmax(logits, dim=-1).detach().cpu()
            pred_label = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_label].item()

        sentiment = self.label_map[pred_label]
        is_negative = pred_label == 0 and confidence >= NEGATIVE_THRESHOLD

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "is_negative": is_negative,
            "probs": {
                "negative": probs[0][0].item(),
                "neutral": probs[0][1].item(),
                "positive": probs[0][2].item(),
            },
        }

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
            probs = torch.softmax(logits, dim=-1).detach().cpu()
            pred_labels = torch.argmax(probs, dim=-1).detach().cpu()

        results = []
        for label, prob in zip(pred_labels, probs):
            label_idx = label.item()
            confidence = prob[label_idx].item()
            sentiment = self.label_map[label_idx]
            is_negative = label_idx == 0 and confidence >= NEGATIVE_THRESHOLD

            results.append({
                "sentiment": sentiment,
                "confidence": confidence,
                "is_negative": is_negative,
                "probs": {
                    "negative": prob[0].item(),
                    "neutral": prob[1].item(),
                    "positive": prob[2].item(),
                },
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

        self.session = ort.InferenceSession(model_path, providers=providers)
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


class TensorRTRunner:
    def __init__(self, engine_path: str) -> None:
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

    def __call__(self, **inputs):
        cuda_inputs = {name: tensor.contiguous() for name, tensor in inputs.items()}
        for name, tensor in cuda_inputs.items():
            if name in self.tensor_names:
                self.context.set_input_shape(name, tuple(tensor.shape))

        output_name = "logits" if "logits" in self.tensor_names else self.tensor_names[-1]
        output_shape = tuple(self.context.get_tensor_shape(output_name))
        logits = torch.empty(output_shape, dtype=torch.float32, device="cuda")

        tensors = {**cuda_inputs, output_name: logits}
        for name, tensor in tensors.items():
            if name in self.tensor_names:
                self.context.set_tensor_address(name, tensor.data_ptr())

        stream = torch.cuda.current_stream().cuda_stream
        ok = self.context.execute_async_v3(stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return logits
