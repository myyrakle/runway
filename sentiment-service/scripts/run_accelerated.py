from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sample texts through a backend.")
    parser.add_argument(
        "--backend",
        choices=("onnx-cuda", "ort-trt", "tensorrt"),
        required=True,
    )
    parser.add_argument("--precision", default="fp16", choices=("fp32", "fp16"))
    parser.add_argument("--texts", default="sample/texts.json")
    parser.add_argument("--aspect", default="overall")
    parser.add_argument("--onnx", default="artifacts/model.onnx")
    parser.add_argument("--engine", default="artifacts/model.plan")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=16384)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["INFERENCE_BACKEND"] = args.backend
    os.environ["DEFAULT_PRECISION"] = args.precision
    os.environ["ALLOWED_PRECISIONS"] = args.precision
    os.environ["ONNX_MODEL_PATH"] = args.onnx
    os.environ["TRT_ENGINE_PATH"] = args.engine
    os.environ["INFERENCE_BATCH_SIZE"] = str(args.batch_size)
    os.environ["MAX_BATCH_TOKENS"] = str(args.max_batch_tokens)

    from app.model import DeBERTaABSA

    texts = json.loads(Path(args.texts).read_text())
    model = DeBERTaABSA(args.precision)

    start = time.perf_counter()
    results = model.analyze_batch(texts, args.aspect)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    payload = {
        "backend": args.backend,
        "precision": args.precision,
        "items": len(texts),
        "total_ms": elapsed_ms,
        "per_item_ms": elapsed_ms / len(texts) if texts else None,
        "results": results,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
