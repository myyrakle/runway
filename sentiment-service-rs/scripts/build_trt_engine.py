from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from ONNX.")
    parser.add_argument("--onnx", default="artifacts/model.onnx")
    parser.add_argument("--output", default="artifacts/model.plan")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=32)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--min-seq", type=int, default=8)
    parser.add_argument("--opt-seq", type=int, default=256)
    parser.add_argument("--max-seq", type=int, default=512)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument(
        "--version-compatible",
        action="store_true",
        help="Build a version-compatible engine loadable by the TensorRT LEAN runtime "
        "(libnvinfer_lean.so), and EXCLUDE the embedded lean runtime from the plan so the "
        "lean .so can be shipped externally (smaller image). May restrict tactics / cost "
        "some speed vs a standard engine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise SystemExit(
            "TensorRT is required. Run inside a TensorRT CUDA image and install "
            "the `tensorrt` extra."
        ) from exc

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    onnx_path = Path(args.onnx)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise SystemExit(f"Failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(args.workspace_gb * 1024**3),
    )
    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if args.version_compatible:
        # VERSION_COMPATIBLE makes the engine loadable by the lean runtime;
        # EXCLUDE_LEAN_RUNTIME keeps the lean runtime OUT of the plan (we ship the lean
        # .so externally instead), so the plan stays small.
        config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
        config.set_flag(trt.BuilderFlag.EXCLUDE_LEAN_RUNTIME)

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        profile.set_shape(
            tensor.name,
            (args.min_batch, args.min_seq),
            (args.opt_batch, args.opt_seq),
            (args.max_batch, args.max_seq),
        )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("TensorRT engine build failed")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(serialized)
    print(f"Built TensorRT engine at {output}")


if __name__ == "__main__":
    main()
