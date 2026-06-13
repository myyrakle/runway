from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ABSA model to ONNX.")
    parser.add_argument("--model-name", default="yangheng/deberta-v3-base-absa-v1.1")
    parser.add_argument("--output", default="artifacts/model.onnx")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export a float16 graph by tracing model.half() on CUDA (int inputs "
        "stay int; float weights/activations run in fp16). Requires a GPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.eval()

    # IMPORTANT: do NOT feed token_type_ids to the export. deberta-v3-base has
    # type_vocab_size=0, so PyTorch ignores token_type_ids at runtime (fp32/fp16
    # are correct). But tracing the model WITH a token_type_ids input collapses
    # the embedding/attention path to a constant in the exported graph, so the
    # ONNX model ignores its inputs and always predicts the same class (~"positive"
    # 0.817 here). That single bug also poisons every artifact derived from this
    # ONNX file (ort-trt, tensorrt). Exporting with only {input_ids, attention_mask}
    # — exactly what `optimum` does — produces a graph that matches fp32 logits.
    # The runner (OnnxRuntimeRunner) already filters feeds to the model's declared
    # inputs, so dropping token_type_ids here needs no runtime change.
    encoded = tokenizer(
        ["The battery life is terrible and the screen is great."],
        ["overall"],
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        padding=True,
        return_token_type_ids=False,
    )
    model_inputs = {
        name: tensor
        for name, tensor in encoded.items()
        if name in {"input_ids", "attention_mask"}
    }
    input_names = list(model_inputs.keys())

    # fp16: trace the half model on CUDA so the exported graph is natively fp16
    # and internally consistent. Post-hoc float16 conversion of DeBERTa produces
    # Cast-node type mismatches that ONNX Runtime refuses to load. Integer inputs
    # (input_ids/attention_mask) stay integer.
    if args.fp16:
        if not torch.cuda.is_available():
            raise SystemExit("--fp16 ONNX export requires a CUDA device")
        model = model.half().to("cuda")
        model_inputs = {name: tensor.to("cuda") for name, tensor in model_inputs.items()}

    dynamic_axes = {
        name: {0: "batch", 1: "sequence"}
        for name in input_names
    }
    dynamic_axes["logits"] = {0: "batch"}

    torch.onnx.export(
        model,
        tuple(model_inputs[name] for name in input_names),
        output,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Exported {'float16 ' if args.fp16 else ''}ONNX model to {output}")


if __name__ == "__main__":
    main()
