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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.eval()

    encoded = tokenizer(
        ["The battery life is terrible and the screen is great."],
        ["overall"],
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        padding=True,
    )
    model_inputs = {
        name: tensor
        for name, tensor in encoded.items()
        if name in {"input_ids", "attention_mask", "token_type_ids"}
    }
    input_names = list(model_inputs.keys())

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
    print(f"Exported ONNX model to {output}")


if __name__ == "__main__":
    main()
