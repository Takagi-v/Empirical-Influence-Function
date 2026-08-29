#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baseline.common import read_jsonl, write_jsonl
from src.baseline.xtf import compute_xtf_score_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute XTF RI/KN/TR token scores for compact SFT data.")
    parser.add_argument("--input", required=True, help="Input compact JSONL with input_ids + label/labels.")
    parser.add_argument("--model", required=True, help="Base model path/name used to score RI/PCP/TR.")
    parser.add_argument("--output", required=True, help="Output score JSONL aligned with sample labels.")
    parser.add_argument("--report", default="", help="Optional report JSON path.")
    parser.add_argument("--device", default="", help="Optional device, e.g. cuda:0 or cpu.")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = "auto"
    if args.torch_dtype == "float16":
        dtype = torch.float16
    elif args.torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.torch_dtype == "float32":
        dtype = torch.float32

    samples = read_jsonl(args.input, max_rows=args.max_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    rows = compute_xtf_score_rows(samples, model=model, tokenizer=tokenizer, device=args.device or None)
    n = write_jsonl(args.output, rows)
    report = {
        "method": "xtf_score",
        "input": args.input,
        "model": args.model,
        "output": args.output,
        "score_rows": n,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
