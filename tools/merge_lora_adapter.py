#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a full CausalLM checkpoint for EIF.

Fully offline: loads the base model from --base-model, then attaches the local
adapter. Does NOT use AutoPeftModelForCausalLM (that path re-reads
adapter_config.base_model_name_or_path, often a relative string like
`models/Qwen2.5-Coder-7B-Instruct`, which HuggingFace treats as a Hub repo id
and tries to download).

Example:
  python tools/merge_lora_adapter.py \\
    --base-model /path/to/Qwen2.5-Coder-7B-Instruct \\
    --adapter    /path/to/outputs/go_single/models/ce_only \\
    --output     /path/to/outputs/go_single/merged/ce_only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Force offline before importing huggingface / transformers / peft.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _require_local_dir(path: Path, label: str, required_files: list[str]) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    missing = [name for name in required_files if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{label} missing {missing} under {path}")
    return path.resolve()


def merge_one(
    base_model: str,
    adapter: str,
    output: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
) -> None:
    base_path = _require_local_dir(
        Path(base_model),
        "base model",
        ["config.json"],
    )
    adapter_path = _require_local_dir(
        Path(adapter),
        "adapter",
        ["adapter_config.json"],
    )
    out_path = Path(output).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if torch_dtype not in dtype_map:
        raise ValueError(f"unsupported --torch-dtype {torch_dtype!r}; choose from {sorted(dtype_map)}")

    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    recorded_base = getattr(peft_config, "base_model_name_or_path", None)
    print(f"[info] base   = {base_path}", flush=True)
    print(f"[info] adapter= {adapter_path}", flush=True)
    print(f"[info] output = {out_path}", flush=True)
    print(f"[info] dtype  = {torch_dtype}, device_map={device_map}", flush=True)
    print(f"[info] adapter_config.base_model_name_or_path (ignored for load) = {recorded_base!r}", flush=True)

    print("[info] loading base model from local disk (offline)...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        torch_dtype=dtype_map[torch_dtype],
        device_map=device_map,
        local_files_only=True,
        trust_remote_code=True,
    )

    print("[info] attaching local LoRA adapter...", flush=True)
    peft_model = PeftModel.from_pretrained(
        base,
        str(adapter_path),
        local_files_only=True,
    )

    print("[info] merging LoRA into base weights...", flush=True)
    merged = peft_model.merge_and_unload(progressbar=True)

    print(f"[info] saving merged model -> {out_path}", flush=True)
    merged.save_pretrained(str(out_path), safe_serialization=True)

    print("[info] saving tokenizer from base model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(str(out_path))

    meta = {
        "base_model": str(base_path),
        "adapter": str(adapter_path),
        "torch_dtype": torch_dtype,
        "adapter_recorded_base": recorded_base,
        "peft_type": getattr(peft_config, "peft_type", None),
        "r": getattr(peft_config, "r", None),
        "lora_alpha": getattr(peft_config, "lora_alpha", None),
        "offline": True,
    }
    (out_path / "merge_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[done] merge complete.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into a full CausalLM checkpoint (offline).")
    parser.add_argument("--base-model", required=True, help="Local base HF model directory.")
    parser.add_argument("--adapter", required=True, help="Local LoRA adapter directory.")
    parser.add_argument("--output", required=True, help="Output directory for merged full model.")
    parser.add_argument("--torch-dtype", default="bfloat16", help="bfloat16|float16|float32")
    parser.add_argument(
        "--device-map",
        default="auto",
        help='device_map for loading (default "auto"). Use "cpu" if GPU RAM is tight.',
    )
    args = parser.parse_args()

    try:
        merge_one(
            base_model=args.base_model,
            adapter=args.adapter,
            output=args.output,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
