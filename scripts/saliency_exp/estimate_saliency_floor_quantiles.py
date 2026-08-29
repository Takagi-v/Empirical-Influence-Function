#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftConfig, PeftModel
except Exception:  # pragma: no cover
    PeftConfig = None
    PeftModel = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRAIN_DIR = ROOT / "src" / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from dataset import AnnotatedSFTDataset, DataCollatorForAnnotatedSFT  # noqa: E402
from loss import _annotation_rows_from_pairs, build_contribution_rows  # noqa: E402


def parse_quantiles(text: str) -> list[float]:
    qs = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value > 1.0:
            value = value / 100.0
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"quantile must be in [0,1] or [0,100], got {part!r}")
        qs.append(value)
    if not qs:
        raise ValueError("at least one quantile is required")
    return qs


def resolve_checkpoint(path_or_repo: str) -> str:
    p = Path(path_or_repo)
    if not p.exists():
        return path_or_repo
    if (p / "adapter_config.json").exists() or (p / "config.json").exists():
        return str(p)
    checkpoints: list[tuple[int, Path]] = []
    for child in p.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            step = child.name.rsplit("-", 1)[-1]
            if step.isdigit():
                checkpoints.append((int(step), child))
    if checkpoints:
        return str(sorted(checkpoints)[-1][1])
    return str(p)


def load_model_and_tokenizer(model_path: str, *, dtype: torch.dtype, local_files_only: bool):
    resolved = resolve_checkpoint(model_path)
    tokenizer_path = resolved
    if Path(resolved).exists() and (Path(resolved) / "adapter_config.json").exists():
        if PeftConfig is None or PeftModel is None:
            raise RuntimeError("PEFT model detected but peft is not importable")
        peft_config = PeftConfig.from_pretrained(resolved, local_files_only=local_files_only)
        tokenizer_path = resolved
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
            pad_token="<|endoftext|>",
            eos_token="<|im_end|>",
            padding_side="right",
        )
        base = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model = PeftModel.from_pretrained(base, resolved, is_trainable=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
            pad_token="<|endoftext|>",
            eos_token="<|im_end|>",
            padding_side="right",
        )
        model = AutoModelForCausalLM.from_pretrained(
            resolved,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, resolved


def maybe_cap(values: torch.Tensor, max_values: int, *, generator: torch.Generator) -> torch.Tensor:
    if max_values <= 0 or values.numel() <= max_values:
        return values
    idx = torch.randperm(values.numel(), generator=generator)[:max_values]
    return values[idx]


def quantile_dict(values: torch.Tensor, quantiles: list[float]) -> dict[str, float]:
    if values.numel() == 0:
        return {f"p{int(q * 100):02d}": 0.0 for q in quantiles}
    vals = values.float()
    qs = torch.tensor(quantiles, dtype=torch.float32)
    out = torch.quantile(vals, qs)
    return {f"p{int(q * 100):02d}": float(v) for q, v in zip(quantiles, out)}


def summarize_chunks(chunks: list[torch.Tensor], quantiles: list[float], max_values: int, seed: int) -> dict[str, Any]:
    if not chunks:
        values = torch.empty(0, dtype=torch.float32)
    else:
        values = torch.cat(chunks).float()
    generator = torch.Generator().manual_seed(seed)
    sampled = maybe_cap(values, max_values, generator=generator)
    summary: dict[str, Any] = {
        "n_values": int(values.numel()),
        "n_values_used": int(sampled.numel()),
        "sampled": bool(max_values > 0 and values.numel() > max_values),
        "mean": float(sampled.mean()) if sampled.numel() else 0.0,
        "std": float(sampled.std(unbiased=False)) if sampled.numel() else 0.0,
        "min": float(sampled.min()) if sampled.numel() else 0.0,
        "max": float(sampled.max()) if sampled.numel() else 0.0,
    }
    summary.update(quantile_dict(sampled, quantiles))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate global saliency quantiles on annotated query rows with teacher forcing."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", default="outputs/saliency_exp/ce500_saliency_floor_quantiles.json")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--quantiles", default="0.60,0.70,0.75,0.90,0.95")
    parser.add_argument(
        "--max_values_per_bucket",
        type=int,
        default=0,
        help="If >0, randomly subsample this many saliency values per bucket before quantile computation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--source_chunk_size", type=int, default=16)
    args = parser.parse_args()

    quantiles = parse_quantiles(args.quantiles)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    model, tokenizer, resolved_model = load_model_and_tokenizer(
        args.model_path,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.eval()

    dataset = AnnotatedSFTDataset(args.data_path, tokenizer=tokenizer, max_len=args.max_len)
    if args.max_samples and args.max_samples > 0:
        dataset.items = dataset.items[: args.max_samples]
    collator = DataCollatorForAnnotatedSFT(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    all_causal_chunks: list[torch.Tensor] = []
    labeled_chunks: list[torch.Tensor] = []
    non_labeled_chunks: list[torch.Tensor] = []
    n_samples_seen = 0
    n_samples_with_edges = 0
    n_query_rows = 0
    n_annotation_edges = 0

    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            annot_pairs = batch["annot_pairs"]
            n_samples_seen += input_ids.size(0)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
            )
            attn_last = outputs.attentions[-1]
            last_hidden_in = outputs.hidden_states[-2]
            B, T, _ = last_hidden_in.shape

            row_batch, row_qry, src_all, inv = _annotation_rows_from_pairs(
                annot_pairs,
                B=B,
                T=T,
                device=device,
            )
            if row_qry.numel() == 0:
                continue

            C_rows = build_contribution_rows(
                model,
                last_hidden_in,
                attn_last,
                row_batch,
                row_qry,
                source_chunk_size=args.source_chunk_size,
            ).float()

            Q = C_rows.size(0)
            src_idx = torch.arange(T, device=device).unsqueeze(0)
            causal_mask = src_idx < row_qry.unsqueeze(1)

            A_adj = torch.zeros(Q, T, device=device, dtype=C_rows.dtype)
            A_adj.index_put_(
                (inv.to(torch.long), src_all.to(torch.long)),
                torch.ones_like(src_all, dtype=C_rows.dtype),
                accumulate=True,
            )
            labeled_mask = (A_adj > 0) & causal_mask
            non_labeled_mask = causal_mask & (A_adj <= 0)

            all_causal_chunks.append(C_rows[causal_mask].detach().cpu())
            labeled_chunks.append(C_rows[labeled_mask].detach().cpu())
            non_labeled_chunks.append(C_rows[non_labeled_mask].detach().cpu())

            n_query_rows += int(Q)
            n_annotation_edges += int(src_all.numel())
            n_samples_with_edges += int(row_batch.unique().numel())

            if (step + 1) % 10 == 0:
                print(
                    f"processed_batches={step + 1} samples={n_samples_seen} "
                    f"query_rows={n_query_rows} all_causal_values={sum(x.numel() for x in all_causal_chunks)}",
                    flush=True,
                )

    result = {
        "model_path": args.model_path,
        "resolved_model_path": resolved_model,
        "data_path": args.data_path,
        "max_samples": args.max_samples,
        "n_samples_seen": n_samples_seen,
        "n_samples_with_annotation_queries": n_samples_with_edges,
        "n_annotation_query_rows": n_query_rows,
        "n_annotation_edges": n_annotation_edges,
        "quantiles": quantiles,
        "scope": {
            "primary": "all_causal",
            "description": "For query rows with at least one strict-causal annotation edge, collect C[q,s] for all causal sources s<q, then compute global quantiles.",
        },
        "all_causal": summarize_chunks(all_causal_chunks, quantiles, args.max_values_per_bucket, args.seed),
        "labeled": summarize_chunks(labeled_chunks, quantiles, args.max_values_per_bucket, args.seed),
        "non_labeled": summarize_chunks(non_labeled_chunks, quantiles, args.max_values_per_bucket, args.seed),
    }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["all_causal"], ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
