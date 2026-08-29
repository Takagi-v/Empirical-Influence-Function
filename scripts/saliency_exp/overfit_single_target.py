#!/usr/bin/env python3
"""Single-sample, single-target saliency overfit experiment.

This script is intentionally narrow.  It fixes one training sample and one
completion target token, keeps only the annotation edges that point to that
target under the current loss convention, and trains for a few steps in either:

* saliency_only: total loss = saliency loss
* saliency_ce:   total loss = saliency loss + ce_lambda * target CE

Every logged step records labeled-source saliency/rank/r_qs/l_qs/p_qs,
saliency top-k source tokens and ids, hit@k, and the single-query loss L_q.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/interim/benchmark_legacy_fim/sft_data/ours_graphsignal_train_first500.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_RUN_DIR = ROOT / "runs/saliency_exp"
IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=str(DEFAULT_DATA))
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--sample_index", type=int, default=110)
    parser.add_argument("--target_idx", type=int, default=389)
    parser.add_argument("--mode", choices=["saliency_only", "saliency_ce"], default="saliency_only")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--ce_lambda", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--use_peft", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--jsonl_path", default="")
    parser.add_argument("--text_log_path", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local_files_only", type=int, default=1)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_jsonl_row(path: Path, index: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"sample_index={index} not found in {path}")


def token_display(text: str) -> str:
    if text == "\n":
        return "\\n"
    if text == "\t":
        return "\\t"
    return text.replace("\n", "\\n").replace("\t", "\\t")


def decode_tokens(tokenizer: Any, input_ids: list[int]) -> list[dict[str, Any]]:
    tokens = []
    for idx, tid in enumerate(input_ids):
        text = tokenizer.decode([int(tid)], skip_special_tokens=False)
        tokens.append({
            "idx": idx,
            "id": int(tid),
            "text": text,
            "display": token_display(text),
            "is_whitespace": text.strip() == "",
            "is_special": text.startswith("<|") and text.endswith("|>"),
        })
    return tokens


def first_label_index(labels: list[int]) -> int | None:
    return next((i for i, value in enumerate(labels) if int(value) != IGNORE_INDEX), None)


def normalize_edge(edge: dict[str, Any]) -> tuple[int, int, str]:
    src = int(edge.get("src", edge.get("source", -1)))
    dst = int(edge.get("dst", edge.get("target", -1)))
    subtype = str(edge.get("subtype", edge.get("type", "")) or "")
    return src, dst, subtype


def collect_target_edges(row: dict[str, Any], target_idx: int, seq_len: int) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    sources: set[int] = set()
    for edge in row.get("attention_edges", []):
        raw_src, raw_dst, subtype = normalize_edge(edge)
        if not (0 <= raw_src < seq_len and 0 <= raw_dst < seq_len):
            continue
        effective_src = min(raw_src, raw_dst)
        effective_tgt = max(raw_src, raw_dst)
        if effective_tgt != target_idx:
            continue
        item = {
            "raw_src": raw_src,
            "raw_dst": raw_dst,
            "effective_src": effective_src,
            "effective_tgt": effective_tgt,
            "subtype": subtype,
            "flipped": raw_dst != effective_tgt,
        }
        rows.append(item)
        sources.add(effective_src)
    return rows, sorted(sources)


def load_model_and_tokenizer(args: argparse.Namespace):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        padding_side="right",
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype_from_name(args.dtype),
        attn_implementation="eager",
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )
    model.config.use_cache = False
    model.to(args.device)

    if args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def make_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"single_target_sample{args.sample_index}_target{args.target_idx}_{args.mode}"
    jsonl = Path(args.jsonl_path) if args.jsonl_path else run_dir / f"{stem}.jsonl"
    text = Path(args.text_log_path) if args.text_log_path else run_dir / f"{stem}.log"
    for p in (jsonl, text):
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"{p} exists; pass --overwrite to replace it")
        p.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite and p.exists():
            p.unlink()
    return jsonl, text


def target_ce_loss(outputs: Any, input_ids: torch.Tensor, target_idx: int) -> torch.Tensor:
    if target_idx <= 0:
        raise ValueError("target_idx must be > 0 for causal CE")
    logits = outputs.logits[:, target_idx - 1, :].float()
    labels = input_ids[:, target_idx]
    return F.cross_entropy(logits, labels)


def rank_sources(row_scores: torch.Tensor) -> dict[int, int]:
    order = torch.argsort(row_scores.detach().float().cpu(), descending=True).tolist()
    return {int(idx): rank for rank, idx in enumerate(order, start=1)}


def build_step_record(
    *,
    step: int,
    mode: str,
    sample_index: int,
    target_idx: int,
    completion_start: int,
    tokens: list[dict[str, Any]],
    annotation_edges: list[dict[str, Any]],
    annotation_sources: list[int],
    c_matrix: torch.Tensor,
    saliency_loss_value: float,
    cbar: float,
    nbar: float,
    ratio: float,
    ce_loss_value: float | None,
    total_loss_value: float,
    top_k: int,
    temperature: float,
    eps: float,
) -> dict[str, Any]:
    row = c_matrix[0, target_idx, :target_idx].detach().float().cpu()
    ranks = rank_sources(row)
    ann_set = set(annotation_sources)

    causal_indices = list(range(target_idx))
    ann_indices = [src for src in annotation_sources if 0 <= src < target_idx]
    non_ann_indices = [idx for idx in causal_indices if idx not in ann_set]
    scores = torch.log(row.clamp_min(eps))
    logits = scores / temperature
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    ann_tensor = torch.tensor(ann_indices, dtype=torch.long)
    non_ann_tensor = torch.tensor(non_ann_indices, dtype=torch.long)
    if ann_indices:
        ann_log_probs = log_probs[ann_tensor]
        ann_probs = probs[ann_tensor]
        info_nce_loss = float((-ann_log_probs.mean()).item())
        pos_prob_mass = float(ann_probs.sum().item())
        mean_pos_prob = float(ann_probs.mean().item())
        min_pos_prob = float(ann_probs.min().item())
        max_pos_prob = float(ann_probs.max().item())
    else:
        info_nce_loss = 0.0
        pos_prob_mass = 0.0
        mean_pos_prob = 0.0
        min_pos_prob = 0.0
        max_pos_prob = 0.0
    if non_ann_indices:
        non_ann_probs = probs[non_ann_tensor]
        mean_non_pos_prob = float(non_ann_probs.mean().item())
        max_non_pos_prob = float(non_ann_probs.max().item())
    else:
        mean_non_pos_prob = 0.0
        max_non_pos_prob = 0.0

    top_indices = torch.argsort(row, descending=True).tolist()[:top_k]
    top_rows = []
    for rank, idx in enumerate(top_indices, start=1):
        tok = tokens[idx]
        top_rows.append({
            "rank": rank,
            "idx": int(idx),
            "token_id": int(tok["id"]),
            "token": tok["text"],
            "display": tok["display"],
            "sal": float(row[idx]),
            "saliency": float(row[idx]),
            "r_qs": float(scores[idx]),
            "l_qs": float(logits[idx]),
            "log_prob": float(log_probs[idx]),
            "p_qs": float(probs[idx]),
            "prob": float(probs[idx]),
            "is_labeled_source": int(idx) in ann_set,
            "is_annotation_source": int(idx) in ann_set,
        })
    ann_rows = []
    subtype_by_src: dict[int, Counter[str]] = {}
    for src in annotation_sources:
        subtype_by_src[src] = Counter(e["subtype"] for e in annotation_edges if e["effective_src"] == src)
    for src in annotation_sources:
        tok = tokens[src]
        ann_rows.append({
            "idx": int(src),
            "token_id": int(tok["id"]),
            "token": tok["text"],
            "display": tok["display"],
            "sal": float(row[src]),
            "saliency": float(row[src]),
            "rank": int(ranks.get(src, -1)),
            "r_qs": float(scores[src]),
            "l_qs": float(logits[src]),
            "log_prob": float(log_probs[src]),
            "p_qs": float(probs[src]),
            "prob": float(probs[src]),
            "subtypes": dict(subtype_by_src.get(src, Counter())),
        })
    hit_at_k = sum(1 for item in top_rows if item["is_labeled_source"])
    record = {
        "step": step,
        "mode": mode,
        "sample_index": sample_index,
        "target_idx": target_idx,
        "target_token": tokens[target_idx],
        "completion_start": completion_start,
        "ce_query_idx_for_predicting_target": target_idx - 1,
        "saliency_query_idx_used_by_current_loss": target_idx,
        "num_annotation_sources": len(annotation_sources),
        "num_annotation_edges_for_target": len(annotation_edges),
        "hit_at_top_k": hit_at_k,
        "top_k": top_k,
        "Cbar": cbar,
        "Nbar": nbar,
        "ratio": ratio,
        "temperature": temperature,
        "loss_q": info_nce_loss,
        "info_nce_loss_recomputed_for_target": info_nce_loss,
        "positive_probability_mass": pos_prob_mass,
        "mean_annotation_probability": mean_pos_prob,
        "min_annotation_probability": min_pos_prob,
        "max_annotation_probability": max_pos_prob,
        "mean_non_annotation_probability": mean_non_pos_prob,
        "max_non_annotation_probability": max_non_pos_prob,
        "saliency_loss": saliency_loss_value,
        "loss_q_from_training_loss": saliency_loss_value,
        "target_ce_loss": ce_loss_value,
        "total_loss": total_loss_value,
        "labeled_tokens": ann_rows,
        "annotation_sources": ann_rows,
        "sal_top_tokens": top_rows,
        "top_saliency_sources": top_rows,
    }
    return record


def format_record(record: dict[str, Any]) -> str:
    lines = []
    tgt = record["target_token"]
    lines.append(
        f"step={record['step']:04d} mode={record['mode']} "
        f"target=#{record['target_idx']} {tgt['display']!r} "
        f"sal_loss={record['saliency_loss']:.6f} "
        f"Cbar={record['Cbar']:.6g} Nbar={record['Nbar']:.6g} ratio={record['ratio']:.4f} "
        f"pos_mass={record['positive_probability_mass']:.4f} "
        f"L_q={record['loss_q']:.6f} "
        f"hit@{record['top_k']}={record['hit_at_top_k']}/{record['num_annotation_sources']} "
        f"total={record['total_loss']:.6f}"
    )
    if record["target_ce_loss"] is not None:
        lines[-1] += f" target_ce={record['target_ce_loss']:.6f}"
    lines.append(
        f"  indices: completion_start={record['completion_start']} "
        f"ce_query={record['ce_query_idx_for_predicting_target']} "
        f"saliency_query={record['saliency_query_idx_used_by_current_loss']}"
    )
    lines.append("  annotation sources:")
    for src in record["annotation_sources"]:
        lines.append(
            f"    rank={src['rank']:>4} #{src['idx']:>4} {src['display']!r} "
            f"sal={src['sal']:.6g} r_qs={src['r_qs']:.4g} l_qs={src['l_qs']:.4g} p_qs={src['p_qs']:.4g} "
            f"subtypes={src['subtypes']}"
        )
    lines.append("  top saliency sources:")
    for item in record["top_saliency_sources"]:
        mark = "*" if item["is_annotation_source"] else " "
        lines.append(
            f"   {mark}rank={item['rank']:>2} #{item['idx']:>4} {item['display']!r} "
            f"id={item['token_id']} sal={item['sal']:.6g} p_qs={item['p_qs']:.4g}"
        )
    return "\n".join(lines)


def compute_forward_diagnostics(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    annot_pairs: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[Any, torch.Tensor, Any, torch.Tensor]:
    from src.train.loss import build_contribution_matrix, compute_saliency_loss

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    c_matrix = build_contribution_matrix(model, outputs.hidden_states[-2], outputs.attentions[-1])
    diag = compute_saliency_loss(c_matrix, [annot_pairs], alpha=args.alpha, eps=args.eps)
    ce = target_ce_loss(outputs, input_ids, args.target_idx)
    return outputs, c_matrix, diag, ce


def main() -> None:
    args = parse_args()
    jsonl_path, text_log_path = make_output_paths(args)

    row = load_jsonl_row(Path(args.data_path), args.sample_index)
    input_ids_full = [int(x) for x in row.get("input_ids", [])]
    labels_full = [int(x) for x in row.get("label", row.get("labels", []))]
    if not input_ids_full or not labels_full:
        raise ValueError("row must contain input_ids and label/labels")
    if args.target_idx >= len(input_ids_full):
        raise ValueError(f"target_idx={args.target_idx} outside sequence length {len(input_ids_full)}")

    completion_start = first_label_index(labels_full)
    if completion_start is None:
        raise ValueError("row has no completion labels")
    if args.target_idx < completion_start:
        raise ValueError(f"target_idx={args.target_idx} is before completion_start={completion_start}")

    annotation_edges, annotation_sources = collect_target_edges(row, args.target_idx, len(input_ids_full))
    if not annotation_sources:
        raise ValueError(f"No effective annotation sources found for target_idx={args.target_idx}")

    model, tokenizer = load_model_and_tokenizer(args)
    tokens = decode_tokens(tokenizer, input_ids_full)

    truncated_ids = input_ids_full[: args.target_idx + 1]
    input_ids = torch.tensor([truncated_ids], dtype=torch.long, device=args.device)
    attention_mask = torch.ones_like(input_ids, device=args.device)
    annot_pairs = torch.tensor(
        [[int(src), int(args.target_idx)] for src in annotation_sources],
        dtype=torch.long,
        device=args.device,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters. Use --use_peft or unfreeze the model.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    header = {
        "config": vars(args),
        "sample_index": args.sample_index,
        "target_idx": args.target_idx,
        "target_token": tokens[args.target_idx],
        "completion_start": completion_start,
        "sequence_len_full": len(input_ids_full),
        "sequence_len_used": len(truncated_ids),
        "annotation_edges_for_target": annotation_edges,
        "annotation_sources": [
            {
                "idx": src,
                "token": tokens[src]["text"],
                "display": tokens[src]["display"],
            }
            for src in annotation_sources
        ],
        "note": (
            "Current loss convention uses effective edge min(src,dst)->max(src,dst); "
            "saliency query index is target_idx. New loss is multi-positive InfoNCE: "
            "logits=log(C_qs+eps)/alpha, so alpha is temperature, not hinge margin."
        ),
    }
    text_log_path.write_text(json.dumps(header, ensure_ascii=False, indent=2) + "\n\n", encoding="utf-8")
    with jsonl_path.open("a", encoding="utf-8") as jf:
        jf.write(json.dumps({"type": "header", **header}, ensure_ascii=False) + "\n")

    print(f"jsonl -> {jsonl_path}")
    print(f"text  -> {text_log_path}")
    print(
        f"sample={args.sample_index} target=#{args.target_idx} {tokens[args.target_idx]['display']!r} "
        f"annotation_sources={len(annotation_sources)} mode={args.mode}"
    )

    for step in range(args.steps + 1):
        should_log = step == 0 or step == args.steps or (step % max(1, args.log_every) == 0)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs, c_matrix, diag, ce = compute_forward_diagnostics(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            annot_pairs=annot_pairs,
            args=args,
        )
        if args.mode == "saliency_only":
            total_loss = diag.loss
            ce_for_log: float | None = None
        else:
            total_loss = diag.loss + args.ce_lambda * ce
            ce_for_log = float(ce.detach().cpu())

        if should_log:
            record = build_step_record(
                step=step,
                mode=args.mode,
                sample_index=args.sample_index,
                target_idx=args.target_idx,
                completion_start=completion_start,
                tokens=tokens,
                annotation_edges=annotation_edges,
                annotation_sources=annotation_sources,
                c_matrix=c_matrix,
                saliency_loss_value=float(diag.loss.detach().cpu()),
                cbar=diag.avg_C,
                nbar=diag.avg_N,
                ratio=diag.avg_ratio,
                ce_loss_value=ce_for_log,
                total_loss_value=float(total_loss.detach().cpu()),
                top_k=args.top_k,
                temperature=args.alpha,
                eps=args.eps,
            )
            with jsonl_path.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps({"type": "step", **record}, ensure_ascii=False) + "\n")
            formatted = format_record(record)
            with text_log_path.open("a", encoding="utf-8") as tf:
                tf.write(formatted + "\n\n")
            print(formatted.splitlines()[0], flush=True)

        if step >= args.steps:
            break
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        del outputs, c_matrix, diag, ce, total_loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("done")


if __name__ == "__main__":
    main()
