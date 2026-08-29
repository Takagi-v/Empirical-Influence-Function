#!/usr/bin/env python3
"""Single-sample, multi-completion-token pure saliency overfit experiment.

This script fixes one annotated training sample, collects completion target
positions q with at least one strict-causal labeled source s < q, and optimizes
only the current saliency InfoNCE loss over all selected q.

Each logged step records only the compact diagnostics needed for this stage:
per-query |Aq|, hit@|Aq|, Lq; plus global mean_hit@|Aq| and Lsal.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    parser.add_argument("--mode", choices=["ce_only", "saliency_only", "saliency_ce"], default="saliency_only")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ce_lambda", type=float, default=1.0, help="Deprecated compatibility arg; saliency_ce now uses CE + saliency_lambda * Lsal.")
    parser.add_argument("--saliency_lambda", type=float, default=0.1, help="Weight for Lsal in saliency_ce: total = CE + saliency_lambda * Lsal")
    parser.add_argument("--alpha", type=float, default=1.5, help="Temperature tau used by current loss.py")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--saliency_loss_type", choices=["softmax", "softmax_margin"], default="softmax_margin")
    parser.add_argument("--floor_eps", type=float, default=0.0, help="Fixed saliency-space negative floor for softmax_margin.")
    parser.add_argument("--floor_logit_eps", type=float, default=None, help="Fixed logit-space negative floor for softmax_margin, e.g. -10.")
    parser.add_argument("--source_chunk_size", type=int, default=16)
    parser.add_argument("--min_sources_per_target", type=int, default=1)
    parser.add_argument("--max_targets", type=int, default=0, help="0 means use all eligible completion targets")
    parser.add_argument("--top_k", type=int, default=10, help="K for recall@K, precision@K, and AP@K diagnostics")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--use_peft", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
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
        })
    return tokens


def first_label_index(labels: list[int]) -> int | None:
    return next((i for i, value in enumerate(labels) if int(value) != IGNORE_INDEX), None)


def normalize_edge(edge: dict[str, Any]) -> tuple[int, int]:
    src = int(edge.get("src", edge.get("source", edge.get("token_i_idx", -1))))
    dst = int(edge.get("dst", edge.get("target", edge.get("token_j_idx", -1))))
    return src, dst


def collect_query_sources(
    row: dict[str, Any],
    labels: list[int],
    completion_start: int,
    seq_len: int,
    min_sources_per_target: int,
    max_targets: int,
) -> dict[int, list[int]]:
    by_q: dict[int, set[int]] = {}
    for edge in row.get("attention_edges", []):
        src, dst = normalize_edge(edge)
        if not (0 <= src < dst < seq_len):
            continue
        if dst < completion_start or int(labels[dst]) == IGNORE_INDEX:
            continue
        by_q.setdefault(dst, set()).add(src)

    filtered = {
        q: sorted(srcs)
        for q, srcs in by_q.items()
        if len(srcs) >= min_sources_per_target and q > 0
    }
    if max_targets and max_targets > 0:
        ranked = sorted(filtered.items(), key=lambda item: (-len(item[1]), item[0]))[:max_targets]
        filtered = dict(sorted(ranked, key=lambda item: item[0]))
    else:
        filtered = dict(sorted(filtered.items()))
    return filtered


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
    stem = f"multi_target_sample{args.sample_index}_{args.mode}"
    jsonl = Path(args.jsonl_path) if args.jsonl_path else run_dir / f"{stem}.jsonl"
    text = Path(args.text_log_path) if args.text_log_path else run_dir / f"{stem}.log"
    for path in (jsonl, text):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite and path.exists():
            path.unlink()
    return jsonl, text


def build_annot_pairs(query_sources: dict[int, list[int]], device: str) -> torch.Tensor:
    pairs = []
    for q, srcs in query_sources.items():
        for src in srcs:
            pairs.append([src, q])
    return torch.tensor(pairs, dtype=torch.long, device=device)


def grad_norm_by_module(model: torch.nn.Module) -> dict[str, Any]:
    groups = {
        "q_proj_lora": ["q_proj", "lora_"],
        "k_proj_lora": ["k_proj", "lora_"],
        "v_proj_lora": ["v_proj", "lora_"],
        "o_proj_lora": ["o_proj", "lora_"],
        "mlp_lora": ["gate_proj", "up_proj", "down_proj", "lora_"],
        "other_trainable": [],
    }
    sq = {key: 0.0 for key in groups}
    nonzero = {key: 0 for key in groups}
    params = {key: 0 for key in groups}
    total_sq = 0.0
    total_nonzero = 0
    total_params = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        key = "other_trainable"
        for candidate, needles in groups.items():
            if candidate == "other_trainable":
                continue
            if all(needle in name for needle in needles):
                key = candidate
                break
        params[key] += 1
        total_params += 1
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        val = float(grad.pow(2).sum().cpu())
        sq[key] += val
        total_sq += val
        if val > 0.0:
            nonzero[key] += 1
            total_nonzero += 1

    out = {
        "total_grad_norm": total_sq ** 0.5,
        "total_nonzero_param_tensors": total_nonzero,
        "total_trainable_param_tensors": total_params,
    }
    for key in groups:
        out[f"{key}_grad_norm"] = sq[key] ** 0.5
        out[f"{key}_nonzero_param_tensors"] = nonzero[key]
        out[f"{key}_trainable_param_tensors"] = params[key]
    return out


def selected_query_ce_loss(outputs: Any, input_ids: torch.Tensor, query_sources: dict[int, list[int]]) -> torch.Tensor:
    queries = torch.tensor(sorted(query_sources), dtype=torch.long, device=input_ids.device)
    logits = outputs.logits[0, queries - 1, :].float()
    labels = input_ids[0, queries]
    return F.cross_entropy(logits, labels)


def compute_query_metrics(
    C_rows: torch.Tensor,
    row_qry: torch.Tensor,
    query_sources: dict[int, list[int]],
    tokens: list[dict[str, Any]],
    *,
    alpha: float,
    eps: float,
    loss_type: str,
    floor_eps: float,
    floor_logit_eps: float | None,
    top_k: int,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    losses = []
    query_rows: list[dict[str, Any]] = []
    row_lookup = {int(q): idx for idx, q in enumerate(row_qry.detach().cpu().tolist())}
    for q, srcs in query_sources.items():
        if q not in row_lookup:
            continue
        row = C_rows[row_lookup[q], :q].float()
        labeled = torch.tensor(srcs, dtype=torch.long, device=row.device)
        tau = max(float(alpha), float(eps))
        if loss_type == "softmax":
            logits = row / tau
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            Lq = -log_probs[labeled].mean()
            labeled_probs = probs[labeled]
            floor_logit = None
        else:
            logits = torch.log(row + float(eps)) / tau
            neg_mask = torch.ones_like(row, dtype=torch.bool)
            neg_mask[labeled] = False
            if floor_logit_eps is None:
                floor = torch.log(logits.new_tensor(float(floor_eps) + float(eps))) / tau
            else:
                floor = logits.new_tensor(float(floor_logit_eps))
            floor_logit = float(floor.detach().cpu())
            neg_inf = torch.finfo(logits.dtype).min
            neg_logits = torch.maximum(logits, floor).masked_fill(~neg_mask, neg_inf)
            neg_lse = torch.logsumexp(neg_logits, dim=-1)
            pos_log_denom = torch.logaddexp(neg_lse.expand_as(logits), logits)
            Lq = (pos_log_denom[labeled] - logits[labeled]).mean()
            labeled_probs = torch.exp(logits[labeled] - pos_log_denom[labeled])
        losses.append(Lq)

        k = len(srcs)
        k_eff = min(max(1, int(top_k)), int(row.numel()))
        order = torch.argsort(row.detach(), descending=True).cpu().tolist()
        top_at_a = order[:max(1, min(k, len(order)))]
        top = order[:k_eff]
        labeled_set = set(srcs)
        hit_at_a_count = sum(1 for idx in top_at_a if idx in labeled_set)
        hit_count = sum(1 for idx in top if idx in labeled_set)
        precision_at_k = hit_count / max(1, k_eff)
        recall_at_k = hit_count / max(1, k)
        hits_so_far = 0
        precision_sum = 0.0
        for rank, idx in enumerate(top, start=1):
            if idx in labeled_set:
                hits_so_far += 1
                precision_sum += hits_so_far / rank
        ap_at_k = precision_sum / max(1, k)
        rank_lookup = {int(idx): rank for rank, idx in enumerate(order, start=1)}
        positive_ranks = [rank_lookup[src] for src in srcs if src in rank_lookup]
        pos_values = row[labeled]
        neg_mask_for_stats = torch.ones_like(row, dtype=torch.bool)
        neg_mask_for_stats[labeled] = False
        neg_values = row[neg_mask_for_stats]
        query_rows.append({
            "q": int(q),
            "target_token": tokens[q]["text"],
            "target_display": tokens[q]["display"],
            "num_labeled_sources": int(k),
            "num_causal_sources": int(row.numel()),
            "top_k": int(top_k),
            "hit_at_Aq_count": int(hit_at_a_count),
            "hit_at_Aq": float(hit_at_a_count / max(1, k)),
            "recall_at_k": float(recall_at_k),
            "precision_at_k": float(precision_at_k),
            "ap_at_k": float(ap_at_k),
            "Lq": float(Lq.detach().cpu()),
            "C_pos_mean": float(pos_values.detach().mean().cpu()),
            "C_neg_mean": float(neg_values.detach().mean().cpu()) if neg_values.numel() else 0.0,
            "ratio": float((pos_values.detach().mean() / neg_values.detach().mean().clamp_min(float(eps))).cpu()) if neg_values.numel() else 0.0,
            "mean_positive_rank": float(sum(positive_ranks) / max(1, len(positive_ranks))),
            "min_positive_rank": int(min(positive_ranks)) if positive_ranks else 0,
            "max_positive_rank": int(max(positive_ranks)) if positive_ranks else 0,
            "positive_probs": [float(x) for x in labeled_probs.detach().cpu().tolist()],
            "mean_positive_prob": float(labeled_probs.detach().mean().cpu()),
            "floor_logit": floor_logit,
        })
    if not losses:
        raise RuntimeError("No selected queries to optimize")
    return query_rows, torch.stack(losses).mean()

def format_record(record: dict[str, Any]) -> str:
    grad = record.get("grad_norms", {})
    head = (
        f"step={record['step']:04d} Lsal={record['Lsal']:.6f} "
        f"mean_hit@Aq={record['mean_hit_at_Aq']:.4f} "
        f"R@{record['top_k']}={record['mean_recall_at_k']:.4f} "
        f"mAP@{record['top_k']}={record['mean_map_at_k']:.4f} "
        f"rank={record['mean_positive_rank']:.2f} "
        f"p_pos={record['mean_positive_prob']:.4g} "
        f"grad={grad.get('total_grad_norm', 0.0):.4g} "
        f"num_q={record['num_queries']}"
    )
    if record.get("CE") is not None:
        head += f" CE={record['CE']:.6f} sal_lambda={record.get('saliency_lambda', 0.0):.4f} total={record['total_loss']:.6f}"
    lines = [head]
    if grad:
        lines.append(
            "  grad_by_module "
            f"q={grad.get('q_proj_lora_grad_norm', 0.0):.3g} "
            f"k={grad.get('k_proj_lora_grad_norm', 0.0):.3g} "
            f"v={grad.get('v_proj_lora_grad_norm', 0.0):.3g} "
            f"o={grad.get('o_proj_lora_grad_norm', 0.0):.3g} "
            f"mlp={grad.get('mlp_lora_grad_norm', 0.0):.3g} "
            f"clip_pre={record.get('clip_grad_norm_pre', 0.0):.3g}"
        )
    for qrow in record["queries"]:
        lines.append(
            f"  q=#{qrow['q']:>4} {qrow['target_display']!r} "
            f"|Aq|={qrow['num_labeled_sources']:<2}/{qrow['num_causal_sources']:<3} "
            f"hit@|Aq|={qrow['hit_at_Aq_count']}/{qrow['num_labeled_sources']} "
            f"R@{qrow['top_k']}={qrow['recall_at_k']:.3f} "
            f"AP@{qrow['top_k']}={qrow['ap_at_k']:.3f} "
            f"rank={qrow['mean_positive_rank']:.1f} "
            f"C+={qrow['C_pos_mean']:.4g} C-={qrow['C_neg_mean']:.4g} "
            f"ratio={qrow['ratio']:.3f} p+={qrow['mean_positive_prob']:.4g} "
            f"Lq={qrow['Lq']:.6f}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    jsonl_path, text_log_path = make_output_paths(args)

    row = load_jsonl_row(Path(args.data_path), args.sample_index)
    input_ids_full = [int(x) for x in row.get("input_ids", [])]
    labels_full = [int(x) for x in row.get("label", row.get("labels", []))]
    if not input_ids_full or len(input_ids_full) != len(labels_full):
        raise ValueError("row must contain same-length input_ids and label/labels")

    completion_start = first_label_index(labels_full)
    if completion_start is None:
        raise ValueError("row has no completion labels")

    model, tokenizer = load_model_and_tokenizer(args)
    tokens = decode_tokens(tokenizer, input_ids_full)
    query_sources = collect_query_sources(
        row,
        labels_full,
        completion_start,
        len(input_ids_full),
        args.min_sources_per_target,
        args.max_targets,
    )
    if not query_sources:
        raise ValueError("No eligible completion target with strict-causal annotation sources")

    input_ids = torch.tensor([input_ids_full], dtype=torch.long, device=args.device)
    attention_mask = torch.ones_like(input_ids, device=args.device)
    annot_pairs = build_annot_pairs(query_sources, args.device)

    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters. Use --use_peft or unfreeze the model.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    header = {
        "type": "header",
        "config": vars(args),
        "sample_index": args.sample_index,
        "completion_start": completion_start,
        "sequence_len": len(input_ids_full),
        "num_queries": len(query_sources),
        "num_edges": int(annot_pairs.size(0)),
        "queries": [
            {
                "q": int(q),
                "target_token": tokens[q]["text"],
                "target_display": tokens[q]["display"],
                "num_labeled_sources": len(srcs),
                "labeled_sources": srcs,
            }
            for q, srcs in query_sources.items()
        ],
        "note": (
            "Single-sample multi-target experiment. Lsal is mean_q Lq. "
            "hit@|Aq| uses top-|Aq| saliency sources for each q. "
            "recall@K, precision@K, and AP@K use strict-causal top-K sources. "
            "In saliency_ce/ce_only mode, CE is averaged over the same selected q positions."
        ),
    }
    text_log_path.write_text(json.dumps(header, ensure_ascii=False, indent=2) + "\n\n", encoding="utf-8")
    with jsonl_path.open("a", encoding="utf-8") as jf:
        jf.write(json.dumps(header, ensure_ascii=False) + "\n")

    print(f"jsonl -> {jsonl_path}")
    print(f"text  -> {text_log_path}")
    print(f"sample={args.sample_index} num_queries={len(query_sources)} num_edges={annot_pairs.size(0)}")

    from src.train.loss import (
        _annotation_rows_from_pairs,
        build_contribution_rows,
        compute_saliency_loss_from_rows,
    )

    for step in range(args.steps + 1):
        should_log = step == 0 or step == args.steps or (step % max(1, args.log_every) == 0)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        B, T, _ = outputs.hidden_states[-2].shape
        row_batch, row_qry, src_all, inv = _annotation_rows_from_pairs(
            [annot_pairs],
            B=B,
            T=T,
            device=input_ids.device,
        )
        C_rows = build_contribution_rows(
            model,
            outputs.hidden_states[-2],
            outputs.attentions[-1],
            row_batch,
            row_qry,
            source_chunk_size=max(1, int(args.source_chunk_size)),
        )
        diag = compute_saliency_loss_from_rows(
            C_rows,
            row_batch,
            row_qry,
            src_all,
            inv,
            alpha=args.alpha,
            eps=args.eps,
            floor_eps=args.floor_eps,
            floor_eps_mode="fixed",
            floor_logit_eps=args.floor_logit_eps,
            loss_type=args.saliency_loss_type,
        )
        query_rows, Lsal_recomputed = compute_query_metrics(
            C_rows,
            row_qry,
            query_sources,
            tokens,
            alpha=args.alpha,
            eps=args.eps,
            loss_type=args.saliency_loss_type,
            floor_eps=diag.floor_eps,
            floor_logit_eps=diag.floor_logit_eps if diag.floor_eps_kind == "logit" else None,
            top_k=args.top_k,
        )
        Lsal = diag.loss
        if args.mode in {"saliency_ce", "ce_only"}:
            ce_loss = selected_query_ce_loss(outputs, input_ids, query_sources)
        else:
            ce_loss = None
        if args.mode == "saliency_ce":
            total_loss = ce_loss + args.saliency_lambda * Lsal
        elif args.mode == "ce_only":
            total_loss = ce_loss
        else:
            total_loss = Lsal

        total_loss.backward()
        grad_norms = grad_norm_by_module(model)
        if args.grad_clip and args.grad_clip > 0:
            clip_norm = torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
            clip_norm_value = float(clip_norm.detach().cpu()) if isinstance(clip_norm, torch.Tensor) else float(clip_norm)
        else:
            clip_norm_value = grad_norms["total_grad_norm"]

        if should_log:
            mean_hit = sum(row["hit_at_Aq"] for row in query_rows) / len(query_rows)
            mean_recall = sum(row["recall_at_k"] for row in query_rows) / len(query_rows)
            mean_precision = sum(row["precision_at_k"] for row in query_rows) / len(query_rows)
            mean_map = sum(row["ap_at_k"] for row in query_rows) / len(query_rows)
            mean_rank = sum(row["mean_positive_rank"] for row in query_rows) / len(query_rows)
            mean_prob = sum(row["mean_positive_prob"] for row in query_rows) / len(query_rows)
            record = {
                "type": "step",
                "step": int(step),
                "mode": args.mode,
                "saliency_loss_type": args.saliency_loss_type,
                "tau": float(args.alpha),
                "floor_eps": float(diag.floor_eps),
                "floor_logit_eps": float(diag.floor_logit_eps),
                "floor_eps_kind": diag.floor_eps_kind,
                "Lsal": float(Lsal.detach().cpu()),
                "Lsal_recomputed": float(Lsal_recomputed.detach().cpu()),
                "CE": None if ce_loss is None else float(ce_loss.detach().cpu()),
                "ce_lambda": args.ce_lambda,
                "saliency_lambda": args.saliency_lambda,
                "total_loss": float(total_loss.detach().cpu()),
                "top_k": int(args.top_k),
                "mean_hit_at_Aq": float(mean_hit),
                "mean_recall_at_k": float(mean_recall),
                "mean_precision_at_k": float(mean_precision),
                "mean_map_at_k": float(mean_map),
                "mean_positive_rank": float(mean_rank),
                "mean_positive_prob": float(mean_prob),
                "clip_grad_norm_pre": clip_norm_value,
                "grad_norms": grad_norms,
                "num_queries": len(query_rows),
                "queries": query_rows,
            }
            formatted = format_record(record)
            with jsonl_path.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            with text_log_path.open("a", encoding="utf-8") as tf:
                tf.write(formatted + "\n\n")
            print(formatted + "\n", flush=True)

        if step >= args.steps:
            break
        optimizer.step()
        del outputs, C_rows, diag, query_rows, Lsal_recomputed, Lsal, total_loss
        del row_batch, row_qry, src_all, inv
        if ce_loss is not None:
            del ce_loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("done")


if __name__ == "__main__":
    main()
