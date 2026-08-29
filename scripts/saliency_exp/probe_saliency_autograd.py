#!/usr/bin/env python3
"""Probe saliency-loss autograd health on one annotated sample.

This diagnostic does not train or save a model. It runs one forward pass with
real eager attentions, computes CE and the current saliency loss, and reports:

* whether saliency tensors are attached to the graph;
* CE vs saliency gradient norms and cosine over trainable parameters;
* LoRA gradient norms split by q/k/v/o/mlp modules;
* last-layer PEFT wrapper metadata, especially whether manual `.weight` reads
  are likely seeing base weights rather than LoRA-effective weights.
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

from src.train.loss import (  # noqa: E402
    _annotation_rows_from_pairs,
    _unwrap_to_decoder_stack,
    build_contribution_rows,
    compute_saliency_loss_from_rows,
)

DEFAULT_DATA = ROOT / "data/go_single/train_data/go_single_train_v2_graphsignal_500_compact.json"
DEFAULT_MODEL = ROOT / "/mnt/nvme0n1/wenhao/models/Empirical-Influence-Function/Qwen2.5-Coder-7B-Instruct"
IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=str(DEFAULT_DATA))
    parser.add_argument("--model_name_or_path", default=str(DEFAULT_MODEL))
    parser.add_argument("--sample_index", type=int, default=479)
    parser.add_argument("--target_idx", type=int, default=0, help="If >0, keep only annotation edges targeting this q.")
    parser.add_argument("--max_targets", type=int, default=1, help="0 means all eligible annotated completion targets.")
    parser.add_argument("--min_sources_per_target", type=int, default=1)
    parser.add_argument("--saliency_loss_type", choices=["softmax", "softmax_margin"], default="softmax_margin")
    parser.add_argument("--alpha", type=float, default=0.1, help="Temperature tau.")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--floor_eps", type=float, default=0.0)
    parser.add_argument("--floor_logit_eps", type=float, default=-10.0)
    parser.add_argument("--saliency_lambda", type=float, default=0.1)
    parser.add_argument("--source_chunk_size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--use_peft", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--local_files_only", type=int, default=1)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_jsonl_row(path: Path, index: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"sample_index={index} not found in {path}")


def first_label_index(labels: list[int]) -> int | None:
    return next((i for i, value in enumerate(labels) if int(value) != IGNORE_INDEX), None)


def normalize_edge(edge: dict[str, Any]) -> tuple[int, int, str]:
    src = int(edge.get("src", edge.get("source", edge.get("token_i_idx", -1))))
    dst = int(edge.get("dst", edge.get("target", edge.get("token_j_idx", -1))))
    subtype = str(edge.get("subtype", edge.get("type", "")) or "")
    return src, dst, subtype


def collect_query_sources(
    row: dict[str, Any],
    labels: list[int],
    completion_start: int,
    seq_len: int,
    *,
    min_sources_per_target: int,
    max_targets: int,
    target_idx: int,
) -> dict[int, list[int]]:
    by_q: dict[int, set[int]] = {}
    for edge in row.get("attention_edges", []):
        src, dst, _ = normalize_edge(edge)
        if not (0 <= src < dst < seq_len):
            continue
        if dst < completion_start or int(labels[dst]) == IGNORE_INDEX:
            continue
        if target_idx > 0 and dst != target_idx:
            continue
        by_q.setdefault(dst, set()).add(src)

    filtered = {
        q: sorted(srcs)
        for q, srcs in by_q.items()
        if len(srcs) >= min_sources_per_target and q > 0
    }
    if max_targets and max_targets > 0 and target_idx <= 0:
        ranked = sorted(filtered.items(), key=lambda item: (-len(item[1]), item[0]))[:max_targets]
        filtered = dict(sorted(ranked, key=lambda item: item[0]))
    else:
        filtered = dict(sorted(filtered.items()))
    return filtered


def build_annot_pairs(query_sources: dict[int, list[int]], device: torch.device) -> torch.Tensor:
    pairs = [[src, q] for q, srcs in query_sources.items() for src in srcs]
    return torch.tensor(pairs, dtype=torch.long, device=device)


def token_display(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return text.replace("\n", "\\n").replace("\t", "\\t")


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


def trainable_named_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def group_for_name(name: str) -> str:
    if "q_proj" in name and "lora_" in name:
        return "q_proj_lora"
    if "k_proj" in name and "lora_" in name:
        return "k_proj_lora"
    if "v_proj" in name and "lora_" in name:
        return "v_proj_lora"
    if "o_proj" in name and "lora_" in name:
        return "o_proj_lora"
    if any(x in name for x in ("gate_proj", "up_proj", "down_proj")) and "lora_" in name:
        return "mlp_lora"
    return "other_trainable"


def grad_group_stats(names: list[str], grads: tuple[torch.Tensor | None, ...]) -> dict[str, Any]:
    groups = ["q_proj_lora", "k_proj_lora", "v_proj_lora", "o_proj_lora", "mlp_lora", "other_trainable"]
    sq = {key: 0.0 for key in groups}
    params = {key: 0 for key in groups}
    nonzero = {key: 0 for key in groups}
    missing = {key: 0 for key in groups}
    total_sq = 0.0
    total_nonzero = 0
    total_missing = 0
    for name, grad in zip(names, grads):
        key = group_for_name(name)
        params[key] += 1
        if grad is None:
            missing[key] += 1
            total_missing += 1
            continue
        val = float(grad.detach().float().pow(2).sum().cpu())
        sq[key] += val
        total_sq += val
        if val > 0.0:
            nonzero[key] += 1
            total_nonzero += 1
    out = {
        "total_grad_norm": total_sq ** 0.5,
        "total_nonzero_param_tensors": total_nonzero,
        "total_missing_param_tensors": total_missing,
        "total_trainable_param_tensors": len(names),
    }
    for key in groups:
        out[key] = {
            "grad_norm": sq[key] ** 0.5,
            "nonzero_param_tensors": nonzero[key],
            "missing_param_tensors": missing[key],
            "trainable_param_tensors": params[key],
        }
    return out


def grad_cosine_and_total(
    ce_grads: tuple[torch.Tensor | None, ...],
    sal_grads: tuple[torch.Tensor | None, ...],
    *,
    gamma: float,
    device: torch.device,
) -> dict[str, float]:
    ce_sq = torch.zeros((), device=device, dtype=torch.float32)
    sal_sq = torch.zeros((), device=device, dtype=torch.float32)
    total_sq = torch.zeros((), device=device, dtype=torch.float32)
    dot = torch.zeros((), device=device, dtype=torch.float32)
    for ce_g, sal_g in zip(ce_grads, sal_grads):
        ce_f = None if ce_g is None else ce_g.detach().float()
        sal_f = None if sal_g is None else sal_g.detach().float()
        if ce_f is not None:
            ce_sq = ce_sq + ce_f.pow(2).sum()
        if sal_f is not None:
            sal_sq = sal_sq + sal_f.pow(2).sum()
        if ce_f is not None and sal_f is not None:
            dot = dot + (ce_f * sal_f).sum()
            total_sq = total_sq + (ce_f + float(gamma) * sal_f).pow(2).sum()
        elif ce_f is not None:
            total_sq = total_sq + ce_f.pow(2).sum()
        elif sal_f is not None:
            total_sq = total_sq + (float(gamma) * sal_f).pow(2).sum()
    ce_norm = torch.sqrt(ce_sq).item()
    sal_norm = torch.sqrt(sal_sq).item()
    denom = ce_norm * sal_norm
    return {
        "ce_grad_norm": float(ce_norm),
        "saliency_grad_norm": float(sal_norm),
        "ce_saliency_grad_cosine": float(dot.item() / denom) if denom > 0 else 0.0,
        "ce_plus_lambda_saliency_grad_norm": float(torch.sqrt(total_sq).item()),
    }


def selected_query_ce_loss(outputs: Any, input_ids: torch.Tensor, query_sources: dict[int, list[int]]) -> torch.Tensor:
    queries = torch.tensor(sorted(query_sources), dtype=torch.long, device=input_ids.device)
    logits = outputs.logits[0, queries - 1, :].float()
    labels = input_ids[0, queries]
    return F.cross_entropy(logits, labels)


def query_rank_stats(
    C_rows: torch.Tensor,
    row_qry: torch.Tensor,
    query_sources: dict[int, list[int]],
    input_ids: list[int],
    tokenizer: Any,
    eps: float,
) -> list[dict[str, Any]]:
    out = []
    row_lookup = {int(q): idx for idx, q in enumerate(row_qry.detach().cpu().tolist())}
    for q, srcs in query_sources.items():
        if q not in row_lookup:
            continue
        row = C_rows[row_lookup[q], :q].detach().float()
        labeled = torch.tensor(srcs, dtype=torch.long, device=row.device)
        order = torch.argsort(row, descending=True).cpu().tolist()
        rank_lookup = {int(idx): rank for rank, idx in enumerate(order, start=1)}
        ranks = [rank_lookup[src] for src in srcs if src in rank_lookup]
        pos = row[labeled]
        neg_mask = torch.ones_like(row, dtype=torch.bool)
        neg_mask[labeled] = False
        neg = row[neg_mask]
        out.append({
            "query": int(q),
            "target_token": token_display(tokenizer, input_ids[q]),
            "num_annotation_sources": len(srcs),
            "num_causal_sources": int(row.numel()),
            "C_pos_mean": float(pos.mean().cpu()),
            "C_neg_mean": float(neg.mean().cpu()) if neg.numel() else 0.0,
            "ratio": float((pos.mean() / neg.mean().clamp_min(float(eps))).cpu()) if neg.numel() else 0.0,
            "mean_positive_rank": float(sum(ranks) / max(1, len(ranks))),
            "min_positive_rank": int(min(ranks)) if ranks else 0,
            "max_positive_rank": int(max(ranks)) if ranks else 0,
        })
    return out


def inspect_last_layer_modules(model: torch.nn.Module) -> dict[str, Any]:
    decoder = _unwrap_to_decoder_stack(model)
    layer = decoder.layers[-1]
    modules = {
        "q_proj": layer.self_attn.q_proj,
        "k_proj": layer.self_attn.k_proj,
        "v_proj": layer.self_attn.v_proj,
        "o_proj": layer.self_attn.o_proj,
        "gate_proj": layer.mlp.gate_proj,
        "up_proj": layer.mlp.up_proj,
        "down_proj": layer.mlp.down_proj,
    }
    report = {}
    for name, module in modules.items():
        lora_A = getattr(module, "lora_A", None)
        lora_keys = list(lora_A.keys()) if hasattr(lora_A, "keys") else []
        base_layer = getattr(module, "base_layer", None)
        weight = getattr(module, "weight", None)
        base_weight = getattr(base_layer, "weight", None) if base_layer is not None else None
        report[name] = {
            "class": type(module).__name__,
            "has_base_layer": base_layer is not None,
            "has_lora_A": bool(lora_keys),
            "lora_adapter_keys": lora_keys,
            "weight_requires_grad": None if weight is None else bool(weight.requires_grad),
            "base_weight_requires_grad": None if base_weight is None else bool(base_weight.requires_grad),
            "manual_weight_reads_likely_base_only": bool(base_layer is not None and lora_keys),
        }
    return report


def main() -> None:
    args = parse_args()
    row = load_jsonl_row(Path(args.data_path), args.sample_index)
    input_ids_full = [int(x) for x in row.get("input_ids", [])]
    labels_full = [int(x) for x in row.get("label", row.get("labels", []))]
    if not input_ids_full or len(input_ids_full) != len(labels_full):
        raise ValueError("row must contain same-length input_ids and label/labels")
    completion_start = first_label_index(labels_full)
    if completion_start is None:
        raise ValueError("row has no completion labels")

    model, tokenizer = load_model_and_tokenizer(args)
    model.train()
    device = torch.device(args.device)
    query_sources = collect_query_sources(
        row,
        labels_full,
        completion_start,
        len(input_ids_full),
        min_sources_per_target=args.min_sources_per_target,
        max_targets=args.max_targets,
        target_idx=args.target_idx,
    )
    if not query_sources:
        raise ValueError("No eligible completion target with strict-causal annotation sources")

    input_ids = torch.tensor([input_ids_full], dtype=torch.long, device=device)
    labels = torch.tensor([labels_full], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    annot_pairs = build_annot_pairs(query_sources, device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    ce_loss = outputs.loss.mean() if outputs.loss.dim() > 0 else outputs.loss
    selected_ce = selected_query_ce_loss(outputs, input_ids, query_sources)
    B, T, _ = outputs.hidden_states[-2].shape
    row_batch, row_qry, src_all, inv = _annotation_rows_from_pairs([annot_pairs], B=B, T=T, device=device)
    C_rows = build_contribution_rows(
        model,
        outputs.hidden_states[-2],
        outputs.attentions[-1],
        row_batch,
        row_qry,
        source_chunk_size=max(1, int(args.source_chunk_size)),
    )
    sal_diag = compute_saliency_loss_from_rows(
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

    named = trainable_named_params(model)
    names = [name for name, _ in named]
    params = [param for _, param in named]
    ce_grads = torch.autograd.grad(ce_loss, params, retain_graph=True, allow_unused=True)
    selected_ce_grads = torch.autograd.grad(selected_ce, params, retain_graph=True, allow_unused=True)
    sal_grads = torch.autograd.grad(sal_diag.loss, params, retain_graph=False, allow_unused=True)

    report = {
        "config": vars(args),
        "sample_index": int(args.sample_index),
        "completion_start": int(completion_start),
        "sequence_len": len(input_ids_full),
        "num_queries": len(query_sources),
        "num_edges": int(annot_pairs.size(0)),
        "losses": {
            "ce_loss_full_labels": float(ce_loss.detach().cpu()),
            "ce_loss_selected_queries": float(selected_ce.detach().cpu()),
            "saliency_loss": float(sal_diag.loss.detach().cpu()),
            "total_ce_plus_lambda_saliency": float((ce_loss + float(args.saliency_lambda) * sal_diag.loss).detach().cpu()),
        },
        "autograd_health": {
            "saliency_loss_requires_grad": bool(sal_diag.loss.requires_grad),
            "ce_loss_requires_grad": bool(ce_loss.requires_grad),
            "attentions_last_requires_grad": bool(outputs.attentions[-1].requires_grad),
            "hidden_states_last_layer_input_requires_grad": bool(outputs.hidden_states[-2].requires_grad),
            "C_rows_requires_grad": bool(C_rows.requires_grad),
            "num_trainable_param_tensors": len(params),
        },
        "saliency_diag": {
            "loss_type": sal_diag.loss_type,
            "Cbar": sal_diag.avg_C,
            "Nbar": sal_diag.avg_N,
            "ratio": sal_diag.avg_ratio,
            "floor_eps": sal_diag.floor_eps,
            "floor_logit_eps": sal_diag.floor_logit_eps,
            "floor_eps_kind": sal_diag.floor_eps_kind,
            "n_queries": sal_diag.n_queries,
        },
        "gradients": {
            "full_ce": grad_group_stats(names, ce_grads),
            "selected_query_ce": grad_group_stats(names, selected_ce_grads),
            "saliency": grad_group_stats(names, sal_grads),
            "full_ce_vs_saliency": grad_cosine_and_total(ce_grads, sal_grads, gamma=args.saliency_lambda, device=device),
            "selected_ce_vs_saliency": grad_cosine_and_total(selected_ce_grads, sal_grads, gamma=args.saliency_lambda, device=device),
        },
        "last_layer_module_inspection": inspect_last_layer_modules(model),
        "query_rank_stats": query_rank_stats(C_rows, row_qry, query_sources, input_ids_full, tokenizer, args.eps),
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        if out.exists() and not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
