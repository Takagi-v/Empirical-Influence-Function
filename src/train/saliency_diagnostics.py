from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from loss import build_contribution_rows, flatten_annot_pairs


@dataclass
class SaliencyDetailDiagnostics:
    avg_C: float
    avg_N: float
    avg_ratio: float
    loss: float
    active_margin_rate: float
    hit_at_k: int
    recall_at_k: float
    precision_at_k: float
    map_at_k: float
    num_annotation_edges: int
    n_queries: int
    n_samples: int
    query_stats: list[dict[str, Any]]
    edge_stats: list[dict[str, Any]]


def saliency_details_from_outputs(
    model,
    outputs,
    annot_pairs: list[Tensor] | Tensor,
    *,
    alpha: float,
    eps: float,
    floor_eps: float = 0.0,
    floor_logit_eps: float | None = None,
    top_k: int = 10,
    source_chunk_size: int = 32,
) -> SaliencyDetailDiagnostics:
    """Compute strict-causal saliency diagnostics without changing training loss.

    This helper is intentionally separate from src/train/loss.py's saliency loss.
    It uses the same contribution-row definition, but only records diagnostic
    values for annotation edges that already satisfy source < query.
    """
    attn_last = outputs.attentions[-1]
    if attn_last is None:
        raise RuntimeError("outputs.attentions[-1] is None; use eager attention")

    last_hidden_in = outputs.hidden_states[-2]
    B, T, _ = last_hidden_in.shape
    device = last_hidden_in.device

    flat = flatten_annot_pairs(annot_pairs, device=device)
    if flat.numel() == 0:
        return SaliencyDetailDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, 0, 0, [], [])

    batch_ids = flat[:, 0]
    src_all = flat[:, 1]
    qry_all = flat[:, 2]
    keep = (
        (batch_ids >= 0) & (batch_ids < B) &
        (src_all >= 0) & (src_all < T) &
        (qry_all >= 0) & (qry_all < T) &
        (src_all < qry_all)
    )
    batch_ids = batch_ids[keep]
    src_all = src_all[keep]
    qry_all = qry_all[keep]
    if batch_ids.numel() == 0:
        return SaliencyDetailDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, 0, 0, [], [])

    keys = batch_ids * T + qry_all
    unique_keys, inv = torch.unique(keys, return_inverse=True)
    row_batch = unique_keys // T
    row_qry = unique_keys % T

    with torch.no_grad():
        C_rows = build_contribution_rows(
            model,
            last_hidden_in.detach(),
            attn_last.detach(),
            row_batch,
            row_qry,
            source_chunk_size=source_chunk_size,
        ).float()

        Q = C_rows.size(0)
        A_adj = torch.zeros(Q, T, device=device, dtype=C_rows.dtype)
        A_adj.index_put_((inv, src_all), torch.ones_like(src_all, dtype=C_rows.dtype), accumulate=True)
        annot_mask = (A_adj > 0).to(C_rows.dtype)

        src_idx = torch.arange(T, device=device).unsqueeze(0)
        qry_col = row_qry.unsqueeze(1)
        causal_mask = ((src_idx < qry_col)).to(C_rows.dtype)
        annot_mask = annot_mask * causal_mask
        non_annot_mask = causal_mask * (1.0 - annot_mask)

        den_C = annot_mask.sum(dim=-1).clamp_min(eps)
        den_N = non_annot_mask.sum(dim=-1).clamp_min(eps)
        C_bar = (annot_mask * C_rows).sum(dim=-1) / den_C
        N_bar = (non_annot_mask * C_rows).sum(dim=-1) / den_N
        ratio = C_bar / (N_bar + eps)

        eps_f = float(eps)
        floor_eps_f = float(floor_eps)
        scores = torch.log(C_rows.float() + eps_f)
        logits = scores / alpha
        if floor_logit_eps is not None:
            floor_logit = logits.new_tensor(float(floor_logit_eps))
        else:
            floor_logit = torch.log(logits.new_tensor(floor_eps_f + eps_f)) / alpha
        neg_inf = torch.finfo(logits.dtype).min
        neg_logits = torch.maximum(logits, floor_logit).masked_fill(non_annot_mask == 0, neg_inf)
        neg_logsumexp = torch.logsumexp(neg_logits, dim=-1)
        pos_log_denom = torch.logaddexp(neg_logsumexp.unsqueeze(-1), logits)
        pos_nll = (pos_log_denom - logits) * annot_mask
        query_loss = pos_nll.sum(dim=-1) / den_C
        loss = float(query_loss.mean().detach().cpu())

        active_margin_rate = 0.0

        query_stats: list[dict[str, Any]] = []
        edge_stats: list[dict[str, Any]] = []
        hit_at_k = 0
        recall_values: list[float] = []
        precision_values: list[float] = []
        ap_values: list[float] = []
        rank_lookup_by_row: list[dict[int, int]] = []

        for qrow in range(Q):
            q = int(row_qry[qrow].detach().cpu())
            annot_sources = torch.nonzero(annot_mask[qrow] > 0, as_tuple=False).flatten()
            if annot_sources.numel() == 0:
                rank_lookup_by_row.append({})
                continue

            causal_sources = torch.arange(q, device=device, dtype=torch.long)
            if causal_sources.numel() == 0:
                rank_lookup_by_row.append({})
                continue
            causal_order_local = torch.argsort(C_rows[qrow, :q], descending=True)
            causal_order = causal_sources[causal_order_local]
            rank_lookup = {int(src.detach().cpu()): rank for rank, src in enumerate(causal_order, start=1)}
            rank_lookup_by_row.append(rank_lookup)
            annot_ranks = torch.tensor(
                [rank_lookup[int(src.detach().cpu())] for src in annot_sources if int(src.detach().cpu()) in rank_lookup],
                device=device,
                dtype=torch.long,
            )
            if annot_ranks.numel() == 0:
                continue

            k_eff = min(max(1, int(top_k)), int(causal_order.numel()))
            top_sources = set(int(src.detach().cpu()) for src in causal_order[:k_eff])
            annot_set = set(int(src.detach().cpu()) for src in annot_sources)
            hit_count = len(top_sources & annot_set)
            hit_at_k += hit_count
            recall_at_k = hit_count / max(1, len(annot_set))
            precision_at_k = hit_count / max(1, k_eff)

            hits_so_far = 0
            precision_sum = 0.0
            for rank, src in enumerate(causal_order[:k_eff], start=1):
                if int(src.detach().cpu()) in annot_set:
                    hits_so_far += 1
                    precision_sum += hits_so_far / rank
            ap_at_k = precision_sum / max(1, len(annot_set))
            recall_values.append(float(recall_at_k))
            precision_values.append(float(precision_at_k))
            ap_values.append(float(ap_at_k))

            query_stats.append({
                "batch": int(row_batch[qrow].detach().cpu()),
                "query": q,
                "num_annotation_sources": int(annot_sources.numel()),
                "Cbar": float(C_bar[qrow].detach().cpu()),
                "Nbar": float(N_bar[qrow].detach().cpu()),
                "ratio": float(ratio[qrow].detach().cpu()),
                "loss": float(query_loss[qrow].detach().cpu()),
                "active_margin_rate": 0.0,
                "top_k": int(top_k),
                "hit_at_k_count": int(hit_count),
                "recall_at_k": float(recall_at_k),
                "precision_at_k": float(precision_at_k),
                "ap_at_k": float(ap_at_k),
                "mean_annotation_rank": float(annot_ranks.float().mean().detach().cpu()),
                "min_annotation_rank": int(annot_ranks.min().detach().cpu()),
                "max_annotation_rank": int(annot_ranks.max().detach().cpu()),
            })

        for eidx in range(src_all.numel()):
            qrow = int(inv[eidx].detach().cpu())
            src = int(src_all[eidx].detach().cpu())
            q = int(row_qry[qrow].detach().cpu())
            c_val = float(C_rows[qrow, src].detach().cpu())
            l_val = float(logits[qrow, src].detach().cpu())
            edge_stats.append({
                "batch": int(row_batch[qrow].detach().cpu()),
                "query": q,
                "source": src,
                "Cqs": c_val,
                "Cbar": float(C_bar[qrow].detach().cpu()),
                "Nbar": float(N_bar[qrow].detach().cpu()),
                "ratio": float(ratio[qrow].detach().cpu()),
                "r_qs": float(scores[qrow, src].detach().cpu()),
                "l_qs": l_val,
                "p_qs": float(torch.exp(logits[qrow, src] - pos_log_denom[qrow, src]).detach().cpu()),
                "loss_qs": float(pos_nll[qrow, src].detach().cpu()),
                "negative_floor_logit": float(floor_logit.detach().cpu()),
                "rank": int(rank_lookup_by_row[qrow].get(src, 0)) if qrow < len(rank_lookup_by_row) else 0,
                "active": False,
            })

        return SaliencyDetailDiagnostics(
            avg_C=float(C_bar.mean().detach().cpu()),
            avg_N=float(N_bar.mean().detach().cpu()),
            avg_ratio=float(ratio.mean().detach().cpu()),
            loss=loss,
            active_margin_rate=active_margin_rate,
            hit_at_k=hit_at_k,
            recall_at_k=float(sum(recall_values) / len(recall_values)) if recall_values else 0.0,
            precision_at_k=float(sum(precision_values) / len(precision_values)) if precision_values else 0.0,
            map_at_k=float(sum(ap_values) / len(ap_values)) if ap_values else 0.0,
            num_annotation_edges=int(src_all.numel()),
            n_queries=len(query_stats),
            n_samples=int(row_batch.unique().numel()),
            query_stats=query_stats,
            edge_stats=edge_stats,
        )
