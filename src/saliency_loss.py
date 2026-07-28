from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def _unwrap_to_decoder_stack(model):
    """
    Walk through HF / PEFT / DDP wrappers and return the module that owns `.layers` (ecoderLayer list).
    """
    m = model
    # DDP / DeepSpeed wrappers
    if hasattr(m, "module") and not hasattr(m, "layers"):
        m = m.module
    while True:
        if hasattr(m, "layers"):
            return m
        if hasattr(m, "model"):
            m = m.model
            continue
        if hasattr(m, "base_model"):
            m = m.base_model
            continue
        raise RuntimeError(f"Cannot locate decoder layer stack on {type(model).__name__}.")



def build_contribution_matrix(
    model,
    last_hidden_in: Tensor,   # [B, T, D]   input to the selected decoder layer
    attn_probs: Tensor,       # [B, H, T, T] Step 1
    *,
    layer_index: int = -1,
) -> Tensor:
    """
        Strict Kobayashi/ALTI contribution c_{i,j} = ||T_i(x_j)||_2 for the
        decoder layer ``layer_index`` (default -1 = last layer).
    """
    decoder = _unwrap_to_decoder_stack(model)
    layer = decoder.layers[layer_index] # selected decoder block
    self_attn = layer.self_attn

    B, T, D = last_hidden_in.shape
    H = attn_probs.size(1)
    device = last_hidden_in.device
    dtype = last_hidden_in.dtype

    head_dim = getattr(self_attn, "head_dim", None) or (
        self_attn.q_proj.weight.shape[0] // H
    ) 
    v_out = self_attn.v_proj.weight.shape[0]
    num_kv_heads = v_out // head_dim
    assert H % num_kv_heads == 0, f"H={H} not divisible by num_kv_heads={num_kv_heads}"
    n_rep = H // num_kv_heads

    gamma = layer.input_layernorm.weight.to(device).float()           # [D]
    gamma_x = last_hidden_in.float() * gamma                          # [B, T, D]
    v_w = self_attn.v_proj.weight.to(device).float()                  # [num_kv_heads*head_dim, D]
    v_b = self_attn.v_proj.bias
    v_proj = gamma_x @ v_w.t()                                        # [B, T, num_kv_heads*head_dim]
    if v_b is not None:
        v_proj = v_proj + v_b.to(device).float()
    v_states = v_proj.view(B, T, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    if n_rep > 1:
        v_states = (
            v_states.unsqueeze(2)
            .expand(B, num_kv_heads, n_rep, T, head_dim)
            .reshape(B, H, T, head_dim)
        )

    o_w = self_attn.o_proj.weight.to(device).float()    
    o_w_by_head = o_w.view(D, H, head_dim)
    transformed = torch.einsum("bhsd,ohd->bhso", v_states, o_w_by_head)

    #  Σ_h A^h_{i,j} · (W_O^h W_V^h x_j)
    attn_f = attn_probs.float()                                       # [B, H, T, T]
    T_ij = torch.einsum("bhij,bhjo->bijo", attn_f, transformed)       # [B, T, T, D]

    # ignore the diagonals (j == i branch)
    diag_idx = torch.arange(T, device=device)
    T_ij[:, diag_idx, diag_idx, :] = (
        T_ij[:, diag_idx, diag_idx, :] + last_hidden_in.float()[:, diag_idx, :]
    )

    # divide by σ_i = RMS(x_i)  (query-side) 
    eps_rms = getattr(layer.input_layernorm, "variance_epsilon", 1e-6)
    sigma_i = last_hidden_in.float().pow(2).mean(dim=-1).add(eps_rms).sqrt()  # [B, T]
    T_ij = T_ij / sigma_i.unsqueeze(-1).unsqueeze(-1).clamp_min(1e-12)

    # Take L2 norm for easier optimization!
    c = T_ij.norm(dim=-1, p=2)                                        # [B, T, T]
    return c.to(dtype)


@dataclass
class SaliencyDiagnostics:
    loss: Tensor
    avg_C: float
    avg_N: float
    avg_ratio: float
    n_queries: int
    n_samples: int
    floor_eps: float = 0.0
    batch_floor_eps: float = 0.0
    floor_eps_mode: str = "fixed"
    floor_eps_step: int = 0
    floor_eps_warmup_steps: int = 0
    floor_logit_eps: float = 0.0
    floor_eps_kind: str = "saliency"
    loss_type: str = "softmax_margin"


def canonical_saliency_loss_type(loss_type: str | None) -> str:
    """Canonical public saliency-loss names.

    `softmax_margin` is the former `infonce_floor` implementation: log(C+eps)/tau
    with a negative floor. Keep old aliases readable so previous commands still run.
    """
    value = (loss_type or "softmax_margin").strip().lower().replace("-", "_")
    if value in {"", "default", "softmax_margin", "softmax_margin_loss", "infonce", "nll", "floor", "infonce_floor"}:
        return "softmax_margin"
    if value in {"softmax", "softmax_loss"}:
        return "softmax"
    if value in {"margin_bce", "bce_margin", "decoupled_bce", "per_edge_bce"}:
        return "margin_bce"
    if value in {"ranknet", "pairwise", "pairwise_ranknet", "rank_net"}:
        return "ranknet"
    if value in {"contrastive", "triplet", "pair_hinge", "pairwise_hinge"}:
        return "contrastive"
    raise ValueError(f"Unsupported saliency loss_type={loss_type!r}; expected softmax_margin, softmax, margin_bce, ranknet, or contrastive.")


def saliency_loss_display_name(loss_type: str | None) -> str:
    kind = canonical_saliency_loss_type(loss_type)
    if kind == "softmax":
        return "softmax loss"
    if kind == "margin_bce":
        return "decoupled margin-BCE loss"
    if kind == "ranknet":
        return "pairwise RankNet loss"
    if kind == "contrastive":
        return "pairwise contrastive (triplet hinge) loss"
    return "softmax-margin loss"


def flatten_annot_pairs(
    annot_pairs_batch: list[Tensor] | Tensor,
    device,
) -> Tensor:
    """
    Flatten annotation edges to [B, 3] of int64 columns
    (batch_idx, pos_a, pos_b). Accepts:

      • list[Tensor]   — per-sample [Number of pairs, 2] tensors (the dataset collator form)
      • Tensor [B, 3]  — already flat (batch_idx, pos_a, pos_b)
      • Tensor [B, M, 2] — padded with -1 for empty slots
    """
    if isinstance(annot_pairs_batch, torch.Tensor):
        t = annot_pairs_batch.to(device=device, dtype=torch.long)
        if t.dim() == 2 and t.size(-1) == 3:
            return t
        if t.dim() == 3 and t.size(-1) == 2:
            B_, M, _ = t.shape
            batch_idx = torch.arange(B_, device=device).unsqueeze(1).expand(B_, M)
            flat = torch.stack([batch_idx, t[..., 0], t[..., 1]], dim=-1).reshape(-1, 3)
            keep = (flat[:, 1] >= 0) & (flat[:, 2] >= 0)
            return flat[keep]
        raise ValueError(f"Unsupported tensor shape {tuple(t.shape)} for annot_pairs.")

    # list[Tensor]
    chunks = []
    for b, pairs in enumerate(annot_pairs_batch):
        if pairs is None or pairs.numel() == 0:
            continue
        p = pairs.to(device=device, dtype=torch.long)
        bid = torch.full((p.size(0),), b, device=device, dtype=torch.long)
        chunks.append(torch.stack([bid, p[:, 0], p[:, 1]], dim=-1))
    if not chunks:
        return torch.empty((0, 3), device=device, dtype=torch.long)
    return torch.cat(chunks, dim=0)


def compute_saliency_loss(
        C: Tensor,  # [B, T, T]
        annot_pairs: list[Tensor] | Tensor,  # flat [B, 3] or list of [Number of edges, 2]
        *,
        alpha: float,
        eps: float,
        floor_eps: float = 0.0,
        floor_logit_eps: float | None = None,
        loss_type: str = "softmax_margin",
) -> SaliencyDiagnostics:
    """
    Multi-positive NLL over causal sources, fully batched across samples.

    `annot_pairs` may be either
      • a flat tensor [B, 3] with columns (batch_idx, pos_a, pos_b), or
      • a list of [Number of edges, 2] tensors (for batch size = 1, one sample only).

    Cbar_i and Nbar_i are retained as diagnostics. loss_type selects either
    the softmax-margin log-saliency/floor objective or raw saliency softmax loss.

    Returns SaliencyDiagnostics.
    If no valid query rows across the batch, loss=0.
    """
    B, T, _ = C.shape
    device = C.device
    dtype = C.dtype

    # Normalize input to a flat [B, 3] tensor
    flat = flatten_annot_pairs(annot_pairs, device=device)
    if flat.numel() == 0:
        return SaliencyDiagnostics(
            loss=torch.tensor(0.0, device=device, dtype=dtype),
            avg_C=0.0, avg_N=0.0, avg_ratio=0.0,
            n_queries=0, n_samples=0,
        )

    batch_ids = flat[:, 0]
    p0 = flat[:, 1]
    p1 = flat[:, 2]

    # Swap so source ≤ query (respect causal direction).
    src_all = torch.minimum(p0, p1)
    qry_all = torch.maximum(p0, p1)

    # Filter out-of-range indices
    keep = (
            (src_all < T) &
            (qry_all < T) & (qry_all > src_all) &
            (batch_ids >= 0) & (batch_ids < B)
    )
    batch_ids = batch_ids[keep]
    src_all = src_all[keep]
    qry_all = qry_all[keep]

    if batch_ids.numel() == 0:
        return SaliencyDiagnostics(
            loss=torch.tensor(0.0, device=device, dtype=dtype),
            avg_C=0.0, avg_N=0.0, avg_ratio=0.0,
            n_queries=0, n_samples=0,
        )

    # Group edges by (batch, query) → unique rows Q
    keys = batch_ids * T + qry_all  # [B]
    unique_keys, inv = torch.unique(keys, return_inverse=True)  # inv: [B] → [0..Q)
    Q = unique_keys.numel()
    row_batch = unique_keys // T  # [Q]
    row_qry = unique_keys % T  # [Q]

    # A_adj[q, s] counts how many annotated edges of row q point to source s.
    A_adj = torch.zeros(Q, T, device=device, dtype=dtype)
    A_adj.index_put_(
        (inv, src_all),
        torch.ones_like(src_all, dtype=dtype),
        accumulate=True,
    )
    A_adj_bin = (A_adj > 0).to(dtype)  # binary mask, [Q, T]

    # Causal Mask
    src_idx = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
    qry_col = row_qry.unsqueeze(1)  # [Q, 1]
    M_causal = (
            (src_idx <= qry_col) &
            (src_idx != qry_col)
    ).to(dtype)  # [Q, T]

    # Gather c_{i,j} rows for each unique (b, q)
    C_rows = C[row_batch, row_qry, :]  # [Q, T]

    # C̄_i and N̄_i with the standard split
    annot_mask = M_causal * A_adj_bin  # [Q, T]
    non_annot_mask = M_causal * (1.0 - A_adj_bin)  # [Q, T]

    den_C = annot_mask.sum(dim=-1).clamp_min(eps)
    den_N = non_annot_mask.sum(dim=-1).clamp_min(eps)
    C_bar = (annot_mask * C_rows).sum(dim=-1) / den_C            # [Q] 仅诊断用
    N_bar = (non_annot_mask * C_rows).sum(dim=-1) / den_N        # [Q] 干扰基线

    # Only train queries that have at least one positive AND at least one negative.
    I_mask = (annot_mask.sum(dim=-1) > 0) & (non_annot_mask.sum(dim=-1) > 0)
    if not bool(I_mask.any()):
        return SaliencyDiagnostics(
            loss=torch.tensor(0.0, device=device, dtype=dtype),
            avg_C=0.0, avg_N=0.0, avg_ratio=0.0,
            n_queries=0, n_samples=int(batch_ids.unique().numel()),
        )
        
    # V2 loss: Hinge loss
    # C_rows_I = C_rows[I_mask]                                    # [Qi, T]
    # annot_I = annot_mask[I_mask]                                 # [Qi, T]
    # N_bar_I = N_bar[I_mask]                                      # [Qi]

    # # 逐标注边 hinge:每条标注边都要高过 alpha * N̄ 的相对 margin
    # target = alpha * N_bar_I.unsqueeze(-1)                       # [Qi, 1]
    # margin = F.relu(target - C_rows_I) * annot_I                 # [Qi, T] 只在标注边上罚
    # loss = margin.sum(dim=-1) / annot_I.sum(dim=-1).clamp_min(eps)
    # loss = loss.mean()

    # C_bar_I = C_bar[I_mask]

    # with torch.no_grad():
    #     ratio = C_bar_I / (N_bar_I + eps)
    #     diag = SaliencyDiagnostics(
    #         loss=loss,
    #         avg_C=float(C_bar_I.mean()),
    #         avg_N=float(N_bar_I.mean()),
    #         avg_ratio=float(ratio.mean()),
    #         n_queries=int(C_bar_I.numel()),
    #         n_samples=int(row_batch[I_mask].unique().numel()),
    #     )

    C_rows_I = C_rows[I_mask]                                    # [Qi, T]
    annot_I = annot_mask[I_mask]                                  # [Qi, T]
    nonannot_I = non_annot_mask[I_mask]                           # [Qi, T]
    causal_I = (annot_I + nonannot_I).clamp_max(1.0)               # [Qi, T]
    loss_kind = canonical_saliency_loss_type(loss_type)

    effective_floor_eps = max(float(floor_eps), 0.0)
    batch_floor_eps = None
    effective_floor_logit = 0.0
    floor_eps_kind = "saliency"
    eps_f = float(eps)

    if loss_kind == "softmax":
        # Softmax loss from the experiment note:
        # l_qs = C_qs / tau, p_qs = softmax over all causal sources M_q,
        # L_q = - mean_{s in A_q} log p_qs. Positives and negatives compete
        # in the same denominator, so no margin or negative floor is used.
        tau = max(float(alpha), eps_f)
        logits_I = C_rows_I.float() / tau
        neg_inf = torch.finfo(logits_I.dtype).min
        logits_I = logits_I.masked_fill(causal_I == 0, neg_inf)
        log_probs = F.log_softmax(logits_I, dim=-1)
        n_pos = annot_I.sum(dim=-1).clamp_min(eps)
        pos_nll = (-log_probs) * annot_I
        loss = (pos_nll.sum(dim=-1) / n_pos).mean().to(dtype)
        effective_floor_eps = 0.0
        floor_eps_kind = "disabled"
    elif loss_kind == "softmax_margin":
        # Legacy V4 loss: multi-positive NLL with a negative-logit floor.
        # For every positive source s, its denominator is exp(l_qs) plus
        # floored negative causal sources; positives do not compete with each other.
        floor_eps_f = float(effective_floor_eps)
        tau = max(float(alpha), eps_f)
        scores_I = torch.log(C_rows_I.float() + eps_f)             # [Qi, T]
        logits_I = scores_I / tau                                  # [Qi, T]

        neg_inf = torch.finfo(logits_I.dtype).min
        if floor_logit_eps is not None:
            floor = logits_I.new_tensor(float(floor_logit_eps))
            effective_floor_logit = float(floor_logit_eps)
            floor_eps_kind = "logit"
        else:
            floor = torch.log(logits_I.new_tensor(floor_eps_f + eps_f)) / tau
            effective_floor_logit = float(floor.detach().cpu())
            floor_eps_kind = "saliency"
        neg_logits = torch.maximum(logits_I, floor).masked_fill(nonannot_I == 0, neg_inf)
        neg_logsumexp = torch.logsumexp(neg_logits, dim=-1)        # [Qi]

        n_pos = annot_I.sum(dim=-1).clamp_min(eps)                 # [Qi]
        pos_log_denom = torch.logaddexp(neg_logsumexp.unsqueeze(-1), logits_I)
        pos_nll = (pos_log_denom - logits_I) * annot_I             # [Qi, T]
        loss = (pos_nll.sum(dim=-1) / n_pos).mean().to(dtype)
    else:
        raise ValueError(f"Unsupported saliency loss_type={loss_type!r}; expected softmax_margin or softmax.")

    C_bar_I = C_bar[I_mask]
    N_bar_I = N_bar[I_mask]

    with torch.no_grad():
        ratio = C_bar_I / (N_bar_I + eps)
        diag = SaliencyDiagnostics(
            loss=loss,
            avg_C=float(C_bar_I.mean()),
            avg_N=float(N_bar_I.mean()),
            avg_ratio=float(ratio.mean()),
            n_queries=int(C_bar_I.numel()),
            n_samples=int(row_batch[I_mask].unique().numel()),
            floor_eps=float(effective_floor_eps),
            batch_floor_eps=float(batch_floor_eps or 0.0),
            floor_eps_mode="disabled" if loss_kind == "softmax" else "fixed_logit" if floor_eps_kind == "logit" else "fixed",
            floor_logit_eps=float(effective_floor_logit),
            floor_eps_kind=floor_eps_kind,
            loss_type=loss_kind,
        )
    return diag


def build_contribution_rows(
    model,
    last_hidden_in: Tensor,   # [B, T, D]
    attn_probs: Tensor,       # [B, H, T, T]
    row_batch: Tensor,        # [Q]
    row_qry: Tensor,          # [Q]
    *,
    layer_index: int = -1,
    source_chunk_size: int = 16,
    query_chunk_size: int = 64,
) -> Tensor:
    """Compute only selected query rows C[b, q, :] of the contribution matrix.

    This is mathematically equivalent to gathering rows from
    build_contribution_matrix(...), but avoids materializing the full
    [B, T, T, D] contribution tensor. It is the training path for the
    InfoNCE saliency loss.

    Memory optimizations vs. the original implementation:
      * **B==1 fast path**: when there is a single example in the batch,
        ``transformed`` is sliced as ``[H, S, D]`` instead of ``[Q, H, S, D]``,
        avoiding the Q-fold duplication that previously dominated activation
        memory (Q can be 200+ on long sequences with dense annotation).
      * **query chunking**: outer loop over query batches of size
        ``query_chunk_size`` so the peak ``[Q, H, S, D]`` chunk for B>1
        becomes ``[Qc, H, S, D]``. Mathematically identical to the original
        single-pass implementation (each query row is independent in the
        einsum).

    ``layer_index`` selects which decoder block's attention/value path is used
    (default -1 = last layer). The caller must pass the matching
    ``attn_probs`` (= outputs.attentions[layer_index]) and ``last_hidden_in``
    (= input to that layer = outputs.hidden_states[layer_index]).
    """
    decoder = _unwrap_to_decoder_stack(model)
    layer = decoder.layers[layer_index]
    self_attn = layer.self_attn

    B, T, D = last_hidden_in.shape
    H = attn_probs.size(1)
    device = last_hidden_in.device
    dtype = last_hidden_in.dtype

    head_dim = getattr(self_attn, "head_dim", None) or (
        self_attn.q_proj.weight.shape[0] // H
    )
    v_out = self_attn.v_proj.weight.shape[0]
    num_kv_heads = v_out // head_dim
    assert H % num_kv_heads == 0, f"H={H} not divisible by num_kv_heads={num_kv_heads}"
    n_rep = H // num_kv_heads

    gamma = layer.input_layernorm.weight.to(device).float()
    gamma_x = last_hidden_in.float() * gamma
    v_w = self_attn.v_proj.weight.to(device).float()
    v_b = self_attn.v_proj.bias
    v_proj = gamma_x @ v_w.t()
    if v_b is not None:
        v_proj = v_proj + v_b.to(device).float()
    v_states = v_proj.view(B, T, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    if n_rep > 1:
        v_states = (
            v_states.unsqueeze(2)
            .expand(B, num_kv_heads, n_rep, T, head_dim)
            .reshape(B, H, T, head_dim)
        )

    o_w = self_attn.o_proj.weight.to(device).float()
    o_w_by_head = o_w.view(D, H, head_dim)
    transformed = torch.einsum("bhsd,ohd->bhso", v_states, o_w_by_head)

    row_batch = row_batch.to(device=device, dtype=torch.long)
    row_qry = row_qry.to(device=device, dtype=torch.long)
    Q = row_qry.numel()
    if Q == 0:
        return torch.empty((0, T), device=device, dtype=dtype)

    eps_rms = getattr(layer.input_layernorm, "variance_epsilon", 1e-6)
    query_hidden_all = last_hidden_in.float()[row_batch, row_qry, :]   # [Q, D]
    sigma_q_all = (
        query_hidden_all.pow(2).mean(dim=-1).add(eps_rms).sqrt().clamp_min(1e-12)
    )

    attn_f = attn_probs.float()

    # B==1 fast path: drop the Q-fold gather along the batch axis.
    single_example = (B == 1)
    if single_example:
        transformed_b0 = transformed[0]   # [H, T, D]
        attn_b0 = attn_f[0]               # [H, T, T]

    out = torch.empty((Q, T), device=device, dtype=dtype)
    qchunk = max(1, int(query_chunk_size))
    for q_start in range(0, Q, qchunk):
        q_end = min(q_start + qchunk, Q)
        rb = row_batch[q_start:q_end]
        rq = row_qry[q_start:q_end]
        Qc = rq.numel()
        query_hidden = query_hidden_all[q_start:q_end]
        sigma_q = sigma_q_all[q_start:q_end]

        chunks: list[Tensor] = []
        for start in range(0, T, source_chunk_size):
            stop = min(start + source_chunk_size, T)
            if single_example:
                # [Qc, H, S]
                attn_chunk = attn_b0[:, rq, start:stop].permute(1, 0, 2).contiguous()
                # [H, S, D] -> einsum without Q-fold duplication
                transformed_chunk = transformed_b0[:, start:stop, :]
                contrib = torch.einsum("qhs,hso->qso", attn_chunk, transformed_chunk)
            else:
                attn_chunk = attn_f[rb, :, rq, start:stop]                # [Qc, H, S]
                transformed_chunk = transformed[rb, :, start:stop, :]     # [Qc, H, S, D]
                contrib = torch.einsum("qhs,qhso->qso", attn_chunk, transformed_chunk)

            diag_mask = (rq >= start) & (rq < stop)
            if bool(diag_mask.any()):
                local = rq[diag_mask] - start
                contrib[diag_mask, local, :] = contrib[diag_mask, local, :] + query_hidden[diag_mask]

            contrib = contrib / sigma_q.view(Qc, 1, 1)
            chunks.append(contrib.norm(dim=-1, p=2).to(dtype))

        out[q_start:q_end] = torch.cat(chunks, dim=1)

    return out


def _annotation_rows_from_pairs(
    annot_pairs: list[Tensor] | Tensor,
    *,
    B: int,
    T: int,
    device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    flat = flatten_annot_pairs(annot_pairs, device=device)
    if flat.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return empty, empty, empty, empty

    batch_ids = flat[:, 0]
    p0 = flat[:, 1]
    p1 = flat[:, 2]
    src_all = torch.minimum(p0, p1)
    qry_all = torch.maximum(p0, p1)
    keep = (
        (src_all < T) &
        (qry_all < T) & (qry_all > src_all) &
        (batch_ids >= 0) & (batch_ids < B)
    )
    batch_ids = batch_ids[keep]
    src_all = src_all[keep]
    qry_all = qry_all[keep]
    if batch_ids.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return empty, empty, empty, empty

    keys = batch_ids * T + qry_all
    unique_keys, inv = torch.unique(keys, return_inverse=True)
    row_batch = unique_keys // T
    row_qry = unique_keys % T
    return row_batch, row_qry, src_all, inv




def estimate_causal_saliency_quantile_from_rows(
    C_rows: Tensor,
    row_qry: Tensor,
    *,
    quantile: float = 0.75,
    min_eps: float = 1e-8,
) -> float | None:
    """Estimate Q_quantile over causal source saliency for selected query rows.

    The returned value is detached and intended as a training hyperparameter
    estimate, not as a differentiable part of the objective.
    """
    if C_rows.numel() == 0 or row_qry.numel() == 0:
        return None
    with torch.no_grad():
        q = float(max(0.0, min(1.0, quantile)))
        values = []
        row_qry_list = row_qry.detach().to(device="cpu", dtype=torch.long).tolist()
        for row_idx, qry in enumerate(row_qry_list):
            if qry <= 0:
                continue
            values.append(C_rows[row_idx, :qry].detach().float().reshape(-1))
        if not values:
            return None
        flat = torch.cat(values)
        if flat.numel() == 0:
            return None
        est = torch.quantile(flat, q)
        return max(float(est.item()), float(min_eps))


def resolve_saliency_floor_eps(
    *,
    mode: str,
    fixed_floor_eps: float,
    batch_floor_eps: float | None,
    prev_floor_eps: float | None,
    ema_beta: float,
    min_eps: float,
    step: int = 0,
    warmup_steps: int = 0,
) -> float:
    """Resolve the saliency-space negative floor used by the current step.

    For ema_quantile, the current batch first produces hat_eps_t, then the
    returned eps_t is used in this same step's saliency loss:

        eps_t = 0,                                  t < T_warmup
        eps_t = hat_eps_t,                        t = T_warmup
        eps_t = beta * eps_{t-1} + (1-beta)*hat_eps_t, t > T_warmup

    ``batch_floor_eps`` is detached before this helper is called, so eps_t is a
    dynamic hyperparameter rather than a differentiable target.
    """
    mode = (mode or "fixed").strip().lower()
    fixed = max(float(fixed_floor_eps), 0.0)
    floor_min = float(min_eps)
    step_i = int(step or 0)
    warmup_i = max(int(warmup_steps or 0), 0)

    if mode == "fixed":
        return fixed

    if mode in {"batch_quantile", "ema_quantile"} and warmup_i > 0 and 0 < step_i < warmup_i:
        return 0.0

    if batch_floor_eps is None:
        if prev_floor_eps is not None:
            return max(float(prev_floor_eps), floor_min)
        return max(fixed, floor_min) if fixed > 0 else 0.0

    batch = max(float(batch_floor_eps), floor_min)
    if mode == "batch_quantile":
        return batch
    if mode == "ema_quantile":
        beta = min(max(float(ema_beta), 0.0), 0.9999)
        # At the first active floor step, initialize eps_t directly from the
        # current batch quantile. This matches eps_Twarmup = \hat{eps}_Twarmup.
        if warmup_i > 0 and step_i == warmup_i:
            return batch
        if prev_floor_eps is not None and float(prev_floor_eps) > 0:
            base = float(prev_floor_eps)
            return max(beta * base + (1.0 - beta) * batch, floor_min)
        if fixed > 0:
            return max(beta * fixed + (1.0 - beta) * batch, floor_min)
        return batch
    raise ValueError(f"Unsupported floor_eps_mode={mode!r}")

def compute_saliency_loss_from_rows(
    C_rows: Tensor,          # [Q, T]
    row_batch: Tensor,       # [Q]
    row_qry: Tensor,         # [Q]
    src_all: Tensor,         # [E]
    inv: Tensor,             # [E], edge -> row index
    *,
    alpha: float,
    eps: float,
    floor_eps: float = 0.0,
    floor_eps_mode: str = "fixed",
    floor_eps_quantile: float = 0.75,
    floor_eps_ema_beta: float = 0.95,
    prev_floor_eps: float | None = None,
    floor_eps_min: float = 1e-8,
    floor_eps_step: int = 0,
    floor_eps_warmup_steps: int = 0,
    floor_logit_eps: float | None = None,
    loss_type: str = "softmax_margin",
    margin_plus: float = 2.08,    # log(8) — desired log-saliency lower bound for annotated edges
    margin_minus: float = 0.41,   # log(1.5) — log-saliency upper bound for non-annot edges (no penalty below)
    margin_gamma: float = 2.0,    # softplus sharpness
    neg_weight: float = 0.5,      # multiplier on the negative-side mean penalty
    neg_hard_only: bool = False,  # if True, average only over negatives with r > margin_minus
    neg_sample_k: int = 0,        # if >0, randomly subsample this many negatives per query for softmax/softmax_margin (MoCo-style)
    exclude_source_rows: Tensor | None = None,  # [Q, T] bool/float, 1 = drop this source from the NEGATIVE set (sink / special tokens)
) -> SaliencyDiagnostics:
    """Saliency objective with C already row-gathered."""
    Q, T = C_rows.shape
    device = C_rows.device
    dtype = C_rows.dtype
    if Q == 0:
        return SaliencyDiagnostics(
            loss=torch.tensor(0.0, device=device, dtype=dtype),
            avg_C=0.0,
            avg_N=0.0,
            avg_ratio=0.0,
            n_queries=0,
            n_samples=0,
        )

    A_adj = torch.zeros(Q, T, device=device, dtype=dtype)
    A_adj.index_put_(
        (inv.to(device=device, dtype=torch.long), src_all.to(device=device, dtype=torch.long)),
        torch.ones_like(src_all, device=device, dtype=dtype),
        accumulate=True,
    )
    A_adj_bin = (A_adj > 0).to(dtype)

    src_idx = torch.arange(T, device=device).unsqueeze(0)
    qry_col = row_qry.to(device=device, dtype=torch.long).unsqueeze(1)
    M_causal = ((src_idx <= qry_col) & (src_idx != qry_col)).to(dtype)

    annot_mask = M_causal * A_adj_bin
    non_annot_mask = M_causal * (1.0 - A_adj_bin)

    # Drop sink / special-token sources from the NEGATIVE set so the loss never
    # pushes down the attention sink (a load-bearing no-op valve, per Gu/Pang
    # 2025, Barbero 2025). Positives are untouched — annotation edges never
    # point at sink/special tokens.
    if exclude_source_rows is not None:
        keep_src = (1.0 - exclude_source_rows.to(device=device, dtype=dtype)).clamp_(0.0, 1.0)
        non_annot_mask = non_annot_mask * keep_src

    den_C = annot_mask.sum(dim=-1).clamp_min(eps)
    den_N = non_annot_mask.sum(dim=-1).clamp_min(eps)
    C_bar = (annot_mask * C_rows).sum(dim=-1) / den_C
    N_bar = (non_annot_mask * C_rows).sum(dim=-1) / den_N

    I_mask = (annot_mask.sum(dim=-1) > 0) & (non_annot_mask.sum(dim=-1) > 0)
    if not bool(I_mask.any()):
        return SaliencyDiagnostics(
            loss=torch.tensor(0.0, device=device, dtype=dtype),
            avg_C=0.0,
            avg_N=0.0,
            avg_ratio=0.0,
            n_queries=0,
            n_samples=int(row_batch.unique().numel()),
        )

    C_rows_I = C_rows[I_mask]
    annot_I = annot_mask[I_mask]
    nonannot_I = non_annot_mask[I_mask]

    loss_kind = canonical_saliency_loss_type(loss_type)

    # Optional MoCo-style negative subsampling for softmax / softmax_margin.
    # For each query, keep at most `neg_sample_k` randomly-chosen negatives
    # active inside the softmax denominator. Positives and the non-causal mask
    # are untouched. Sampling weights are uniform over current negatives so
    # every above-floor negative has an equal expected gradient share, breaking
    # the InfoNCE winner-takes-all bias. Across steps the iid resampling covers
    # the full N_q (MoCo / SimCLR style).
    if (
        int(neg_sample_k) > 0
        and loss_kind in ("softmax", "softmax_margin")
        and nonannot_I.size(0) > 0
    ):
        k = int(neg_sample_k)
        with torch.no_grad():
            neg_counts = nonannot_I.sum(dim=-1)
            # rows where we actually need to subsample (>k negatives)
            sample_rows = (neg_counts > k).nonzero(as_tuple=True)[0]
            if sample_rows.numel() > 0:
                # build per-row uniform weights on the negative positions
                weights = nonannot_I[sample_rows].to(torch.float32) + 1e-12  # [R, T]
                # zero-weight non-causal/positive positions stay near-zero; we want exactly 0
                weights = weights * nonannot_I[sample_rows].to(torch.float32)
                idx = torch.multinomial(weights, num_samples=k, replacement=False)  # [R, k]
                new_mask = torch.zeros_like(nonannot_I[sample_rows])
                new_mask.scatter_(1, idx, 1.0)
                nonannot_I = nonannot_I.clone()
                nonannot_I[sample_rows] = new_mask

    floor_mode = (floor_eps_mode or "fixed").strip().lower()
    batch_floor_eps = None
    effective_floor_eps = 0.0
    effective_floor_logit = 0.0
    floor_eps_kind = "saliency"
    eps_f = float(eps)

    if loss_kind == "softmax":
        causal_I = (annot_I + nonannot_I).clamp_max(1.0)
        tau = max(float(alpha), eps_f)
        logits_I = C_rows_I.float() / tau
        neg_inf = torch.finfo(logits_I.dtype).min
        logits_I = logits_I.masked_fill(causal_I == 0, neg_inf)
        log_probs = F.log_softmax(logits_I, dim=-1)
        n_pos = annot_I.sum(dim=-1).clamp_min(eps)
        pos_nll = (-log_probs) * annot_I
        loss = (pos_nll.sum(dim=-1) / n_pos).mean().to(dtype)
        diag_floor_mode = "disabled"
        floor_eps_kind = "disabled"
    elif loss_kind == "softmax_margin":
        if floor_logit_eps is None:
            batch_floor_eps = estimate_causal_saliency_quantile_from_rows(
                C_rows_I,
                row_qry[I_mask],
                quantile=floor_eps_quantile,
                min_eps=floor_eps_min,
            ) if floor_mode != "fixed" else None
            effective_floor_eps = resolve_saliency_floor_eps(
                mode=floor_mode,
                fixed_floor_eps=floor_eps,
                batch_floor_eps=batch_floor_eps,
                prev_floor_eps=prev_floor_eps,
                ema_beta=floor_eps_ema_beta,
                min_eps=floor_eps_min,
                step=floor_eps_step,
                warmup_steps=floor_eps_warmup_steps,
            )
        else:
            batch_floor_eps = None
            effective_floor_eps = 0.0

        floor_eps_f = float(effective_floor_eps)
        tau = max(float(alpha), eps_f)
        scores_I = torch.log(C_rows_I.float() + eps_f)
        logits_I = scores_I / tau
        neg_inf = torch.finfo(logits_I.dtype).min
        if floor_logit_eps is not None:
            floor = logits_I.new_tensor(float(floor_logit_eps))
            effective_floor_logit = float(floor_logit_eps)
            floor_eps_kind = "logit"
            diag_floor_mode = "fixed_logit"
        else:
            floor = torch.log(logits_I.new_tensor(floor_eps_f + eps_f)) / tau
            effective_floor_logit = float(floor.detach().cpu())
            floor_eps_kind = "saliency"
        neg_logits = torch.maximum(logits_I, floor).masked_fill(nonannot_I == 0, neg_inf)
        neg_logsumexp = torch.logsumexp(neg_logits, dim=-1)

        n_pos = annot_I.sum(dim=-1).clamp_min(eps)
        pos_log_denom = torch.logaddexp(neg_logsumexp.unsqueeze(-1), logits_I)
        pos_nll = (pos_log_denom - logits_I) * annot_I
        loss = (pos_nll.sum(dim=-1) / n_pos).mean().to(dtype)
        if floor_logit_eps is None:
            diag_floor_mode = floor_mode
    elif loss_kind == "margin_bce":
        # Per-edge decoupled soft-margin BCE on log-saliency.
        # r_{q,s} = log(C+eps); positives pushed above margin_plus, negatives
        # pushed below margin_minus. Both penalties use softplus(gamma * delta)/gamma
        # so per-edge gradient is sigmoid-bounded in (0, 1) and saturates smoothly
        # once the edge is on the right side of its margin. No shared denominator,
        # so positives don't compete, and margins are absolute constants (set from
        # the base model once) so they cannot drift with training.
        r = torch.log(C_rows_I.float() + eps_f)              # [Qi, T]
        gamma = max(float(margin_gamma), eps_f)
        # Positive side:
        pos_arg = gamma * (float(margin_plus) - r)
        pos_term = F.softplus(pos_arg) / gamma               # [Qi, T]
        pos_term = pos_term * annot_I                        # mask to positives
        n_pos = annot_I.sum(dim=-1).clamp_min(eps)
        pos_loss_q = pos_term.sum(dim=-1) / n_pos            # [Qi]
        # Negative side:
        neg_arg = gamma * (r - float(margin_minus))
        neg_term = F.softplus(neg_arg) / gamma               # [Qi, T]
        if neg_hard_only:
            active = nonannot_I * (r > float(margin_minus)).to(nonannot_I.dtype)
            neg_term = neg_term * active
            n_neg = active.sum(dim=-1).clamp_min(eps)
        else:
            neg_term = neg_term * nonannot_I
            n_neg = nonannot_I.sum(dim=-1).clamp_min(eps)
        neg_loss_q = neg_term.sum(dim=-1) / n_neg            # [Qi]

        loss = (pos_loss_q + float(neg_weight) * neg_loss_q).mean().to(dtype)
        diag_floor_mode = "margin_bce"
        floor_eps_kind = "absolute_margin"
        # surface margins in the existing diag fields for log readability
        effective_floor_eps = float(math.exp(float(margin_minus)) - eps_f)
        effective_floor_logit = float(margin_minus)
    elif loss_kind == "ranknet":
        # Pairwise RankNet on log-saliency. For each query q and each
        # (positive s, negative s') pair:
        #   loss_{q,s,s'} = softplus( (log C_{q,s'} - log C_{q,s}) / tau )
        # Each pair contributes an independent sigmoid-bounded gradient — no
        # shared softmax denominator, so no winner-takes-all on either side.
        # Softplus saturates as the negative falls below the positive, which
        # acts as a natural "robust floor" without any explicit threshold.
        tau = max(float(alpha), eps_f)
        scores = torch.log(C_rows_I.float() + eps_f)         # [Qi, T]
        Qi = scores.shape[0]
        per_q_losses = []
        for q in range(Qi):
            p_idx = annot_I[q].bool().nonzero(as_tuple=True)[0]
            n_idx = nonannot_I[q].bool().nonzero(as_tuple=True)[0]
            if p_idx.numel() == 0 or n_idx.numel() == 0:
                continue
            pos = scores[q, p_idx]                            # [|A|]
            neg = scores[q, n_idx]                            # [|N|]
            diff = (neg.unsqueeze(0) - pos.unsqueeze(1)) / tau  # [|A|, |N|]
            per_q_losses.append(F.softplus(diff).mean())
        if per_q_losses:
            loss = torch.stack(per_q_losses).mean().to(dtype)
        else:
            loss = torch.zeros((), device=device, dtype=dtype)
        diag_floor_mode = "ranknet"
        floor_eps_kind = "pairwise"
        effective_floor_eps = 0.0
        effective_floor_logit = 0.0
    elif loss_kind == "contrastive":
        # Classic triplet-style pair hinge on log-saliency. For each query q and
        # each (positive s, negative s') pair:
        #   loss_{q,s,s'} = max(0, margin + (log C_{q,s'} - log C_{q,s}) / tau)
        # Margin is set in log-saliency units via `margin_plus`. Once a pair is
        # on the right side of the margin, its gradient is exactly zero — so
        # already-correct negatives stop being pushed down and already-strong
        # positives stop being pulled up. This gives a sparse, robust signal
        # and avoids both InfoNCE's winner-takes-all and RankNet's softplus
        # tail still leaking gradient to easy pairs.
        tau = max(float(alpha), eps_f)
        margin = float(margin_plus)
        scores = torch.log(C_rows_I.float() + eps_f)         # [Qi, T]
        Qi_, T_ = scores.shape
        k_neg = int(neg_sample_k) if neg_sample_k else 0

        annot_b = annot_I.bool()      # [Qi, T]
        nonannot_b = nonannot_I.bool()  # [Qi, T]

        # Optional uniform-random per-query subsampling of negatives. We
        # implement randperm-equivalence via topk of uniform noise restricted
        # to the negative mask; queries with |N_q| <= k_neg keep the full mask.
        if k_neg > 0:
            n_per_q = nonannot_b.sum(dim=-1)              # [Qi]
            need_sub = n_per_q > k_neg
            if bool(need_sub.any()):
                rand = torch.rand(nonannot_b.shape, device=scores.device, dtype=torch.float32)
                rand = rand.masked_fill(~nonannot_b, -1.0)
                # topk along dim=-1; for rows with <k_neg negatives we still
                # take k_neg indices (some may be -1.0 = non-neg), so we mask
                # those out by intersecting with original nonannot_b below.
                k_take = min(k_neg, T_)
                _, top_idx = rand.topk(k_take, dim=-1)    # [Qi, k_take]
                sub_mask = torch.zeros_like(nonannot_b)
                sub_mask.scatter_(1, top_idx, True)
                sub_mask &= nonannot_b                    # keep only real negs
                # for queries with n_per_q <= k_neg, keep the full set
                neg_mask = torch.where(need_sub.unsqueeze(-1), sub_mask, nonannot_b)
            else:
                neg_mask = nonannot_b
        else:
            neg_mask = nonannot_b

        # Vectorized pair hinge. pair[q,i,j] = relu(margin + (scores[q,j] -
        # scores[q,i]) / tau) for positive i, negative j. Memory ~ Qi*T*T*4
        # bytes; with typical Qi=30, T=120 this is ~1.7MB, so we use a single
        # guard to fall back to the loop if a future config blows past 256MB.
        bytes_needed = Qi_ * T_ * T_ * 4
        if bytes_needed > 256 * 1024 * 1024:  # 256MB safety
            per_q = []
            for q in range(Qi_):
                p_idx = annot_b[q].nonzero(as_tuple=True)[0]
                n_idx = neg_mask[q].nonzero(as_tuple=True)[0]
                if p_idx.numel() == 0 or n_idx.numel() == 0:
                    continue
                pos = scores[q, p_idx]
                neg = scores[q, n_idx]
                diff = (neg.unsqueeze(0) - pos.unsqueeze(1)) / tau
                per_q.append(F.relu(margin + diff).mean())
            if per_q:
                loss = torch.stack(per_q).mean().to(dtype)
            else:
                loss = torch.zeros((), device=device, dtype=dtype)
        else:
            # [Qi, T_pos=T, T_neg=T] pair difference
            diff = (scores.unsqueeze(1) - scores.unsqueeze(2)) / tau  # diff[q,i,j] = (s[q,j]-s[q,i])/tau
            pair_loss = F.relu(margin + diff)                          # [Qi, T, T]
            pair_mask = annot_b.unsqueeze(2) & neg_mask.unsqueeze(1)   # [Qi, T_pos, T_neg]
            pair_mask_f = pair_mask.to(pair_loss.dtype)
            n_pairs_per_q = pair_mask_f.sum(dim=(1, 2))                # [Qi]
            sum_loss_per_q = (pair_loss * pair_mask_f).sum(dim=(1, 2))  # [Qi]
            valid_q = n_pairs_per_q > 0
            if bool(valid_q.any()):
                per_q_mean = sum_loss_per_q[valid_q] / n_pairs_per_q[valid_q].clamp_min(1.0)
                loss = per_q_mean.mean().to(dtype)
            else:
                loss = torch.zeros((), device=device, dtype=dtype)
        diag_floor_mode = "contrastive"
        floor_eps_kind = "pair_hinge"
        effective_floor_eps = 0.0
        effective_floor_logit = float(margin)
    else:
        raise ValueError(f"Unsupported saliency loss_type={loss_type!r}; expected softmax_margin, softmax, margin_bce, ranknet, or contrastive.")

    C_bar_I = C_bar[I_mask]
    N_bar_I = N_bar[I_mask]
    with torch.no_grad():
        ratio = C_bar_I / (N_bar_I + eps)
        diag = SaliencyDiagnostics(
            loss=loss,
            avg_C=float(C_bar_I.mean()),
            avg_N=float(N_bar_I.mean()),
            avg_ratio=float(ratio.mean()),
            n_queries=int(C_bar_I.numel()),
            n_samples=int(row_batch[I_mask].unique().numel()),
            floor_eps=float(effective_floor_eps),
            batch_floor_eps=float(batch_floor_eps or 0.0),
            floor_eps_mode=diag_floor_mode,
            floor_eps_step=int(floor_eps_step or 0),
            floor_eps_warmup_steps=int(floor_eps_warmup_steps or 0),
            floor_logit_eps=float(effective_floor_logit),
            floor_eps_kind=floor_eps_kind,
            loss_type=loss_kind,
        )
    return diag


def saliency_loss_from_outputs(
        model,
        outputs,  # HF output with attentions + hidden_states
        annot_pairs: list[Tensor] | Tensor,  # flat [B, 3] or list of [Number of pairs, 2]
        *,
        saliency_layer: int = -1,
        exclude_source_mask: Tensor | None = None,
        alpha: float = 1.5,
        eps: float = 1e-8,
        floor_eps: float = 0.0,
        floor_eps_mode: str = "fixed",
        floor_eps_quantile: float = 0.75,
        floor_eps_ema_beta: float = 0.95,
        prev_floor_eps: float | None = None,
        floor_eps_min: float = 1e-8,
        floor_eps_step: int = 0,
        floor_eps_warmup_steps: int = 0,
        floor_logit_eps: float | None = None,
        loss_type: str = "softmax_margin",
        margin_plus: float = 2.08,
        margin_minus: float = 0.41,
        margin_gamma: float = 2.0,
        neg_weight: float = 0.5,
        neg_hard_only: bool = False,
        neg_sample_k: int = 0,
) -> SaliencyDiagnostics:
    """
    Takes the model's forward output and produces the saliency diagnostics in one call.

    `outputs` must have `.attentions` (non-None) and `.hidden_states`.
    `annot_pairs` may be either a flat [B, 3] tensor (batch_idx, pos_a, pos_b)
    or a list of [Number of pairs, 2] tensors.
    """
    n_layers = len(outputs.attentions)
    li = int(saliency_layer) if int(saliency_layer) >= 0 else n_layers + int(saliency_layer)
    li = max(0, min(li, n_layers - 1))
    attn_sel = outputs.attentions[li]
    assert attn_sel is not None, "outputs.attentions[...] is None — switch to eager attention."

    # hidden_states tuple: [embedding_out, layer_0_out, ..., layer_{L-1}_out];
    # the input to decoder layer ``li`` is hidden_states[li].
    last_hidden_in = outputs.hidden_states[li]  # [B, T, D]

    B, T, _ = last_hidden_in.shape
    row_batch, row_qry, src_all, inv = _annotation_rows_from_pairs(
        annot_pairs,
        B=B,
        T=T,
        device=last_hidden_in.device,
    )
    if row_qry.numel() == 0:
        return SaliencyDiagnostics(
            loss=torch.tensor(0.0, device=last_hidden_in.device, dtype=last_hidden_in.dtype),
            avg_C=0.0,
            avg_N=0.0,
            avg_ratio=0.0,
            n_queries=0,
            n_samples=0,
        )

    C_rows = build_contribution_rows(
        model,
        last_hidden_in,
        attn_sel,
        row_batch,
        row_qry,
        layer_index=li,
    )
    exclude_rows = None
    if exclude_source_mask is not None:
        em = exclude_source_mask.to(device=C_rows.device)
        exclude_rows = em[row_batch.to(C_rows.device)]
    return compute_saliency_loss_from_rows(
        C_rows,
        row_batch,
        row_qry,
        src_all,
        inv,
        exclude_source_rows=exclude_rows,
        alpha=alpha,
        eps=eps,
        floor_eps=floor_eps,
        floor_eps_mode=floor_eps_mode,
        floor_eps_quantile=floor_eps_quantile,
        floor_eps_ema_beta=floor_eps_ema_beta,
        prev_floor_eps=prev_floor_eps,
        floor_eps_min=floor_eps_min,
        floor_eps_step=floor_eps_step,
        floor_eps_warmup_steps=floor_eps_warmup_steps,
        floor_logit_eps=floor_logit_eps,
        loss_type=loss_type,
        margin_plus=margin_plus,
        margin_minus=margin_minus,
        margin_gamma=margin_gamma,
        neg_weight=neg_weight,
        neg_hard_only=neg_hard_only,
        neg_sample_k=neg_sample_k,
    )


# ── Counterfactual shortcut-masking augmentation ──────────────────────────────
# Alternative to the saliency loss. Instead of forcing an attribution metric
# (ALTI mAP) to match annotations — which we showed only reshapes last-layer
# attention weights without changing the prediction (Goodhart) — we corrupt the
# putative *shortcut* tokens (non-annotated prefix tokens) and keep training with
# CE on the real target. Because the loss is on behaviour (P(target | corrupted
# input)) rather than on a proxy, any reduction means the model genuinely routes
# information from the surviving annotated tokens to the output.
#
# Corruption = attention-masking (set selected key positions unattendable). This
# keeps every other token's RoPE/positions intact and stays in-distribution (no
# novel [MASK] token).

IGNORE_INDEX = -100


@dataclass
class ShortcutMaskStats:
    masked_attention_mask: Tensor  # [B, T] — 0 at padding + masked shortcut keys
    n_masked: float                # avg #positions masked per sample
    n_candidates: float            # avg #maskable (non-annotated) positions per sample
    frac_masked: float             # n_masked / n_candidates over the batch


def build_shortcut_mask(
    input_ids: Tensor,                       # [B, T]
    labels: Tensor,                          # [B, T]  (IGNORE_INDEX on context)
    attention_mask: Tensor,                  # [B, T]  (0 = padding)
    annot_pairs_batch: "list[Tensor]",       # per-sample [N, 2] (src, dst) positions
    *,
    rate: float = 0.3,
    max_k: int = 0,
    min_k: int = 0,
    recency_window: int = 8,
    protect_prefix: int = 0,
    special_ids: Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    generator: torch.Generator | None = None,
    node_weight_batch: Tensor | None = None,  # [B, T] weight-to-target in [0,1]
    weight_aware: bool = False,
    p_max: float = 0.9,
    p_gamma: float = 1.0,
) -> ShortcutMaskStats:
    """Build a [B, T] attention mask that additionally hides a random subset of
    *non-annotated* context tokens (putative shortcuts) as attention KEYS.

    A context position p is *maskable* iff:
      - it is real (attention_mask == 1) and part of the prompt (labels == ignore);
      - it does NOT appear in any annotation edge (neither src nor dst);
      - it is not a special/role token (when ``special_ids`` is given);
      - it is not inside the first ``protect_prefix`` positions (chat/system header
        + attention sink), nor inside the ``recency_window`` immediately before the
        first target token (local syntax is a genuine dependency, never a shortcut).

    Per sample we mask k = clip(round(rate * n_candidates), min_k, max_k) of the
    maskable positions, chosen uniformly at random. k=0 (or no target) leaves the
    sample's mask untouched → that sample trains as ordinary CE.

    **Weight-aware mode** (``weight_aware=True`` and ``node_weight_batch`` given):
    instead of masking a fixed fraction uniformly, each maskable position is masked
    independently with probability ``p_max * (1 - w)^p_gamma`` where ``w`` is the
    position's augmented-edge weight into any target token (``node_weight``). So a
    first-order source (``w=1``) is never masked, an unrelated token (``w=0``) is
    masked with probability ``p_max`` (the highest, but < 1), and transitively
    related tokens fall in between (lower weight → higher mask probability). In this
    mode annotated endpoints are NOT blanket-protected — relevance to the *target*
    (not membership in any edge) decides masking — and ``rate``/``min_k``/``max_k``
    are ignored. ``special_ids``, ``protect_prefix`` and ``recency_window`` still
    apply, and position 0 is always kept (sink / softmax safety).
    """
    device = input_ids.device
    B, T = input_ids.shape
    masked = attention_mask.clone()
    if special_ids is not None and special_ids.numel() > 0:
        special_ids = special_ids.to(device)
    use_wa = bool(weight_aware) and node_weight_batch is not None
    if use_wa:
        node_weight_batch = node_weight_batch.to(device)

    tot_masked = 0
    tot_cand = 0
    for b in range(B):
        valid = attention_mask[b].bool()
        is_ctx = labels[b].eq(ignore_index)
        cand = valid & is_ctx

        # Never mask position 0 (BOS / attention sink): if its key were hidden,
        # the query-0 row would be all -inf under the causal mask → softmax NaN.
        # Keeping it available also respects the load-bearing attention-sink role.
        cand[0] = False

        # Protect annotated positions (both endpoints of every edge). Skipped in
        # weight-aware mode, where node_weight (relevance to the target) — not
        # membership in any edge — governs masking, so transitively related
        # endpoints stay maskable with a low probability.
        if not use_wa:
            pairs = annot_pairs_batch[b] if b < len(annot_pairs_batch) else None
            if pairs is not None and pairs.numel() > 0:
                p = pairs.to(device).view(-1)
                p = p[(p >= 0) & (p < T)]
                if p.numel() > 0:
                    cand[p] = False

        # Protect special / role tokens.
        if special_ids is not None and special_ids.numel() > 0:
            cand &= ~torch.isin(input_ids[b], special_ids)

        # Protect the leading header / attention-sink prefix.
        if protect_prefix > 0:
            cand[: min(protect_prefix, T)] = False

        # Protect the recency window just before the first target token.
        tgt_pos = (~is_ctx & valid).nonzero(as_tuple=False).flatten()
        if tgt_pos.numel() > 0 and recency_window > 0:
            j0 = int(tgt_pos[0].item())
            lo = max(0, j0 - recency_window)
            cand[lo:j0] = False

        cand_idx = cand.nonzero(as_tuple=False).flatten()
        n_cand = int(cand_idx.numel())
        tot_cand += n_cand
        if n_cand == 0:
            continue

        if use_wa:
            # Per-token Bernoulli: p_mask = p_max * (1 - w)^gamma.
            w = node_weight_batch[b, cand_idx].clamp(0.0, 1.0)
            p_mask = p_max * (1.0 - w).pow(p_gamma)
            r = (torch.rand(n_cand, generator=generator, device=device)
                 if generator is not None else torch.rand(n_cand, device=device))
            chosen = cand_idx[r < p_mask]
            if chosen.numel() > 0:
                masked[b, chosen] = 0
            tot_masked += int(chosen.numel())
            continue

        k = int(round(rate * n_cand))
        if min_k > 0:
            k = max(k, min_k)
        if max_k > 0:
            k = min(k, max_k)
        k = max(0, min(k, n_cand))
        if k == 0:
            continue

        perm = torch.randperm(n_cand, generator=generator, device=device)[:k]
        chosen = cand_idx[perm]
        masked[b, chosen] = 0
        tot_masked += k

    return ShortcutMaskStats(
        masked_attention_mask=masked,
        n_masked=tot_masked / max(1, B),
        n_candidates=tot_cand / max(1, B),
        frac_masked=(tot_masked / tot_cand) if tot_cand > 0 else 0.0,
    )


@dataclass
class PerTargetMaskStats:
    allow: Tensor          # [B, 1, T, T] bool — True = key visible to that query
    n_masked: float
    n_candidates: float
    frac_masked: float


def build_shortcut_mask_per_target(
    input_ids: Tensor,                  # [B, T]
    labels: Tensor,                     # [B, T]
    attention_mask: Tensor,             # [B, T]
    annot_pairs_batch: "list[Tensor]",  # per-sample [N, 2] (src, dst)
    *,
    annot_weights_batch: "list[Tensor] | None" = None,  # per-sample [N] edge weights
    rate: float = 0.3,
    recency_window: int = 8,
    protect_prefix: int = 0,
    special_ids: Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    generator: torch.Generator | None = None,
    weight_aware: bool = False,
    p_max: float = 0.9,
    p_gamma: float = 1.0,
) -> PerTargetMaskStats:
    """Per-target shortcut mask: a 2-D [T_q, T_k] visibility map per sample.

    Unlike :func:`build_shortcut_mask` (one global key mask for all queries),
    this hides a *different* set of keys for each target query row. For each
    target position q (``labels[q] != ignore`` — an annotated dst), only that
    target's own non-annotation context keys are maskable; every other query row
    keeps the plain causal mask. Returns a boolean ``allow[B, 1, T, T]`` (True =
    visible) that the trainer converts into an additive attention bias.

    The source set of target q is ``S_q = {i : (i, q) is an (augmented) edge}``.
    A key k (context, ``k < q``, not special / prefix / recency, ``k != 0``) is
    maskable for row q:
      - uniform      : mask ``round(rate * |candidates|)`` of them at random.
      - weight_aware : mask each with prob ``p_max * (1 - w)^gamma`` where
                       ``w = weight(k -> q)`` is the *per-target* edge weight
                       (first-order ``w=1`` never masked, unconnected ``w=0``
                       masked at ``p_max``).
    Because the source/weight set is per-target, this responds to edge
    augmentation even in the uniform case (denser per-target sources), unlike the
    global mask whose protected node set is augmentation-invariant.

    Position 0 and q itself stay visible (causal-softmax / NaN safety).
    Convention: q is the target token's own position (dst), matching the saliency
    framework C[q=dst, s=src]; masking row q shapes that target token's
    representation.
    """
    device = input_ids.device
    B, T = input_ids.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    allow = torch.zeros(B, 1, T, T, dtype=torch.bool, device=device)
    if special_ids is not None and special_ids.numel() > 0:
        special_ids = special_ids.to(device)

    tot_masked = 0
    tot_cand = 0
    for b in range(B):
        valid_key = attention_mask[b].bool()                 # [T]
        is_ctx = labels[b].eq(ignore_index)
        a = causal & valid_key.unsqueeze(0)                  # [T, T] (q, k): k<=q & key valid
        a = a.clone()

        # dst -> {src: max weight} from the (augmented) edges.
        src_of: dict[int, dict[int, float]] = {}
        pairs = annot_pairs_batch[b] if b < len(annot_pairs_batch) else None
        if pairs is not None and pairs.numel() > 0:
            pl = pairs.tolist()
            wl = None
            if (annot_weights_batch is not None and b < len(annot_weights_batch)
                    and annot_weights_batch[b] is not None
                    and annot_weights_batch[b].numel() == len(pl)):
                wl = annot_weights_batch[b].tolist()
            if wl is None:
                wl = [1.0] * len(pl)
            for (i, j), w in zip(pl, wl):
                d = src_of.setdefault(int(j), {})
                if float(w) > d.get(int(i), -1.0):
                    d[int(i)] = float(w)

        if special_ids is not None and special_ids.numel() > 0:
            is_special = torch.isin(input_ids[b], special_ids)
        else:
            is_special = torch.zeros(T, dtype=torch.bool, device=device)

        tgt_rows = (~is_ctx & valid_key).nonzero(as_tuple=False).flatten().tolist()
        for q in tgt_rows:
            if q == 0:
                continue
            S = src_of.get(q, {})
            cand = (is_ctx & valid_key & ~is_special).clone()
            cand[0] = False                                  # sink / NaN safety
            if q < T:
                cand[q:] = False                             # keys strictly before q
            # Protect this target's annotation sources. Uniform: protect ALL
            # sources. Weight-aware: protect only FIRST-ORDER (w>=1) sources;
            # transitive sources (w<1) stay maskable with weight-dependent prob,
            # which is what makes the gradation meaningful.
            if S:
                if weight_aware:
                    prot = [i for i, w in S.items() if w >= 1.0 - 1e-9 and 0 <= i < T]
                else:
                    prot = [i for i in S if 0 <= i < T]
                if prot:
                    cand[torch.tensor(prot, dtype=torch.long, device=device)] = False
            if protect_prefix > 0:
                cand[: min(protect_prefix, T)] = False
            if recency_window > 0:
                lo = max(0, q - recency_window)
                cand[lo:q] = False
            cand_idx = cand.nonzero(as_tuple=False).flatten()
            nc = int(cand_idx.numel())
            tot_cand += nc
            if nc == 0:
                continue
            if weight_aware:
                w = torch.tensor([S.get(int(k), 0.0) for k in cand_idx.tolist()],
                                 dtype=torch.float32, device=device).clamp_(0.0, 1.0)
                pm = p_max * (1.0 - w).pow(p_gamma)
                r = (torch.rand(nc, generator=generator, device=device)
                     if generator is not None else torch.rand(nc, device=device))
                chosen = cand_idx[r < pm]
            else:
                k = int(round(rate * nc))
                if k <= 0:
                    continue
                perm = torch.randperm(nc, generator=generator, device=device)[:k]
                chosen = cand_idx[perm]
            if chosen.numel() > 0:
                a[q, chosen] = False
                tot_masked += int(chosen.numel())
        allow[b, 0] = a

    return PerTargetMaskStats(
        allow=allow,
        n_masked=tot_masked / max(1, B),
        n_candidates=tot_cand / max(1, B),
        frac_masked=(tot_masked / tot_cand) if tot_cand > 0 else 0.0,
    )


# ── RRR: Right for the Right Reasons input-gradient penalty (Ross et al. 2017) ─
# Penalize the input-EMBEDDING gradient mass of the CE loss that lands OUTSIDE the
# annotated "right reasons" tokens. Unlike attention-based saliency (cheap but
# gameable; cf. Pruthi 2020), input gradients tie directly to the prediction, so
# they are a more faithful attribution to regularize. Cost: needs DOUBLE BACKPROP
# (the penalty contains d(CE)/d(inputs_embeds) computed with create_graph=True),
# so ~2x a normal step; eager attention is required because fused SDPA kernels
# have no double-backward. Training-only: nothing is added to the model/inference.

def annotation_endpoint_mask(annot_pairs_batch, B: int, T: int, device) -> Tensor:
    """``[B, T]`` bool mask: True where a token is an endpoint (source OR target)
    of any annotation edge — i.e. the dependency-relevant ("right reasons") tokens
    that RRR lets the model rely on. Tokens that are never an endpoint are the ones
    whose input-gradient mass gets penalized."""
    m = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b, pairs in enumerate(annot_pairs_batch):
        if pairs is None or int(getattr(pairs, "numel", lambda: 0)()) == 0:
            continue
        idx = pairs.to(device=device, dtype=torch.long).clamp_(0, T - 1).reshape(-1)
        m[b].index_fill_(0, idx, True)
    return m


def rrr_gradient_penalty(
    input_grad: Tensor,        # [B, T, H] = d(CE)/d(inputs_embeds)
    allowed_mask: Tensor,      # [B, T] 1 = right-reason token (exempt from penalty)
    valid_mask: Tensor,        # [B, T] 1 = real (non-pad) token
    *,
    mode: str = "l2",          # "l2" -> sum_h g^2 ; "l1" -> sum_h |g|
    normalize: bool = True,    # True -> fraction of input-grad mass on wrong tokens (in [0, 1])
) -> Tensor:
    """RRR input-gradient penalty (Ross et al. 2017). Pushes input-gradient mass
    OFF tokens that are not annotation endpoints. With ``normalize`` the result is
    the *fraction* of input-gradient mass on wrong tokens (scale-free, in [0, 1]),
    which keeps the penalty well-conditioned across steps. Returns a scalar in
    ``input_grad.dtype`` (0 if there is nothing to penalize)."""
    g = input_grad.float()
    sal = g.abs().sum(-1) if mode == "l1" else g.pow(2).sum(-1)     # [B, T]
    valid = valid_mask.to(sal.dtype)
    penalize = valid * (1.0 - allowed_mask.to(sal.dtype))          # [B, T]
    num = (sal * penalize).sum()
    if normalize:
        denom = (sal * valid).sum().clamp_min(1e-12)
    else:
        denom = penalize.sum().clamp_min(1.0)
    return (num / denom).to(input_grad.dtype)


# ── Auxiliary edge-prediction objective (GraphCodeBERT / GALLa style) ──────────
# A robust, additive alternative to cfmask. Instead of *masking* shortcut tokens
# or forcing an attribution metric (saliency), add a small bilinear head that
# PREDICTS the annotation edges from the model's own hidden states, as a BCE term
# next to CE. It is additive + fully in-distribution (no perturbed forward), so it
# is far more stable than cfmask, and it forces the representation to *encode* the
# dependency (harder to game than attention; cf. Pruthi et al. ACL 2020). The head
# is training-only and dropped at inference (cf. GALLa's GNN, arXiv:2409.04183).
# Differences vs the papers: (1) decoder-only causal LM -> strictly-causal s<q
# edges, not a bidirectional encoder; (2) edges are per-example token->token
# graph-signal annotations, not parsed AST/DFG; (3) a low-rank biaffine scorer on
# the LM's own last hidden state, no external GNN.

def edge_prediction_loss(
    hidden_states: Tensor,                  # [B, T, H] last hidden state
    annot_pairs,                            # list[[N, 2]] or flat [E, 3]
    src_proj,                               # nn.Linear(H, d, bias=False): key/source
    dst_proj,                               # nn.Linear(H, d, bias=False): query/target
    *,
    neg_weight: float = 1.0,
    neg_sample_k: int = 0,
    temperature: float = 1.0,
    exclude_source_mask: Tensor | None = None,   # [B, T] bool: True = drop from negatives
    generator: torch.Generator | None = None,
) -> Tensor:
    """Decoupled per-edge BCE. For each target (query) token q, push the score of
    its annotated source tokens s (edge s->q, s<q) up and non-annotated causal
    sources down: score(q,s) = <dst_proj(h_q), src_proj(h_s)> / (sqrt(d)*tau).
    No shared softmax denominator (positives don't compete), so it is stable.
    Returns a scalar in ``hidden_states.dtype`` (0 if the batch has no edges)."""
    B, T, _ = hidden_states.shape
    device = hidden_states.device
    row_batch, row_qry, src_all, inv = _annotation_rows_from_pairs(
        annot_pairs, B=B, T=T, device=device)
    if row_qry.numel() == 0:
        return hidden_states.new_zeros(())
    Q = row_qry.numel()
    h = hidden_states.float()
    K = src_proj(h)                                   # [B, T, d]
    d = K.shape[-1]
    qv = dst_proj(h[row_batch, row_qry])              # [Q, d]
    scale = 1.0 / (float(d) ** 0.5) / max(float(temperature), 1e-6)
    if B == 1:
        scores = (qv @ K[0].transpose(0, 1)) * scale          # [Q, T] (avoid [Q,T,d])
    else:
        scores = torch.einsum("qd,qsd->qs", qv, K[row_batch]) * scale
    # positives A_pos[q, s]
    A_adj = torch.zeros(Q, T, device=device, dtype=scores.dtype)
    A_adj.index_put_(
        (inv.to(device=device, dtype=torch.long), src_all.to(device=device, dtype=torch.long)),
        torch.ones_like(src_all, device=device, dtype=scores.dtype), accumulate=True)
    A_pos = (A_adj > 0).to(scores.dtype)
    src_idx = torch.arange(T, device=device).unsqueeze(0)
    qry_col = row_qry.to(device=device, dtype=torch.long).unsqueeze(1)
    M_causal = (src_idx < qry_col).to(scores.dtype)            # strictly before q (s < q)
    pos_mask = M_causal * A_pos
    neg_mask = M_causal * (1.0 - A_pos)
    if exclude_source_mask is not None:
        keep = (1.0 - exclude_source_mask.to(device=device, dtype=scores.dtype)[row_batch]).clamp_(0.0, 1.0)
        neg_mask = neg_mask * keep
    if int(neg_sample_k) > 0:                                  # optional MoCo-style subsample
        k = int(neg_sample_k)
        with torch.no_grad():
            rows = (neg_mask.sum(dim=-1) > k).nonzero(as_tuple=True)[0]
            if rows.numel() > 0:
                w = neg_mask[rows].to(torch.float32) + 1e-12
                w = w * neg_mask[rows].to(torch.float32)
                idx = torch.multinomial(w, num_samples=k, replacement=False, generator=generator)
                newm = torch.zeros_like(neg_mask[rows]); newm.scatter_(1, idx, 1.0)
                neg_mask = neg_mask.clone(); neg_mask[rows] = newm
    valid = pos_mask.sum(dim=-1) > 0
    if not bool(valid.any()):
        return hidden_states.new_zeros(())
    # logits BCE: positives -> softplus(-s) = -log sigma(s); negatives -> softplus(+s)
    pos_term = (F.softplus(-scores) * pos_mask).sum(-1) / pos_mask.sum(-1).clamp_min(1.0)
    neg_term = (F.softplus(scores) * neg_mask).sum(-1) / neg_mask.sum(-1).clamp_min(1.0)
    loss = (pos_term + float(neg_weight) * neg_term)[valid].mean()
    return loss.to(hidden_states.dtype)


# ── Soft additive graph attention bias (T5 / ALiBi style, keyed on annotations) ─
# Instead of MASKING shortcut keys to -inf (cfmask), ADD a small *learned* positive
# bias g*w to the attention logits of annotated edges (key s -> query q, s<q),
# passed as a 4-D additive attention mask. The gate g then receives gradient from
# CE through the softmax and is learned end-to-end (it can go ->0 where unhelpful,
# degrading gracefully instead of hurting a strong baseline). At inference with no
# annotation graph the bias is simply not applied (the LM runs unchanged), so the
# benefit is internalized into the weights (cf. GALLa "structure only at
# training"). Differences vs ALiBi/T5: the bias is keyed on a per-example
# dependency graph rather than relative position, and is a single shared learnable
# scalar gate (optionally scaled by the augmented-edge weight), not a fixed
# per-head slope.

def build_graph_attention_bias(
    input_ids: Tensor,                       # [B, T]
    attention_mask: Tensor,                  # [B, T] (1 = real, 0 = pad)
    annot_pairs_batch,                       # per-sample [N, 2] (qi, qj) with qi < qj
    gate,                                    # scalar Tensor / nn.Parameter (trainable)
    *,
    annot_weights_batch=None,                # per-sample [N] edge weights, or None
    comp_dtype=torch.bfloat16,
) -> Tensor:
    """Return a ``[B, 1, T, T]`` additive attention bias: ``-inf`` on disallowed
    (future / padding) keys, ``0`` on allowed non-annotated keys, and
    ``+gate*weight`` on allowed annotated edges (key s -> query q). Differentiable
    w.r.t. ``gate`` (built with ``torch.where`` + ``stack``, no in-place into a
    detached buffer)."""
    device = input_ids.device
    B, T = input_ids.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    neg_t = torch.tensor(torch.finfo(comp_dtype).min, dtype=comp_dtype, device=device)
    rows = []
    for b in range(B):
        valid_key = attention_mask[b].bool()
        allow = causal & valid_key.unsqueeze(0)                # [T(q), T(k)]
        W = torch.zeros(T, T, dtype=torch.float32, device=device)   # W[q, s]
        pairs = annot_pairs_batch[b] if b < len(annot_pairs_batch) else None
        if pairs is not None and pairs.numel() > 0:
            qi = pairs[:, 0].to(device=device, dtype=torch.long).clamp_(0, T - 1)   # src (earlier)
            qj = pairs[:, 1].to(device=device, dtype=torch.long).clamp_(0, T - 1)   # dst (later)
            if (annot_weights_batch is not None and b < len(annot_weights_batch)
                    and annot_weights_batch[b] is not None
                    and annot_weights_batch[b].numel() == pairs.shape[0]):
                wv = annot_weights_batch[b].to(device=device, dtype=torch.float32)
            else:
                wv = torch.ones(pairs.shape[0], dtype=torch.float32, device=device)
            W[qj, qi] = wv                                      # query qj attends key qi
        bonus = (gate * W).to(comp_dtype)                      # differentiable through gate
        rows.append(torch.where(allow, bonus, neg_t))
    return torch.stack(rows, dim=0).unsqueeze(1)               # [B, 1, T, T]


def shortcut_invariance_kl(
    logits_clean: Tensor,   # [B, T, V]
    logits_masked: Tensor,  # [B, T, V]
    labels: Tensor,         # [B, T]
    *,
    ignore_index: int = IGNORE_INDEX,
    only_clean_correct: bool = True,
) -> Tensor:
    """KL( p_clean(stop-grad) || p_masked ) averaged over target prediction rows.

    Uses the standard next-token shift: the distribution that predicts gold token
    at position j lives at logits[:, j-1]. By default the term is applied only
    where the *clean* model already predicts the gold token, so the objective is
    "removing shortcuts must not change answers you already get right" — robust to
    incomplete annotations (it never asserts a possibly-wrong target under heavy
    masking). Gradient flows through ``logits_masked`` only.
    """
    lc = logits_clean[:, :-1, :].float()
    lm = logits_masked[:, :-1, :].float()
    gold = labels[:, 1:]
    valid = gold.ne(ignore_index)
    if only_clean_correct:
        valid = valid & (lc.argmax(dim=-1) == gold)
    if valid.sum() == 0:
        return logits_clean.new_zeros(())
    lc_v = lc[valid]
    lm_v = lm[valid]
    logp_clean = F.log_softmax(lc_v, dim=-1)
    p_clean = logp_clean.exp().detach()
    logp_clean = logp_clean.detach()
    logp_masked = F.log_softmax(lm_v, dim=-1)
    kl = (p_clean * (logp_clean - logp_masked)).sum(dim=-1)
    return kl.mean()
