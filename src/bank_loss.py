"""Train-bank loss helpers aligned with viz/data_attribution.py.

ce_only     -> CE
ce_saliency -> CE + λ * contrastive saliency loss (needs attention_edges)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class BankLossConfig:
    loss_mode: str = "ce_only"  # ce_only | ce_saliency
    saliency_loss_type: str = "contrastive"
    saliency_lambda: float = 1.5
    alpha: float = 1.0
    eps: float = 1e-8
    margin_plus: float = 2.0
    neg_sample_k: int = 0
    saliency_layer: int = -1
    exclude_sink_prefix: int = 0
    exclude_special_tokens: bool = False

    @property
    def cache_tag(self) -> str:
        if self.loss_mode == "ce_only":
            return "ce"
        return (
            f"cesal_{self.saliency_loss_type}_lam{self.saliency_lambda:g}"
            f"_m{self.margin_plus:g}_k{self.neg_sample_k}"
        )


def infer_bank_loss_mode(model_path: str | None, model_tag: str) -> str:
    """Prefer adapter/model-local saliency_training_config.json; else infer from tag."""
    if model_path:
        cfg_path = Path(model_path) / "saliency_training_config.json"
        if cfg_path.is_file():
            try:
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
                mode = str(raw.get("loss_mode") or "").strip()
                if mode in ("ce_only", "ce_saliency", "saliency_only"):
                    return "ce_saliency" if mode != "ce_only" else "ce_only"
            except Exception:
                pass
    tag = (model_tag or "").lower()
    if "ce_only" in tag or tag.endswith("_ce") or tag == "ce":
        return "ce_only"
    if "saliency" in tag or "contrastive" in tag:
        return "ce_saliency"
    return "ce_only"


def load_bank_loss_config(model_path: str | None, model_tag: str) -> BankLossConfig:
    raw: dict = {}
    if model_path:
        cfg_path = Path(model_path) / "saliency_training_config.json"
        if cfg_path.is_file():
            try:
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                raw = {}
    mode = infer_bank_loss_mode(model_path, model_tag)
    return BankLossConfig(
        loss_mode=mode,
        saliency_loss_type=str(raw.get("saliency_loss_type") or "contrastive"),
        saliency_lambda=float(raw.get("saliency_lambda", 1.5)),
        alpha=float(raw.get("saliency_temperature_tau", raw.get("saliency_alpha", 1.0))),
        eps=float(raw.get("saliency_eps_num", 1e-8)),
        margin_plus=float(raw.get("saliency_margin_plus", 2.0)),
        neg_sample_k=int(raw.get("saliency_neg_sample_k", 0) or 0),
        saliency_layer=int(raw.get("saliency_layer", -1)),
        exclude_sink_prefix=int(raw.get("saliency_exclude_sink_prefix", 0) or 0),
        exclude_special_tokens=bool(raw.get("saliency_exclude_special_tokens", False)),
    )


def _import_saliency_loss_from_outputs():
    """Load contrastive saliency loss used by ce_saliency train-bank grads.

    Prefers the vendored copy at ``src/saliency_loss.py`` (moved out of
    code-corr-annotation). Falls back to the old CCA path if present.
    """
    try:
        from src.saliency_loss import saliency_loss_from_outputs
        return saliency_loss_from_outputs
    except ImportError:
        pass

    # Fallback: legacy code-corr-annotation/src/train/loss.py
    import sys

    here = Path(__file__).resolve().parent
    root = here.parent
    candidates = [
        root / "code-corr-annotation" / "src" / "train",
        root.parent / "code-corr-annotation" / "src" / "train",
    ]
    for train_dir in candidates:
        if (train_dir / "loss.py").is_file():
            train_dir_s = str(train_dir)
            if train_dir_s not in sys.path:
                sys.path.insert(0, train_dir_s)
            from loss import saliency_loss_from_outputs  # type: ignore
            return saliency_loss_from_outputs

    raise ImportError(
        "Cannot import saliency_loss_from_outputs. Expected src/saliency_loss.py "
        "(or legacy code-corr-annotation/src/train/loss.py)."
    )


def _annot_pairs_from_edges(edges, n_tokens: int, device) -> list[torch.Tensor]:
    pairs = []
    for e in edges or []:
        try:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                a, b = int(e[0]), int(e[1])
            else:
                a = int(e.get("src", e.get("source", -1)))
                b = int(e.get("dst", e.get("target", -1)))
        except (TypeError, ValueError, AttributeError):
            continue
        qi, qj = (a, b) if a < b else (b, a)
        if 0 <= qi < qj < n_tokens:
            pairs.append([qi, qj])
    if not pairs:
        return [torch.zeros(0, 2, dtype=torch.long, device=device)]
    return [torch.tensor(pairs, dtype=torch.long, device=device)]


def compute_bank_loss(
    model,
    batch,
    *,
    device,
    cfg: BankLossConfig,
    edges=None,
    special_ids: set[int] | None = None,
):
    """Scalar training objective for one bank example (CE or CE+saliency)."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    inputs = {"input_ids": input_ids, "labels": labels}
    if "attention_mask" in batch:
        inputs["attention_mask"] = batch["attention_mask"].to(device)

    need_saliency = cfg.loss_mode == "ce_saliency" and bool(edges)
    if not need_saliency:
        outputs = model(**inputs, use_cache=False, return_dict=True)
        loss = outputs.loss
        if loss is None:
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return loss, "ce_only"

    outputs = model(
        **inputs,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    ce = outputs.loss
    if ce is None:
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    if ce.dim() > 0:
        ce = ce.mean()

    saliency_loss_from_outputs = _import_saliency_loss_from_outputs()
    n_tokens = int(input_ids.size(1))
    annot_pairs = _annot_pairs_from_edges(edges, n_tokens, device)
    exclude = None
    if cfg.exclude_sink_prefix > 0 or (cfg.exclude_special_tokens and special_ids):
        em = torch.zeros_like(input_ids, dtype=torch.bool)
        if cfg.exclude_sink_prefix > 0:
            em[:, : cfg.exclude_sink_prefix] = True
        if cfg.exclude_special_tokens and special_ids:
            special = torch.tensor(sorted(special_ids), device=input_ids.device, dtype=input_ids.dtype)
            em = em | torch.isin(input_ids, special)
        exclude = em

    diag = saliency_loss_from_outputs(
        model,
        outputs,
        annot_pairs,
        saliency_layer=cfg.saliency_layer,
        exclude_source_mask=exclude,
        alpha=cfg.alpha,
        eps=cfg.eps,
        floor_eps=0.0,
        floor_eps_mode="fixed",
        floor_eps_step=0,
        floor_eps_warmup_steps=0,
        floor_logit_eps=None,
        loss_type=cfg.saliency_loss_type,
        margin_plus=cfg.margin_plus,
        neg_sample_k=cfg.neg_sample_k,
    )
    return ce + float(cfg.saliency_lambda) * diag.loss, "ce_saliency"
