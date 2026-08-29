from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from attn_viz import AttentionVisualizationCallback
from dataset import AnnotatedSFTDataset, DataCollatorForAnnotatedSFT, IGNORE_INDEX
from loss import (
    canonical_saliency_loss_type,
    canonical_saliency_agg,
    saliency_loss_display_name,
    saliency_loss_from_outputs,
    capture_attn_proj_weights,
    build_shortcut_mask,
    build_shortcut_mask_per_target,
    shortcut_invariance_kl,
    edge_prediction_loss,
    build_graph_attention_bias,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvalBreakdownCallback(TrainerCallback):
    """Print average eval ntp/sal/total loss + saliency mAP@k after each
    Trainer.evaluate() call. Reads from buffers populated by
    AnnotatedSFTTrainer.compute_loss() during eval-mode forwards.
    """
    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer") or kwargs.get("model_wrapped")  # not always present
        # The HF callback signature does not always include the trainer; fall
        # back to using the global last-seen trainer via a thread-local-ish hack.
        if trainer is None or not hasattr(trainer, "_eval_ntp_buf"):
            t = getattr(EvalBreakdownCallback, "_trainer", None)
            if t is None:
                return
            trainer = t
        ntp = trainer._eval_ntp_buf
        sal = trainer._eval_sal_buf
        tot = trainer._eval_total_buf
        ratio = trainer._eval_ratio_buf
        mAP = trainer._eval_mAP_buf
        rec = trainer._eval_recall_at_k_buf
        prec = trainer._eval_precision_at_k_buf
        if not ntp:
            return
        def avg(xs): return float(np.mean(xs)) if xs else 0.0
        payload = {
            "eval_ntp_loss": avg(ntp),
            "eval_saliency_loss": avg(sal),
            "eval_total_loss": avg(tot),
            "eval_saliency_ratio": avg(ratio) if ratio else 0.0,
            "eval_saliency_mAP_at_k": avg(mAP) if mAP else 0.0,
            "eval_saliency_recall_at_k": avg(rec) if rec else 0.0,
            "eval_saliency_precision_at_k": avg(prec) if prec else 0.0,
            "eval_n_batches": len(ntp),
            "step": int(state.global_step),
        }
        try:
            trainer.log(payload)
        except Exception:
            logger.info(json.dumps(payload))
        # reset
        trainer._eval_ntp_buf.clear()
        trainer._eval_sal_buf.clear()
        trainer._eval_total_buf.clear()
        trainer._eval_ratio_buf.clear()
        trainer._eval_mAP_buf.clear()
        trainer._eval_recall_at_k_buf.clear()
        trainer._eval_precision_at_k_buf.clear()


class ValidCodeBLEUCallback(TrainerCallback):
    """Compute greedy-generation CodeBLEU on a fixed small valid subset after each
    Trainer.evaluate(). Off unless `eval_codebleu_samples > 0`. Mirrors the eval
    prefix policy: prefix = tokens before the first target token; reference =
    target span (specials stripped). Logs `eval_codebleu`.
    """
    def __init__(self, eval_dataset, tokenizer, n_samples: int = 64,
                 max_new_tokens: int = 128, lang: str = "go"):
        self.items = list(getattr(eval_dataset, "items", []))[:n_samples]
        self.tok = tokenizer
        self.max_new_tokens = max_new_tokens
        self.lang = lang
        try:
            from codebleu import calc_codebleu  # type: ignore
            self._calc = calc_codebleu
        except Exception:
            self._calc = None

    def on_evaluate(self, args, state, control, **kwargs):
        if self._calc is None or not self.items:
            return
        trainer = getattr(EvalBreakdownCallback, "_trainer", None) or kwargs.get("trainer")
        model = kwargs.get("model") or (trainer.model if trainer is not None else None)
        if model is None:
            return
        tok = self.tok
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        refs, preds = [], []
        try:
            with torch.no_grad():
                for it in self.items:
                    ids = it["input_ids"].tolist() if hasattr(it["input_ids"], "tolist") else list(it["input_ids"])
                    lbl = it["labels"].tolist() if hasattr(it["labels"], "tolist") else list(it["labels"])
                    tgt_pos = [j for j, l in enumerate(lbl) if l != IGNORE_INDEX]
                    if not tgt_pos:
                        continue
                    first = tgt_pos[0]
                    prefix = ids[:first]
                    ref = tok.decode([ids[j] for j in tgt_pos], skip_special_tokens=True).strip()
                    inp = torch.tensor([prefix], device=device)
                    attn = torch.ones_like(inp)
                    out = model.generate(input_ids=inp, attention_mask=attn,
                                         max_new_tokens=self.max_new_tokens, do_sample=False,
                                         pad_token_id=pad_id, eos_token_id=tok.eos_token_id)
                    pred = tok.decode(out[0, len(prefix):], skip_special_tokens=True)
                    for stop in ("<|im_end|>", "\n```"):
                        if stop in pred:
                            pred = pred.split(stop, 1)[0]
                    refs.append(ref); preds.append(pred.strip())
            if refs:
                cb = self._calc(references=refs, predictions=preds, lang=self.lang)
                payload = {"eval_codebleu": float(cb.get("codebleu", 0.0)), "step": int(state.global_step)}
                try:
                    (trainer.log if trainer is not None else logger.info)(payload)
                except Exception:
                    logger.info(json.dumps(payload))
        except Exception as e:
            # Optional CodeBLEU diagnostic must never crash training (else the
            # checkpoint never saves). Log and continue. Seen: tree-sitter /
            # codebleu version skew -> Language(...) TypeError at end-of-train eval.
            logger.warning(
                f"[EvalBreakdownCallback] CodeBLEU diagnostic skipped (non-fatal): "
                f"{type(e).__name__}: {e}"
            )
        finally:
            if was_training:
                model.train()


# ── Custom Trainer ────────────────────────────────────────────────────────────


class AnnotatedSFTTrainer(Trainer):
    """
        NTP loss + saliency loss.
    """
    def __init__(self, *args,
                 saliency_lambda: float = 0.1,
                 saliency_alpha: float = 1.5,
                 saliency_eps: float = 1e-8,
                 saliency_floor_eps: float = 0.0,
                 saliency_floor_eps_mode: str = "ema_quantile",
                 saliency_floor_quantile: float = 0.75,
                 saliency_floor_ema_beta: float = 0.95,
                 saliency_floor_min_eps: float = 1e-8,
                 saliency_floor_warmup_steps: int = 10,
                 saliency_floor_logit_eps: float | None = None,
                 saliency_loss_type: str = "softmax_margin",
                 loss_mode: str = "ce_saliency",
                 saliency_detail_log_path: str = "",
                 saliency_detail_log_steps: int = 0,
                 saliency_detail_top_k: int = 10,
                 saliency_margin_plus: float = 2.08,
                 saliency_margin_minus: float = 0.41,
                 saliency_margin_gamma: float = 2.0,
                 saliency_neg_weight: float = 0.5,
                 saliency_neg_hard_only: bool = False,
                 saliency_neg_sample_k: int = 0,
                 saliency_layer: int = -1,
                 saliency_agg: str = "last",
                 saliency_exclude_sink_prefix: int = 0,
                 saliency_exclude_special_tokens: bool = False,
                 saliency_oom_fallback: bool = True,
                 saliency_fallback_max_seq_len: int = 1280,
                 cfmask_rate: float = 0.3,
                 cfmask_max_k: int = 0,
                 cfmask_min_k: int = 0,
                 cfmask_recency_window: int = 8,
                 cfmask_protect_prefix: int = 0,
                 cfmask_exclude_special: bool = True,
                 cfmask_invariance_beta: float = 0.0,
                 cfmask_weight_aware: bool = False,
                 cfmask_p_max: float = 0.9,
                 cfmask_p_gamma: float = 1.0,
                 cfmask_per_target: bool = False,
                 edge_lambda: float = 0.5,
                 edge_proj_dim: int = 256,
                 edge_neg_weight: float = 1.0,
                 edge_neg_sample_k: int = 0,
                 edge_temperature: float = 1.0,
                 edge_layer: int = -1,
                 attn_bias_init: float = 1.0,
                 eos_loss_weight: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.eos_loss_weight = float(eos_loss_weight if eos_loss_weight is not None else 1.0)
        tok = (
            getattr(self, "processing_class", None)
            or getattr(self, "tokenizer", None)
            or kwargs.get("processing_class")
            or kwargs.get("tokenizer")
        )
        self._stop_token_ids: set[int] = set()
        if tok is not None:
            for name in ("<|im_end|>", getattr(tok, "eos_token", None)):
                if not name:
                    continue
                tid = tok.convert_tokens_to_ids(name)
                if isinstance(tid, int) and tid >= 0 and tid != getattr(tok, "unk_token_id", None):
                    self._stop_token_ids.add(int(tid))
            if getattr(tok, "eos_token_id", None) is not None:
                self._stop_token_ids.add(int(tok.eos_token_id))
        # Hardcode Qwen chat end id as fallback (common in this repo's compact data).
        if not self._stop_token_ids:
            self._stop_token_ids.add(151645)
        self.saliency_lambda = saliency_lambda
        self.saliency_alpha = saliency_alpha
        self.saliency_eps = saliency_eps
        self.saliency_floor_eps = saliency_floor_eps
        self.saliency_floor_eps_mode = (saliency_floor_eps_mode or "fixed").strip().lower()
        self.saliency_floor_quantile = saliency_floor_quantile
        self.saliency_floor_ema_beta = saliency_floor_ema_beta
        self.saliency_floor_min_eps = saliency_floor_min_eps
        self.saliency_floor_warmup_steps = saliency_floor_warmup_steps
        self.saliency_floor_logit_eps = saliency_floor_logit_eps
        self.saliency_loss_type = canonical_saliency_loss_type(saliency_loss_type)
        self.saliency_floor_eps_ema: float | None = None
        self.saliency_floor_step = 0
        self.loss_mode = loss_mode
        self.saliency_detail_log_path = saliency_detail_log_path
        self.saliency_detail_log_steps = saliency_detail_log_steps
        self.saliency_detail_top_k = saliency_detail_top_k
        self.saliency_margin_plus = saliency_margin_plus
        self.saliency_margin_minus = saliency_margin_minus
        self.saliency_margin_gamma = saliency_margin_gamma
        self.saliency_neg_weight = saliency_neg_weight
        self.saliency_neg_hard_only = saliency_neg_hard_only
        self.saliency_neg_sample_k = int(saliency_neg_sample_k or 0)
        self.saliency_layer = int(saliency_layer)
        self.saliency_agg = canonical_saliency_agg(saliency_agg)
        self.saliency_exclude_sink_prefix = int(saliency_exclude_sink_prefix or 0)
        self.saliency_exclude_special_tokens = bool(saliency_exclude_special_tokens)
        # If saliency/eager-attn OOMs on a long/dense sample, drop saliency for
        # this step and keep CE so full-corpus training does not die.
        self.saliency_oom_fallback = bool(saliency_oom_fallback)
        self.saliency_fallback_max_seq_len = int(saliency_fallback_max_seq_len or 0)
        self._saliency_oom_fallback_count = 0
        self._saliency_preempt_fallback_count = 0
        # Counterfactual shortcut-masking augmentation (loss_mode=ce_shortcut_mask)
        self.cfmask_rate = float(cfmask_rate)
        self.cfmask_max_k = int(cfmask_max_k or 0)
        self.cfmask_min_k = int(cfmask_min_k or 0)
        self.cfmask_recency_window = int(cfmask_recency_window or 0)
        self.cfmask_protect_prefix = int(cfmask_protect_prefix or 0)
        self.cfmask_exclude_special = bool(cfmask_exclude_special)
        self.cfmask_invariance_beta = float(cfmask_invariance_beta or 0.0)
        self.cfmask_weight_aware = bool(cfmask_weight_aware)
        self.cfmask_p_max = float(cfmask_p_max if cfmask_p_max is not None else 0.9)
        self.cfmask_p_gamma = float(cfmask_p_gamma if cfmask_p_gamma is not None else 1.0)
        self.cfmask_per_target = bool(cfmask_per_target)
        self._cfmask_special_ids = None
        if self.cfmask_exclude_special:
            tk = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
            sp = sorted({int(i) for i in getattr(tk, "all_special_ids", []) or []})
            if sp:
                self._cfmask_special_ids = torch.tensor(sp, dtype=torch.long)
        # Special-token ids (sink / role markers) to drop from the saliency
        # NEGATIVE set. Derived once from the tokenizer; None if not excluding.
        self._special_source_ids = None
        if self.saliency_exclude_special_tokens:
            tk = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
            sp = sorted({int(i) for i in getattr(tk, "all_special_ids", []) or []})
            if sp:
                self._special_source_ids = torch.tensor(sp, dtype=torch.long)
        # Buffers for per-eval breakdown of NTP / saliency loss. Filled by
        # compute_loss() when model.training is False, read+reset by
        # EvalBreakdownCallback in on_evaluate.
        self._eval_ntp_buf: list[float] = []
        self._eval_sal_buf: list[float] = []
        self._eval_total_buf: list[float] = []
        self._eval_ratio_buf: list[float] = []
        self._eval_mAP_buf: list[float] = []
        self._eval_recall_at_k_buf: list[float] = []
        self._eval_precision_at_k_buf: list[float] = []
        if self.saliency_detail_log_path and self.is_world_process_zero():
            os.makedirs(os.path.dirname(self.saliency_detail_log_path), exist_ok=True)

        # ── Auxiliary edge-prediction head (ce_edge_pred) + soft graph
        #    attention-bias gate (ce_attn_bias). Both are TRAINING-ONLY signals,
        #    dropped at inference; their params live on the Trainer (not the saved
        #    adapter) and are added to the optimizer in create_optimizer(). Created
        #    only for the mode that needs them, so default paths are unchanged.
        self.edge_lambda = float(edge_lambda)
        self.edge_proj_dim = int(edge_proj_dim)
        self.edge_neg_weight = float(edge_neg_weight)
        self.edge_neg_sample_k = int(edge_neg_sample_k or 0)
        self.edge_temperature = float(edge_temperature)
        self.edge_layer = int(edge_layer)
        self.edge_src_proj = None
        self.edge_dst_proj = None
        if self.loss_mode == "ce_edge_pred":
            hsz = int(self.model.config.hidden_size)
            self.edge_src_proj = torch.nn.Linear(hsz, self.edge_proj_dim, bias=False).to(
                device=self.args.device, dtype=torch.float32)
            self.edge_dst_proj = torch.nn.Linear(hsz, self.edge_proj_dim, bias=False).to(
                device=self.args.device, dtype=torch.float32)
        self.attn_bias_init = float(attn_bias_init)
        self.attn_bias_gate = None
        if self.loss_mode == "ce_attn_bias":
            self.attn_bias_gate = torch.nn.Parameter(
                torch.tensor(self.attn_bias_init, dtype=torch.float32, device=self.args.device))
        self._extra_optim_params_added = False

        # Step 1 needs real attention probabilities; SDPA / FA-2 hide them.
        # Only the saliency objectives consume attention probs, so the eager
        # requirement applies to them alone. ce_only / ce_shortcut_mask (cfmask)
        # use logits only and run fine under SDPA (needed for 14B/32B memory).
        if self.loss_mode in ("ce_saliency", "saliency_only", "ce_shortcut_mask_saliency"):
            attn_impl = getattr(self.model.config, "_attn_implementation", None)
            assert attn_impl == "eager", (
                f"saliency loss_mode={self.loss_mode!r} needs attn_implementation='eager', "
                f"got {attn_impl!r}. Re-init the model with attn_implementation='eager'."
            )

    def create_optimizer(self):
        """Build the base optimizer, then add the training-only auxiliary params
        (edge-prediction head / attention-bias gate) as an extra param group so
        they are actually optimized. They are NOT part of the model/adapter, so
        they are not saved or used at inference (GALLa-style)."""
        optimizer = super().create_optimizer()
        if getattr(self, "_extra_optim_params_added", False):
            return optimizer
        extra = []
        if getattr(self, "edge_src_proj", None) is not None:
            extra += list(self.edge_src_proj.parameters()) + list(self.edge_dst_proj.parameters())
        if getattr(self, "attn_bias_gate", None) is not None:
            extra += [self.attn_bias_gate]
        extra = [p for p in extra if p.requires_grad]
        if extra:
            optimizer.add_param_group({"params": extra, "weight_decay": 0.0})
            logger.info("Added %d auxiliary (edge/attn-bias) params to the optimizer.", len(extra))
        self._extra_optim_params_added = True
        return optimizer

    def _ntp_loss_from_outputs(self, outputs, labels: torch.Tensor) -> torch.Tensor:
        """Standard mean CE, optionally up-weighting stop tokens (``eos_loss_weight``)."""
        if self.eos_loss_weight == 1.0 or not self._stop_token_ids:
            loss = outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()
            return loss
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab = shift_logits.size(-1)
        token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        ).view_as(shift_labels)
        weights = torch.ones_like(token_loss)
        stop = torch.zeros_like(shift_labels, dtype=torch.bool)
        for sid in self._stop_token_ids:
            stop |= shift_labels.eq(sid)
        weights = torch.where(stop, weights * self.eos_loss_weight, weights)
        valid = shift_labels.ne(IGNORE_INDEX)
        denom = valid.sum().clamp_min(1)
        return (token_loss * weights * valid).sum() / denom

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        if not isinstance(exc, RuntimeError):
            return False
        msg = str(exc).lower()
        return any(
            k in msg
            for k in (
                "out of memory",
                "oom",
                "npu out of memory",
                "cuda out of memory",
                "hip out of memory",
            )
        )

    @staticmethod
    def _empty_accel_cache() -> None:
        try:
            if hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available():
                torch.npu.empty_cache()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _ce_only_forward(self, model, inputs):
        """Memory-cheap forward for NTP/CE (no attentions / hidden states)."""
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_attentions=False,
            output_hidden_states=False,
        )
        return outputs, self._ntp_loss_from_outputs(outputs, inputs["labels"])

    @staticmethod
    def _strip_heavy_outputs(outputs) -> None:
        """Drop attention / hidden-state tensors that dominate peak memory."""
        if outputs is None:
            return
        for attr in ("attentions", "hidden_states", "past_key_values"):
            try:
                if getattr(outputs, attr, None) is not None:
                    setattr(outputs, attr, None)
            except Exception:
                pass

    def _should_preempt_saliency(self, inputs) -> bool:
        cap = int(self.saliency_fallback_max_seq_len or 0)
        if cap <= 0:
            return False
        return int(inputs["input_ids"].size(1)) >= cap

    def _build_exclude_source_mask(self, input_ids):
        """[B, T] bool mask: True = drop this source token from the saliency
        NEGATIVE set. Covers the attention-sink prefix (first
        ``saliency_exclude_sink_prefix`` positions) and special / role tokens.
        Returns None when no exclusion is configured."""
        want_prefix = self.saliency_exclude_sink_prefix > 0
        want_special = self.saliency_exclude_special_tokens and self._special_source_ids is not None
        if not (want_prefix or want_special):
            return None
        em = torch.zeros_like(input_ids, dtype=torch.bool)
        if want_prefix:
            em[:, : self.saliency_exclude_sink_prefix] = True
        if want_special:
            em = em | torch.isin(input_ids, self._special_source_ids.to(input_ids.device))
        return em

    def _saliency_term(self, model, input_ids, attention_mask, annot_pairs_batch):
        """Clean (unmasked) eager forward + contrastive saliency loss, used by the
        combined ce_shortcut_mask_saliency objective. Saliency must be measured on
        the CLEAN input: the cfmask masked forward hides the negative context, which
        would trivially satisfy the saliency objective. Returns (saliency_loss, diag)."""
        all_layers = canonical_saliency_agg(self.saliency_agg) == "rollout"
        with capture_attn_proj_weights(
            model, layer_index=self.saliency_layer, all_layers=all_layers,
        ) as attn_proj_weights:
            clean = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
            )
        floor_step = self.saliency_floor_step + 1
        exclude_source_mask = self._build_exclude_source_mask(input_ids)
        diag = saliency_loss_from_outputs(
            model,
            clean,
            annot_pairs_batch,
            saliency_layer=self.saliency_layer,
            saliency_agg=self.saliency_agg,
            exclude_source_mask=exclude_source_mask,
            alpha=self.saliency_alpha,
            eps=self.saliency_eps,
            floor_eps=self.saliency_floor_eps,
            floor_eps_mode=self.saliency_floor_eps_mode,
            floor_eps_quantile=self.saliency_floor_quantile,
            floor_eps_ema_beta=self.saliency_floor_ema_beta,
            prev_floor_eps=self.saliency_floor_eps_ema,
            floor_eps_min=self.saliency_floor_min_eps,
            floor_eps_step=floor_step,
            floor_eps_warmup_steps=self.saliency_floor_warmup_steps,
            floor_logit_eps=self.saliency_floor_logit_eps,
            loss_type=self.saliency_loss_type,
            margin_plus=self.saliency_margin_plus,
            margin_minus=self.saliency_margin_minus,
            margin_gamma=self.saliency_margin_gamma,
            neg_weight=self.saliency_neg_weight,
            neg_hard_only=self.saliency_neg_hard_only,
            neg_sample_k=self.saliency_neg_sample_k,
            attn_proj_weights=attn_proj_weights,
        )
        self.saliency_floor_step = floor_step
        if (
            self.saliency_loss_type != "softmax"
            and self.saliency_floor_logit_eps is None
            and self.saliency_floor_eps_mode == "ema_quantile"
            and diag.n_queries > 0
        ):
            self.saliency_floor_eps_ema = diag.floor_eps
        # Optional saliency-alignment diagnostic: recall@k / hit@k / mAP@k of the
        # annotation edges among each target's top-k most-salient sources. Logged
        # every saliency_detail_log_steps to train.log + saliency_detail.jsonl.
        # Computed once per optimizer step (guarded) to bound the extra cost.
        do_detail = (
            bool(self.saliency_detail_log_path)
            and self.saliency_detail_log_steps > 0
            and self.state.global_step % self.saliency_detail_log_steps == 0
            and self.state.global_step != getattr(self, "_last_detail_step", -1)
            and self.is_world_process_zero()
        )
        if do_detail:
            self._last_detail_step = self.state.global_step
            from saliency_diagnostics import saliency_details_from_outputs
            detail = saliency_details_from_outputs(
                model,
                clean,
                annot_pairs_batch,
                alpha=self.saliency_alpha,
                eps=self.saliency_eps,
                floor_eps=diag.floor_eps,
                floor_logit_eps=diag.floor_logit_eps if diag.floor_eps_kind == "logit" else None,
                top_k=self.saliency_detail_top_k,
            )
            logger.info(
                f"[saliency-align] step={self.state.global_step} "
                f"recall@{self.saliency_detail_top_k}={detail.recall_at_k:.4f} "
                f"hit@k={detail.hit_at_k:.4f} precision@k={detail.precision_at_k:.4f} "
                f"mAP@k={detail.map_at_k:.4f} n_edges={detail.num_annotation_edges} nq={detail.n_queries}"
            )
            with open(self.saliency_detail_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": int(self.state.global_step),
                    "loss_mode": self.loss_mode,
                    "saliency_loss": float(diag.loss.detach().cpu()),
                    "recall_at_k": detail.recall_at_k,
                    "hit_at_k": detail.hit_at_k,
                    "precision_at_k": detail.precision_at_k,
                    "map_at_k": detail.map_at_k,
                    "top_k": self.saliency_detail_top_k,
                    "num_annotation_edges": detail.num_annotation_edges,
                    "n_queries": detail.n_queries,
                }) + "\n")
        return diag.loss, diag

    def _compute_shortcut_mask_loss(self, model, inputs, annot_pairs_batch, return_outputs,
                                    node_weight_batch=None, annot_weights_batch=None):
        """Counterfactual shortcut-masking objective (loss_mode=ce_shortcut_mask).

        Hide a random subset of non-annotated context tokens as attention keys,
        then:
          - beta == 0 (Variant A): plain CE on the real target under the masked
            input. Forces the model to recover the target from surviving
            annotated tokens. One forward pass.
          - beta  > 0 (Variant B): CE on the *clean* input (keeps capability) +
            beta * KL(clean || masked) applied only where the clean model is
            already correct (robustness without asserting a possibly-wrong target).
            Two forward passes.

        cfmask_per_target=False -> one global [B,T] key mask (all queries share it).
        cfmask_per_target=True  -> a 4-D [B,1,T,T] mask where each target query row
        hides only its own non-annotation keys (single forward, leaky variant A).
        """
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]

        if self.cfmask_per_target:
            pstats = build_shortcut_mask_per_target(
                input_ids,
                labels,
                attention_mask,
                annot_pairs_batch,
                annot_weights_batch=annot_weights_batch,
                rate=self.cfmask_rate,
                recency_window=self.cfmask_recency_window,
                protect_prefix=self.cfmask_protect_prefix,
                special_ids=self._cfmask_special_ids,
                ignore_index=IGNORE_INDEX,
                weight_aware=self.cfmask_weight_aware,
                p_max=self.cfmask_p_max,
                p_gamma=self.cfmask_p_gamma,
            )
            # Convert the boolean visibility map to an additive attention bias in
            # the model's compute dtype. SDPA requires the bias dtype to match the
            # query dtype exactly: that's the autocast dtype when mixed precision
            # is on, else the model's loaded dtype (here bf16 via torch_dtype).
            if self.args.bf16:
                comp_dtype = torch.bfloat16
            elif self.args.fp16:
                comp_dtype = torch.float16
            else:
                comp_dtype = model.get_input_embeddings().weight.dtype
            neg = torch.finfo(comp_dtype).min
            masked_attn = torch.zeros_like(pstats.allow, dtype=comp_dtype)
            masked_attn.masked_fill_(~pstats.allow, neg)
            n_masked, n_candidates, frac_masked = (
                pstats.n_masked, pstats.n_candidates, pstats.frac_masked)
        else:
            stats = build_shortcut_mask(
                input_ids,
                labels,
                attention_mask,
                annot_pairs_batch,
                rate=self.cfmask_rate,
                max_k=self.cfmask_max_k,
                min_k=self.cfmask_min_k,
                recency_window=self.cfmask_recency_window,
                protect_prefix=self.cfmask_protect_prefix,
                special_ids=self._cfmask_special_ids,
                ignore_index=IGNORE_INDEX,
                node_weight_batch=node_weight_batch,
                weight_aware=self.cfmask_weight_aware,
                p_max=self.cfmask_p_max,
                p_gamma=self.cfmask_p_gamma,
            )
            masked_attn = stats.masked_attention_mask
            n_masked, n_candidates, frac_masked = (
                stats.n_masked, stats.n_candidates, stats.frac_masked)
        beta = self.cfmask_invariance_beta

        if beta <= 0.0:
            outputs = model(
                input_ids=input_ids,
                attention_mask=masked_attn,
                labels=labels,
            )
            ce_loss = outputs.loss
            if ce_loss.dim() > 0:
                ce_loss = ce_loss.mean()
            inv_loss = torch.zeros((), device=ce_loss.device, dtype=ce_loss.dtype)
            total_loss = ce_loss
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            ce_loss = outputs.loss
            if ce_loss.dim() > 0:
                ce_loss = ce_loss.mean()
            masked_out = model(input_ids=input_ids, attention_mask=masked_attn)
            inv_loss = shortcut_invariance_kl(
                outputs.logits,
                masked_out.logits,
                labels,
                ignore_index=IGNORE_INDEX,
                only_clean_correct=True,
            )
            total_loss = ce_loss + beta * inv_loss

        # Combined cfmask + contrastive saliency objective: also pull the model's
        # attention saliency toward the annotation graph on a CLEAN forward (the
        # masked forward above would trivialize saliency). total = cfmask_CE +
        # saliency_lambda * contrastive_saliency. Keeps the cfmask pass-rate signal
        # while adding the saliency-alignment pressure.
        sal_loss = torch.zeros((), device=ce_loss.device, dtype=ce_loss.dtype)
        if self.loss_mode == "ce_shortcut_mask_saliency":
            sal_loss, _ = self._saliency_term(model, input_ids, attention_mask, annot_pairs_batch)
            total_loss = total_loss + self.saliency_lambda * sal_loss

        if not model.training and self.is_world_process_zero():
            self._eval_ntp_buf.append(float(ce_loss.detach().cpu()))
            self._eval_sal_buf.append(float(sal_loss.detach().cpu()) if self.loss_mode == "ce_shortcut_mask_saliency" else float(inv_loss.detach().cpu()))
            self._eval_total_buf.append(float(total_loss.detach().cpu()))

        if self.state.global_step % self.args.logging_steps == 0 and self.is_world_process_zero():
            self.log({
                "ntp_loss": float(ce_loss.detach().cpu()),
                "cfmask_invariance_loss": float(inv_loss.detach().cpu()),
                "cfmask_saliency_loss": float(sal_loss.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
                "cfmask_n_masked": float(n_masked),
                "cfmask_n_candidates": float(n_candidates),
                "cfmask_frac_masked": float(frac_masked),
            })

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_edge_pred_loss(self, model, inputs, annot_pairs_batch, return_outputs):
        """loss_mode=ce_edge_pred: CE + edge_lambda * BCE edge-prediction. One SDPA
        forward (only hidden states needed). Additive + in-distribution -> stable;
        the bilinear head is dropped at inference (GALLa-style)."""
        input_ids = inputs["input_ids"]
        outputs = model(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states=True,
        )
        ce_loss = outputs.loss
        if ce_loss.dim() > 0:
            ce_loss = ce_loss.mean()
        hidden = outputs.hidden_states[self.edge_layer]
        exclude = self._build_exclude_source_mask(input_ids)
        edge_loss = edge_prediction_loss(
            hidden, annot_pairs_batch, self.edge_src_proj, self.edge_dst_proj,
            neg_weight=self.edge_neg_weight, neg_sample_k=self.edge_neg_sample_k,
            temperature=self.edge_temperature, exclude_source_mask=exclude,
        )
        total_loss = ce_loss + self.edge_lambda * edge_loss
        if not model.training and self.is_world_process_zero():
            self._eval_ntp_buf.append(float(ce_loss.detach().cpu()))
            self._eval_sal_buf.append(float(edge_loss.detach().cpu()))
            self._eval_total_buf.append(float(total_loss.detach().cpu()))
        if self.state.global_step % self.args.logging_steps == 0 and self.is_world_process_zero():
            self.log({
                "ntp_loss": float(ce_loss.detach().cpu()),
                "edge_pred_loss": float(edge_loss.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
            })
        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_attn_bias_loss(self, model, inputs, annot_pairs_batch, annot_weights_batch, return_outputs):
        """loss_mode=ce_attn_bias: plain CE under a 4-D additive attention bias that
        adds gate*weight on annotated edges. The scalar gate is the only new param
        and is learned through CE (can go ->0 where unhelpful). One SDPA forward."""
        input_ids = inputs["input_ids"]
        if self.args.bf16:
            comp_dtype = torch.bfloat16
        elif self.args.fp16:
            comp_dtype = torch.float16
        else:
            comp_dtype = model.get_input_embeddings().weight.dtype
        bias4d = build_graph_attention_bias(
            input_ids, inputs["attention_mask"], annot_pairs_batch, self.attn_bias_gate,
            annot_weights_batch=annot_weights_batch, comp_dtype=comp_dtype,
        )
        outputs = model(input_ids=input_ids, attention_mask=bias4d, labels=inputs["labels"])
        ce_loss = outputs.loss
        if ce_loss.dim() > 0:
            ce_loss = ce_loss.mean()
        total_loss = ce_loss
        if not model.training and self.is_world_process_zero():
            self._eval_ntp_buf.append(float(ce_loss.detach().cpu()))
            self._eval_total_buf.append(float(total_loss.detach().cpu()))
        if self.state.global_step % self.args.logging_steps == 0 and self.is_world_process_zero():
            self.log({
                "ntp_loss": float(ce_loss.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
                "attn_bias_gate": float(self.attn_bias_gate.detach().cpu()),
            })
        return (total_loss, outputs) if return_outputs else total_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        annot_pairs_batch: list[torch.Tensor] = inputs.pop("annot_pairs")
        # Optional augmentation tensors (present only when edge_augment is on).
        # Pop them unconditionally so they never leak into model.forward.
        annot_weights_batch = inputs.pop("annot_weights", None)
        node_weight_batch = inputs.pop("node_weight", None)

        if self.loss_mode in ("ce_shortcut_mask", "ce_shortcut_mask_saliency"):
            return self._compute_shortcut_mask_loss(
                model, inputs, annot_pairs_batch, return_outputs,
                node_weight_batch=node_weight_batch,
                annot_weights_batch=annot_weights_batch,
            )

        if self.loss_mode == "ce_edge_pred":
            return self._compute_edge_pred_loss(
                model, inputs, annot_pairs_batch, return_outputs)
        if self.loss_mode == "ce_attn_bias":
            return self._compute_attn_bias_loss(
                model, inputs, annot_pairs_batch, annot_weights_batch, return_outputs)

        needs_saliency = self.loss_mode != "ce_only"
        preempt_saliency = (
            needs_saliency
            and self.saliency_oom_fallback
            and self._should_preempt_saliency(inputs)
        )
        if preempt_saliency:
            self._saliency_preempt_fallback_count += 1
            if self.is_world_process_zero() and (
                self._saliency_preempt_fallback_count <= 5
                or self._saliency_preempt_fallback_count % 50 == 0
            ):
                logger.warning(
                    "[saliency-fallback] preempt CE-only: seq_len=%d >= "
                    "saliency_fallback_max_seq_len=%d (count=%d)",
                    int(inputs["input_ids"].size(1)),
                    self.saliency_fallback_max_seq_len,
                    self._saliency_preempt_fallback_count,
                )
            outputs, ntp_loss = self._ce_only_forward(model, inputs)
            total_loss = ntp_loss
            if not model.training and self.is_world_process_zero():
                self._eval_ntp_buf.append(float(ntp_loss.detach().cpu()))
                self._eval_sal_buf.append(0.0)
                self._eval_total_buf.append(float(total_loss.detach().cpu()))
            if self.state.global_step % self.args.logging_steps == 0 and self.is_world_process_zero():
                self.log({
                    "ntp_loss": float(ntp_loss.detach().cpu()),
                    "saliency_loss": 0.0,
                    "total_loss": float(total_loss.detach().cpu()),
                    "saliency_fallback": 1.0,
                })
            return (total_loss, outputs) if return_outputs else total_loss

        forward_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "output_attentions": needs_saliency,
            "output_hidden_states": needs_saliency,
        }
        # Always pass labels for CE modes; when eos_loss_weight != 1 we recompute
        # a weighted NTP from logits so HF's mean CE is not used as-is.
        if self.loss_mode != "saliency_only":
            forward_kwargs["labels"] = inputs["labels"]

        attn_proj_weights = None
        outputs = None
        ntp_loss = None
        diag = None
        saliency_loss = None
        used_fallback = False

        def _run_saliency_path():
            nonlocal attn_proj_weights, outputs, ntp_loss, diag, saliency_loss
            if needs_saliency:
                all_layers = canonical_saliency_agg(self.saliency_agg) == "rollout"
                with capture_attn_proj_weights(
                    model, layer_index=self.saliency_layer, all_layers=all_layers,
                ) as attn_proj_weights:
                    outputs = model(**forward_kwargs)
            else:
                outputs = model(**forward_kwargs)

            if self.loss_mode == "saliency_only":
                ntp_loss = torch.zeros(
                    (), device=outputs.logits.device, dtype=outputs.logits.dtype
                )
            else:
                ntp_loss = self._ntp_loss_from_outputs(outputs, inputs["labels"])

            saliency_loss = torch.zeros(
                (), device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            if self.loss_mode != "ce_only":
                floor_step = self.saliency_floor_step + 1
                exclude_source_mask = self._build_exclude_source_mask(inputs["input_ids"])
                diag = saliency_loss_from_outputs(
                    model,
                    outputs,
                    annot_pairs_batch,
                    saliency_layer=self.saliency_layer,
                    saliency_agg=self.saliency_agg,
                    exclude_source_mask=exclude_source_mask,
                    alpha=self.saliency_alpha,
                    eps=self.saliency_eps,
                    floor_eps=self.saliency_floor_eps,
                    floor_eps_mode=self.saliency_floor_eps_mode,
                    floor_eps_quantile=self.saliency_floor_quantile,
                    floor_eps_ema_beta=self.saliency_floor_ema_beta,
                    prev_floor_eps=self.saliency_floor_eps_ema,
                    floor_eps_min=self.saliency_floor_min_eps,
                    floor_eps_step=floor_step,
                    floor_eps_warmup_steps=self.saliency_floor_warmup_steps,
                    floor_logit_eps=self.saliency_floor_logit_eps,
                    loss_type=self.saliency_loss_type,
                    margin_plus=self.saliency_margin_plus,
                    margin_minus=self.saliency_margin_minus,
                    margin_gamma=self.saliency_margin_gamma,
                    neg_weight=self.saliency_neg_weight,
                    neg_hard_only=self.saliency_neg_hard_only,
                    neg_sample_k=self.saliency_neg_sample_k,
                    attn_proj_weights=attn_proj_weights,
                )
                saliency_loss = diag.loss
                self.saliency_floor_step = floor_step
                if (
                    self.saliency_loss_type != "softmax"
                    and self.saliency_floor_logit_eps is None
                    and self.saliency_floor_eps_mode == "ema_quantile"
                    and diag.n_queries > 0
                ):
                    self.saliency_floor_eps_ema = diag.floor_eps

                if diag.n_samples > 0:
                    loss_label = saliency_loss_display_name(diag.loss_type)
                    logger.info(
                        f"[saliency:{loss_label}] C̄={diag.avg_C:.4f}  N̄={diag.avg_N:.4f}  "
                        f"ratio={diag.avg_ratio:.3f}  tau={self.saliency_alpha}  "
                        f"eps_num={self.saliency_eps:.4g}  floor_mode={diag.floor_eps_mode}  "
                        f"floor_step={diag.floor_eps_step}  warmup={diag.floor_eps_warmup_steps}  "
                        f"eps_floor_effective={diag.floor_eps:.4g}  eps_floor_batch={diag.batch_floor_eps:.4g}  "
                        f"eps_floor_logit={diag.floor_logit_eps:.4g}  floor_kind={diag.floor_eps_kind}  "
                        f"loss={diag.loss.item():.4f}  #queries={diag.n_queries}"
                    )

        try:
            _run_saliency_path()
        except RuntimeError as exc:
            if not (
                needs_saliency
                and self.saliency_oom_fallback
                and self.loss_mode in ("ce_saliency", "saliency_only")
                and self._is_oom_error(exc)
            ):
                raise
            used_fallback = True
            self._saliency_oom_fallback_count += 1
            # Keep completed-forward logits when possible (typical: OOM inside
            # build_contribution_rows). Only drop heavy saliency tensors.
            attn_proj_weights = None
            diag = None
            saliency_loss = None
            has_logits = (
                outputs is not None
                and getattr(outputs, "logits", None) is not None
            )
            if has_logits:
                self._strip_heavy_outputs(outputs)
            else:
                outputs = None
            self._empty_accel_cache()
            if self.is_world_process_zero() and (
                self._saliency_oom_fallback_count <= 5
                or self._saliency_oom_fallback_count % 50 == 0
            ):
                logger.warning(
                    "[saliency-fallback] OOM on saliency path → CE-only for this step "
                    "(seq_len=%d reuse_logits=%s count=%d): %s",
                    int(inputs["input_ids"].size(1)),
                    has_logits,
                    self._saliency_oom_fallback_count,
                    str(exc).split("\n", 1)[0][:200],
                )
            if self.loss_mode == "saliency_only":
                # No CE term in this mode; skip the step with a zero loss.
                device = inputs["input_ids"].device
                total_loss = torch.zeros((), device=device)
                if return_outputs:
                    if not has_logits:
                        try:
                            outputs, _ = self._ce_only_forward(model, inputs)
                        except RuntimeError as ce_exc:
                            if not self._is_oom_error(ce_exc):
                                raise
                            self._empty_accel_cache()
                            outputs = None
                    return total_loss, outputs
                return total_loss
            # Prefer reusing logits from a completed forward. A second forward
            # would often OOM again and can desync FSDP vs ranks that did not OOM.
            if has_logits:
                ntp_loss = self._ntp_loss_from_outputs(outputs, inputs["labels"])
            else:
                try:
                    outputs, ntp_loss = self._ce_only_forward(model, inputs)
                except RuntimeError as ce_exc:
                    if not self._is_oom_error(ce_exc):
                        raise
                    # Last resort: skip microbatch rather than kill the run.
                    self._empty_accel_cache()
                    device = inputs["input_ids"].device
                    ntp_loss = torch.zeros((), device=device)
                    outputs = None
                    if self.is_world_process_zero():
                        logger.warning(
                            "[saliency-fallback] CE-only forward also OOM "
                            "(seq_len=%d) → skip step with zero loss",
                            int(inputs["input_ids"].size(1)),
                        )
            saliency_loss = torch.zeros((), device=ntp_loss.device, dtype=ntp_loss.dtype)

        if self.loss_mode == "ce_saliency":
            total_loss = ntp_loss + self.saliency_lambda * saliency_loss
        elif self.loss_mode == "saliency_only":
            total_loss = saliency_loss
        elif self.loss_mode == "ce_only":
            total_loss = ntp_loss
        else:
            raise ValueError(f"Unsupported loss_mode={self.loss_mode!r}")

        detail = None
        should_detail_log = (
            (not used_fallback)
            and self.loss_mode != "ce_only"
            and self.saliency_detail_log_path
            and self.saliency_detail_log_steps > 0
            and self.state.global_step % self.saliency_detail_log_steps == 0
            and self.is_world_process_zero()
        )
        if should_detail_log:
            from saliency_diagnostics import saliency_details_from_outputs

            detail = saliency_details_from_outputs(
                model,
                outputs,
                annot_pairs_batch,
                alpha=self.saliency_alpha,
                eps=self.saliency_eps,
                floor_eps=diag.floor_eps if diag is not None else self.saliency_floor_eps,
                floor_logit_eps=diag.floor_logit_eps if diag is not None and diag.floor_eps_kind == "logit" else None,
                top_k=self.saliency_detail_top_k,
            )
            with open(self.saliency_detail_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": int(self.state.global_step),
                    "loss_mode": self.loss_mode,
                    "saliency_loss_type": self.saliency_loss_type,
                    "ntp_loss": float(ntp_loss.detach().cpu()),
                    "saliency_loss": float(saliency_loss.detach().cpu()),
                    "total_loss": float(total_loss.detach().cpu()),
                    "teacher_diag": {
                        "loss_type": diag.loss_type if diag is not None else self.saliency_loss_type,
                        "Cbar": diag.avg_C if diag is not None else 0.0,
                        "Nbar": diag.avg_N if diag is not None else 0.0,
                        "ratio": diag.avg_ratio if diag is not None else 0.0,
                        "eps_num": self.saliency_eps,
                        "floor_eps": diag.floor_eps if diag is not None else self.saliency_floor_eps,
                        "batch_floor_eps": diag.batch_floor_eps if diag is not None else 0.0,
                        "floor_logit_eps": diag.floor_logit_eps if diag is not None else 0.0,
                        "floor_eps_kind": diag.floor_eps_kind if diag is not None else "saliency",
                        "floor_eps_mode": diag.floor_eps_mode if diag is not None else self.saliency_floor_eps_mode,
                        "floor_eps_step": diag.floor_eps_step if diag is not None else self.saliency_floor_step,
                        "floor_eps_warmup_steps": diag.floor_eps_warmup_steps if diag is not None else self.saliency_floor_warmup_steps,
                        "n_queries": diag.n_queries if diag is not None else 0,
                        "n_samples": diag.n_samples if diag is not None else 0,
                    },
                    "strict_causal_detail": {
                        "Cbar": detail.avg_C,
                        "Nbar": detail.avg_N,
                        "ratio": detail.avg_ratio,
                        "loss": detail.loss,
                        "active_margin_rate": detail.active_margin_rate,
                        "hit_at_k": detail.hit_at_k,
                        "recall_at_k": detail.recall_at_k,
                        "precision_at_k": detail.precision_at_k,
                        "map_at_k": detail.map_at_k,
                        "top_k": self.saliency_detail_top_k,
                        "num_annotation_edges": detail.num_annotation_edges,
                        "n_queries": detail.n_queries,
                        "n_samples": detail.n_samples,
                    },
                    "query_stats": detail.query_stats,
                    "edge_stats": detail.edge_stats,
                }, ensure_ascii=False) + "\n")

        # Capture per-batch breakdown during evaluation. model.training is False
        # inside Trainer.evaluate / predict.
        if not model.training and self.is_world_process_zero():
            self._eval_ntp_buf.append(float(ntp_loss.detach().cpu()))
            self._eval_sal_buf.append(float(saliency_loss.detach().cpu()))
            self._eval_total_buf.append(float(total_loss.detach().cpu()))
            if diag is not None and diag.n_queries > 0:
                self._eval_ratio_buf.append(float(diag.avg_ratio))
            if (
                (not used_fallback)
                and detail is None
                and self.loss_mode != "ce_only"
                and self.saliency_detail_top_k > 0
            ):
                # Compute saliency mAP@k on this eval batch even when we are not
                # writing the detailed JSONL — keeps validation cheap by reusing
                # the same outputs we just forwarded.
                try:
                    from saliency_diagnostics import saliency_details_from_outputs
                    detail_eval = saliency_details_from_outputs(
                        model, outputs, annot_pairs_batch,
                        alpha=self.saliency_alpha, eps=self.saliency_eps,
                        floor_eps=diag.floor_eps if diag is not None else self.saliency_floor_eps,
                        floor_logit_eps=diag.floor_logit_eps if diag is not None and diag.floor_eps_kind == "logit" else None,
                        top_k=self.saliency_detail_top_k,
                    )
                    if detail_eval.n_queries > 0:
                        self._eval_mAP_buf.append(float(detail_eval.map_at_k))
                        self._eval_recall_at_k_buf.append(float(detail_eval.recall_at_k))
                        self._eval_precision_at_k_buf.append(float(detail_eval.precision_at_k))
                except Exception:
                    pass
            elif detail is not None and detail.n_queries > 0:
                self._eval_mAP_buf.append(float(detail.map_at_k))
                self._eval_recall_at_k_buf.append(float(detail.recall_at_k))
                self._eval_precision_at_k_buf.append(float(detail.precision_at_k))

        if self.state.global_step % self.args.logging_steps == 0:
            log_payload = {
                "ntp_loss": ntp_loss.item(),
                "saliency_loss": saliency_loss.item(),
                "saliency_loss_type_id": 1.0 if self.saliency_loss_type == "softmax" else 0.0,
                "total_loss": total_loss.item(),
                "saliency_fallback": 1.0 if used_fallback else 0.0,
                "saliency_oom_fallback_count": float(self._saliency_oom_fallback_count),
            }
            if self.loss_mode != "ce_only":
                log_payload.update({
                    "saliency_eps_num": self.saliency_eps,
                    "saliency_eps_floor_effective": diag.floor_eps if diag is not None else self.saliency_floor_eps,
                    "saliency_eps_floor_batch": diag.batch_floor_eps if diag is not None else 0.0,
                    "saliency_eps_floor_logit": diag.floor_logit_eps if diag is not None else 0.0,
                })
            if detail is not None:
                log_payload.update({
                    "strict_Cbar": detail.avg_C,
                    "strict_Nbar": detail.avg_N,
                    "strict_ratio": detail.avg_ratio,
                    "strict_active_margin_rate": detail.active_margin_rate,
                    "strict_hit_at_k": detail.hit_at_k,
                    "strict_recall_at_k": detail.recall_at_k,
                    "strict_precision_at_k": detail.precision_at_k,
                    "strict_map_at_k": detail.map_at_k,
                })
            self.log(log_payload)

        return (total_loss, outputs) if return_outputs else total_loss




@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-Coder-7B-Instruct")
    use_flash_attention: bool = field(default=False)
    use_peft: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated LoRA target module names. For MoE models pass "
                          "attention-only 'q_proj,k_proj,v_proj,o_proj' so LoRA is NOT "
                          "attached to every expert (thousands of adapters)."},
    )
    device_map: str = field(
        default="",
        metadata={"help": "HF device_map for loading (e.g. 'auto' = shard one large model "
                          "across all visible GPUs, naive model parallelism, single process). "
                          "Empty = single device. Do NOT combine with FSDP/torchrun."},
    )


@dataclass
class DataArguments:
    data_path: str = field(default="./data/mceval/mceval-sft.jsonl")
    eval_data_path: str = field(
        default="",
        metadata={"help": "Optional validation dataset (same compact JSONL format as data_path). If set, used for periodic eval."},
    )
    eval_max_samples: int = field(
        default=0,
        metadata={"help": "Subsample eval dataset to first-N samples for speed; 0 = use all."},
    )
    max_len: int = field(default=8192)
    language: Optional[Literal["Python", "C#", "CPP", "Go", "Java", "C"]] = field(
        default=None,
        metadata={"help": "Filter to one language: Python | C# | CPP | Go | Java | C. None = all."}
    )
    edge_augment: bool = field(
        default=False,
        metadata={"help": "Densify edge labels by transitive closure (a->b, b->c => a->c). Default OFF = legacy single-hop labels."},
    )
    edge_augment_decay: float = field(
        default=0.5,
        metadata={"help": "[edge_augment] Per-hop weight decay. A min-hop-h edge gets weight decay**(h-1). 1.0 = full inheritance (all inherited edges weight 1.0)."},
    )
    edge_augment_max_hops: int = field(
        default=0,
        metadata={"help": "[edge_augment] Cap path length (edges). 0 = unlimited."},
    )
    edge_augment_node_weight: bool = field(
        default=False,
        metadata={"help": "[edge_augment] Also emit a per-token weight-to-target vector for weight-aware cfmask. Default OFF."},
    )
    edge_augment_mode: str = field(
        default="directed",
        metadata={"help": "[edge_augment] 'directed' (default) follows edge direction (a->b,b->c => a->c). 'undirected' links every token pair in the same connected component, hops = undirected graph distance."},
    )
    token_select: bool = field(
        default=False,
        metadata={"help": "[token_select] Teacher-gated informative-token selection: exclude completion tokens whose precomputed teacher NLL (comp_teacher_nll field) exceeds the threshold (missing-info / memorization) from the loss. Default OFF = legacy."},
    )
    token_select_threshold: float = field(
        default=2.0,
        metadata={"help": "[token_select] Teacher-NLL threshold in nats; completion tokens above this are excluded as uninferable from the input."},
    )
    token_select_keep_special: bool = field(
        default=True,
        metadata={"help": "[token_select] Never exclude special tokens (EOS/im_end) even if above threshold."},
    )
    annot_skip: bool = field(
        default=False,
        metadata={"help": "[annot_skip] Drop completion tokens that are not an annotation-edge destination from the CE loss. Protected tokens (first keep_first completion tokens + keyword/punct/whitespace + optional specials) are never dropped. Default OFF."},
    )
    annot_skip_keep_first: int = field(
        default=2,
        metadata={"help": "[annot_skip] Always keep the first K completion tokens in the loss (default 2 = positions 0 and 1)."},
    )
    annot_skip_keep_special: bool = field(
        default=True,
        metadata={"help": "[annot_skip] Never drop special tokens (EOS/im_end) even if unannotated."},
    )
    supervise_eos: bool = field(
        default=True,
        metadata={"help": "If True, unmask a trailing <|im_end|>/EOS that sits right after the last supervised completion token but is labeled -100. Fixes compact data that never trains the model to stop. Set False for legacy label behavior."},
    )
    raw_prompt_response: bool = field(
        default=False,
        metadata={"help": "TEMPORARY DEBUG: load raw {prompt,response,task_id} JSONL (no compact / no annotations), tokenize on the fly for ce_only smoke runs. Default OFF."},
    )
    raw_max_samples: int = field(
        default=0,
        metadata={"help": "TEMPORARY DEBUG: if >0 with raw_prompt_response, only keep the first N valid raw rows (0 = all)."},
    )


@dataclass
class SFTTrainingArguments(TrainingArguments):
    eos_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Multiply CE loss on stop tokens (<|im_end|>/EOS). 1.0 = uniform CE (default). Try 2~5 if the model over-generates past mid."},
    )
    saliency_lambda: float = field(
        default=0.1,
        metadata={"help": "Weight γ in L_total = L_next + γ·L_contrib."}
    )
    saliency_alpha: float = field(
        default=1.5,
        metadata={"help": "Temperature tau for saliency logits. softmax_margin uses log(C + eps) / tau; softmax uses C / tau."}
    )
    saliency_eps: float = field(
        default=1e-8,
        metadata={"help": "Numerical epsilon for log(C + eps), safe division, and diagnostics."}
    )
    saliency_floor_eps: float = field(
        default=0.0,
        metadata={"help": "Fixed saliency-space floor for negative logits when saliency_floor_eps_mode=fixed."}
    )
    saliency_floor_eps_mode: str = field(
        default="ema_quantile",
        metadata={"help": "Floor eps schedule: fixed | batch_quantile | ema_quantile."}
    )
    saliency_floor_quantile: float = field(
        default=0.75,
        metadata={"help": "Quantile over current causal-source saliency values for dynamic floor eps."}
    )
    saliency_floor_ema_beta: float = field(
        default=0.95,
        metadata={"help": "EMA beta for saliency_floor_eps_mode=ema_quantile."}
    )
    saliency_floor_min_eps: float = field(
        default=1e-8,
        metadata={"help": "Minimum positive eps used by dynamic floor schedules."}
    )
    saliency_floor_warmup_steps: int = field(
        default=10,
        metadata={"help": "Use floor_eps=0 for the first N dynamic-floor saliency steps; N=10 by default."}
    )
    saliency_floor_logit_eps: Optional[float] = field(
        default=None,
        metadata={"help": "If set, softmax_margin uses this fixed logit-space negative floor: max(l_neg, saliency_floor_logit_eps). Example: -10."}
    )
    saliency_loss_type: str = field(
        default="softmax_margin",
        metadata={"help": "Saliency objective: softmax_margin log(C+eps)/tau with negative floor, or softmax raw C/tau over all causal sources. Legacy alias infonce_floor is accepted."}
    )
    saliency_margin_plus: float = field(
        default=2.08,
        metadata={"help": "[margin_bce] Lower-bound margin on log(C+eps) for annotated positives (default log(8))."}
    )
    saliency_margin_minus: float = field(
        default=0.41,
        metadata={"help": "[margin_bce] Upper-bound margin on log(C+eps) for non-annotated negatives; no penalty once r <= m_minus (default log(1.5))."}
    )
    saliency_margin_gamma: float = field(
        default=2.0,
        metadata={"help": "[margin_bce] Sharpness of the per-edge softplus (sigmoid gradient slope)."}
    )
    saliency_neg_weight: float = field(
        default=0.5,
        metadata={"help": "[margin_bce] Multiplier for the negative-side mean penalty."}
    )
    saliency_neg_hard_only: bool = field(
        default=False,
        metadata={"help": "[margin_bce] If True, the negative-side mean only averages over negatives with r > m_minus."}
    )
    saliency_neg_sample_k: int = field(
        default=0,
        metadata={"help": "[softmax_margin/softmax] If >0, sample this many negatives per query into the softmax denominator (MoCo-style). 0 = use all causal negatives (default)."}
    )
    saliency_layer: int = field(
        default=-1,
        metadata={"help": "Decoder layer whose attention/value path defines the saliency when saliency_agg=last. -1 = last layer (default). Ignored when saliency_agg=rollout."}
    )
    saliency_agg: str = field(
        default="last",
        metadata={"help": "Saliency aggregation: 'last' = ||T||_2 at saliency_layer (default training surrogate); 'rollout' = paper §B ALTI (L1 min_sum C per layer, full-layer product C_roll=C^L...C^1)."}
    )
    saliency_exclude_sink_prefix: int = field(
        default=0,
        metadata={"help": "Drop the first-N source positions (attention-sink prefix, e.g. <|im_start|> system \\n) from the NEGATIVE set so the loss never fights the sink. 0 = keep (legacy)."}
    )
    saliency_exclude_special_tokens: bool = field(
        default=False,
        metadata={"help": "Drop special-token sources (all_special_ids: <|im_start|>, <|im_end|>, ...) from the saliency NEGATIVE set."}
    )
    cfmask_rate: float = field(
        default=0.3,
        metadata={"help": "[ce_shortcut_mask] Fraction of maskable (non-annotated) context tokens to hide per sample."}
    )
    cfmask_max_k: int = field(
        default=0,
        metadata={"help": "[ce_shortcut_mask] Hard cap on #tokens masked per sample; 0 = no cap."}
    )
    cfmask_min_k: int = field(
        default=0,
        metadata={"help": "[ce_shortcut_mask] Minimum #tokens masked per sample when candidates exist; 0 = no floor."}
    )
    cfmask_recency_window: int = field(
        default=8,
        metadata={"help": "[ce_shortcut_mask] Protect the last-N context tokens before the target (genuine local syntax, never masked)."}
    )
    cfmask_protect_prefix: int = field(
        default=0,
        metadata={"help": "[ce_shortcut_mask] Protect the first-N positions (chat/system header + attention sink) from masking."}
    )
    cfmask_exclude_special: bool = field(
        default=True,
        metadata={"help": "[ce_shortcut_mask] Never mask special/role tokens (all_special_ids)."}
    )
    cfmask_invariance_beta: float = field(
        default=0.0,
        metadata={"help": "[ce_shortcut_mask] 0 = Variant A (plain CE on masked input). >0 = Variant B: CE(clean) + beta*KL(clean||masked) on clean-correct target positions."}
    )
    cfmask_weight_aware: bool = field(
        default=False,
        metadata={"help": "[ce_shortcut_mask] Weight-aware probabilistic masking: mask each context token with prob p_max*(1-w)^gamma where w=node_weight (relevance to target). Requires edge_augment_node_weight=True. Default OFF = uniform fixed-rate masking."}
    )
    cfmask_p_max: float = field(
        default=0.9,
        metadata={"help": "[ce_shortcut_mask weight-aware] Max mask probability, applied to fully-unrelated tokens (w=0). First-order tokens (w=1) get prob 0."}
    )
    cfmask_p_gamma: float = field(
        default=1.0,
        metadata={"help": "[ce_shortcut_mask weight-aware] Shape of the (1-w) ramp: p_mask = p_max*(1-w)^gamma. gamma>1 spares mid-weight tokens more; gamma<1 masks them harder."}
    )
    cfmask_per_target: bool = field(
        default=False,
        metadata={"help": "[ce_shortcut_mask] Per-target masking: build a 4-D [T,T] attention mask where each target query row hides only ITS OWN non-annotation keys (vs one global key mask for all queries). Makes uniform masking augmentation-sensitive too. Default OFF."}
    )
    edge_lambda: float = field(
        default=0.5,
        metadata={"help": "[ce_edge_pred] Weight of the auxiliary edge-prediction BCE: L = CE + edge_lambda * L_edge."}
    )
    edge_proj_dim: int = field(
        default=256,
        metadata={"help": "[ce_edge_pred] Low-rank dim of the biaffine edge scorer (src/dst projections)."}
    )
    edge_neg_weight: float = field(
        default=1.0,
        metadata={"help": "[ce_edge_pred] Multiplier on the negative (non-edge causal source) BCE term."}
    )
    edge_neg_sample_k: int = field(
        default=0,
        metadata={"help": "[ce_edge_pred] If >0, subsample this many negative sources per target (MoCo-style); 0 = all causal negatives."}
    )
    edge_temperature: float = field(
        default=1.0,
        metadata={"help": "[ce_edge_pred] Temperature tau on the edge logit: <q,s>/(sqrt(d)*tau)."}
    )
    edge_layer: int = field(
        default=-1,
        metadata={"help": "[ce_edge_pred] hidden_states index feeding the edge scorer (-1 = last layer)."}
    )
    attn_bias_init: float = field(
        default=1.0,
        metadata={"help": "[ce_attn_bias] Initial value of the learnable scalar gate g; the additive attention bias on annotated edges is g*weight."}
    )
    loss_mode: str = field(
        default="ce_saliency",
        metadata={"help": "Training objective: ce_saliency | saliency_only | ce_only | ce_shortcut_mask | ce_shortcut_mask_saliency | ce_edge_pred | ce_attn_bias."}
    )
    saliency_oom_fallback: bool = field(
        default=True,
        metadata={"help": "If True, catch OOM on the saliency/eager-attn path and fall back to CE-only for that step instead of crashing. Recommended for long full-corpus runs."},
    )
    saliency_fallback_max_seq_len: int = field(
        default=1280,
        metadata={
            "help": (
                "If >0, skip saliency (CE-only) whenever padded seq_len >= this "
                "threshold, without waiting for OOM. Default 1280 avoids eager "
                "T^2 peaks on long samples; set 0 to disable preemptive skip."
            )
        },
    )
    saliency_detail_log_path: str = field(
        default="",
        metadata={"help": "Optional JSONL path for strict-causal query/edge saliency diagnostics."}
    )
    saliency_detail_log_steps: int = field(
        default=0,
        metadata={"help": "Write detailed saliency diagnostics every N global steps; 0 disables."}
    )
    saliency_detail_top_k: int = field(
        default=10,
        metadata={"help": "Top-k threshold for annotation hit@k in detailed diagnostics."}
    )
    enable_attn_viz: bool = field(
        default=False,
        metadata={"help": "Capture step-0 attention visualization samples during training. Disabled by default to save memory."}
    )
    attn_viz_num_samples: int = field(
        default=10,
        metadata={"help": "Number of samples for step-0 attention visualization when enable_attn_viz=True."}
    )
    eval_codebleu_samples: int = field(
        default=0,
        metadata={"help": "If >0, compute greedy-generation CodeBLEU on the first-N valid samples at each eval and log eval_codebleu. 0 = off."}
    )
    cache_dir: Optional[str] = field(default=None)




def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, SFTTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.saliency_loss_type = canonical_saliency_loss_type(training_args.saliency_loss_type)
    # Keep custom fields such as annot_pairs. HF Trainer otherwise prunes keys
    # that are not in model.forward(), which silently disables saliency loss.
    training_args.remove_unused_columns = False

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        cache_dir=training_args.cache_dir,
        model_max_length=data_args.max_len,
        truncation=True,
        padding_side="right",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|im_end|>", "<|im_start|>"]
    })

    # ── Data ──────────────────────────────────────────────────────────────────
    train_dataset = AnnotatedSFTDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_len=data_args.max_len,
        language=data_args.language,
        edge_augment=data_args.edge_augment,
        edge_augment_decay=data_args.edge_augment_decay,
        edge_augment_max_hops=data_args.edge_augment_max_hops,
        edge_augment_node_weight=data_args.edge_augment_node_weight,
        edge_augment_mode=data_args.edge_augment_mode,
        token_select=data_args.token_select,
        token_select_threshold=data_args.token_select_threshold,
        token_select_keep_special=data_args.token_select_keep_special,
        annot_skip=data_args.annot_skip,
        annot_skip_keep_first=data_args.annot_skip_keep_first,
        annot_skip_keep_special=data_args.annot_skip_keep_special,
        supervise_eos=data_args.supervise_eos,
        raw_prompt_response=data_args.raw_prompt_response,
        raw_max_samples=data_args.raw_max_samples,
    )
    if data_args.raw_prompt_response and training_args.loss_mode not in ("ce_only",):
        logger.warning(
            "raw_prompt_response=True but loss_mode=%s: raw rows have no "
            "attention_edges; saliency/cfmask will be a no-op or weak. Prefer ce_only.",
            training_args.loss_mode,
        )
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = AnnotatedSFTDataset(
            data_path=data_args.eval_data_path,
            tokenizer=tokenizer,
            max_len=data_args.max_len,
            language=data_args.language,
            edge_augment=data_args.edge_augment,
            edge_augment_decay=data_args.edge_augment_decay,
            edge_augment_max_hops=data_args.edge_augment_max_hops,
            edge_augment_node_weight=data_args.edge_augment_node_weight,
            edge_augment_mode=data_args.edge_augment_mode,
            annot_skip=data_args.annot_skip,
            annot_skip_keep_first=data_args.annot_skip_keep_first,
            annot_skip_keep_special=data_args.annot_skip_keep_special,
            supervise_eos=data_args.supervise_eos,
            raw_prompt_response=data_args.raw_prompt_response,
            raw_max_samples=data_args.raw_max_samples,
        )
        if data_args.eval_max_samples > 0 and len(eval_dataset) > data_args.eval_max_samples:
            # Deterministic first-N subsample. Direct slice of internal list keeps
            # the Dataset API working (no torch.utils.data.Subset needed).
            eval_dataset.items = eval_dataset.items[: data_args.eval_max_samples]
            logger.info(f"Subsampled eval dataset to first {len(eval_dataset)} samples")
    data_collator = DataCollatorForAnnotatedSFT(tokenizer=tokenizer)

    # ── Model ─────────────────────────────────────────────────────────────────
    # Only the saliency objectives read attention probabilities (A^h_{i,j}),
    # which requires eager attention. ce_only and ce_shortcut_mask (cfmask) do
    # NOT — they only need logits — so they can use the memory-efficient SDPA
    # path, which is essential for large (14B/32B) models where eager would
    # materialize the full [heads, seq, seq] attention tensor and OOM.
    needs_eager_attn = training_args.loss_mode in ("ce_saliency", "saliency_only", "ce_shortcut_mask_saliency")
    attn_impl = "eager" if needs_eager_attn else "sdpa"
    if model_args.use_flash_attention and needs_eager_attn:
        logger.warning(
            "use_flash_attention=True is incompatible with the saliency loss "
            "(needs eager attention to expose A^h_{i,j}). Falling back to eager."
        )
    logger.info(f"Loading model with attn_implementation={attn_impl!r} (loss_mode={training_args.loss_mode!r}).")
    _device_map = model_args.device_map or None
    _low_cpu = bool(_device_map) or bool(getattr(training_args, "fsdp", None))
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=_device_map,
        low_cpu_mem_usage=_low_cpu,
    )
    model.config.use_cache = False

    if model_args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=[m.strip() for m in model_args.lora_target_modules.split(",") if m.strip()],
            bias="none",
        )
        logger.info(
            "LoRA config: r=%d alpha=%d dropout=%s",
            model_args.lora_r, model_args.lora_alpha, model_args.lora_dropout,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # ── Trainer ───────────────────────────────────────────────────────────────
    callbacks = []
    if training_args.enable_attn_viz:
        callbacks.append(AttentionVisualizationCallback(
            dataset=train_dataset,
            tokenizer=tokenizer,
            output_dir=os.path.join(training_args.output_dir, "attn_viz"),
            num_samples=training_args.attn_viz_num_samples,
        ))
    eval_breakdown_cb = EvalBreakdownCallback() if eval_dataset is not None else None
    if eval_breakdown_cb is not None:
        callbacks.append(eval_breakdown_cb)
    if eval_dataset is not None and training_args.eval_codebleu_samples > 0:
        callbacks.append(ValidCodeBLEUCallback(
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            n_samples=training_args.eval_codebleu_samples,
            lang=(data_args.language or "go").lower(),
        ))

    trainer = AnnotatedSFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        eval_dataset=eval_dataset if eval_dataset is not None else train_dataset,
        train_dataset=train_dataset,
        data_collator=data_collator,
        saliency_lambda=training_args.saliency_lambda,
        saliency_alpha=training_args.saliency_alpha,
        saliency_eps=training_args.saliency_eps,
        saliency_floor_eps=training_args.saliency_floor_eps,
        saliency_floor_eps_mode=training_args.saliency_floor_eps_mode,
        saliency_floor_quantile=training_args.saliency_floor_quantile,
        saliency_floor_ema_beta=training_args.saliency_floor_ema_beta,
        saliency_floor_min_eps=training_args.saliency_floor_min_eps,
        saliency_floor_warmup_steps=training_args.saliency_floor_warmup_steps,
        saliency_floor_logit_eps=training_args.saliency_floor_logit_eps,
        saliency_loss_type=training_args.saliency_loss_type,
        loss_mode=training_args.loss_mode,
        saliency_detail_log_path=training_args.saliency_detail_log_path,
        saliency_detail_log_steps=training_args.saliency_detail_log_steps,
        saliency_detail_top_k=training_args.saliency_detail_top_k,
        saliency_margin_plus=training_args.saliency_margin_plus,
        saliency_margin_minus=training_args.saliency_margin_minus,
        saliency_margin_gamma=training_args.saliency_margin_gamma,
        saliency_neg_weight=training_args.saliency_neg_weight,
        saliency_neg_hard_only=training_args.saliency_neg_hard_only,
        saliency_neg_sample_k=training_args.saliency_neg_sample_k,
        saliency_layer=training_args.saliency_layer,
        saliency_agg=training_args.saliency_agg,
        saliency_exclude_sink_prefix=training_args.saliency_exclude_sink_prefix,
        saliency_exclude_special_tokens=training_args.saliency_exclude_special_tokens,
        saliency_oom_fallback=training_args.saliency_oom_fallback,
        saliency_fallback_max_seq_len=training_args.saliency_fallback_max_seq_len,
        cfmask_rate=training_args.cfmask_rate,
        cfmask_max_k=training_args.cfmask_max_k,
        cfmask_min_k=training_args.cfmask_min_k,
        cfmask_recency_window=training_args.cfmask_recency_window,
        cfmask_protect_prefix=training_args.cfmask_protect_prefix,
        cfmask_exclude_special=training_args.cfmask_exclude_special,
        cfmask_invariance_beta=training_args.cfmask_invariance_beta,
        cfmask_weight_aware=training_args.cfmask_weight_aware,
        cfmask_p_max=training_args.cfmask_p_max,
        cfmask_p_gamma=training_args.cfmask_p_gamma,
        cfmask_per_target=training_args.cfmask_per_target,
        edge_lambda=training_args.edge_lambda,
        edge_proj_dim=training_args.edge_proj_dim,
        edge_neg_weight=training_args.edge_neg_weight,
        edge_neg_sample_k=training_args.edge_neg_sample_k,
        edge_temperature=training_args.edge_temperature,
        edge_layer=training_args.edge_layer,
        attn_bias_init=training_args.attn_bias_init,
        eos_loss_weight=training_args.eos_loss_weight,
        callbacks=callbacks,
    )
    if eval_breakdown_cb is not None:
        # HF callbacks do not receive `trainer` in their kwargs; stash a class
        # attribute so on_evaluate can read the latest trainer.
        EvalBreakdownCallback._trainer = trainer

    logger.info(
        "Training objective: loss_mode=%s saliency_agg=%s saliency_layer=%s "
        "saliency_loss=%s saliency_loss_type=%s "
        "saliency_lambda=%s tau=%s eps_num=%s floor_eps=%s floor_logit_eps=%s floor_mode=%s "
        "floor_quantile=%s floor_warmup=%s saliency_oom_fallback=%s "
        "saliency_fallback_max_seq_len=%s output_dir=%s",
        training_args.loss_mode,
        training_args.saliency_agg,
        training_args.saliency_layer,
        saliency_loss_display_name(training_args.saliency_loss_type),
        training_args.saliency_loss_type,
        training_args.saliency_lambda,
        training_args.saliency_alpha,
        training_args.saliency_eps,
        training_args.saliency_floor_eps,
        training_args.saliency_floor_logit_eps,
        training_args.saliency_floor_eps_mode,
        training_args.saliency_floor_quantile,
        training_args.saliency_floor_warmup_steps,
        training_args.saliency_oom_fallback,
        training_args.saliency_fallback_max_seq_len,
        training_args.output_dir,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    if trainer.is_world_process_zero():
        os.makedirs(training_args.output_dir, exist_ok=True)
        objective = {
            "loss_mode": training_args.loss_mode,
            "saliency_loss_type": training_args.saliency_loss_type,
            "saliency_loss_name": saliency_loss_display_name(training_args.saliency_loss_type),
            "saliency_lambda": training_args.saliency_lambda,
            "saliency_temperature_tau": training_args.saliency_alpha,
            "saliency_eps_num": training_args.saliency_eps,
            "saliency_floor_eps": training_args.saliency_floor_eps,
            "saliency_floor_logit_eps": training_args.saliency_floor_logit_eps,
            "saliency_floor_eps_mode": training_args.saliency_floor_eps_mode,
            "saliency_floor_quantile": training_args.saliency_floor_quantile,
            "saliency_floor_ema_beta": training_args.saliency_floor_ema_beta,
            "saliency_floor_min_eps": training_args.saliency_floor_min_eps,
            "saliency_floor_warmup_steps": training_args.saliency_floor_warmup_steps,
            "data_path": data_args.data_path,
            "model_name_or_path": model_args.model_name_or_path,
            "run_name": training_args.run_name,
        }
        with open(os.path.join(training_args.output_dir, "saliency_training_config.json"), "w", encoding="utf-8") as f:
            json.dump(objective, f, ensure_ascii=False, indent=2)
        readme_path = os.path.join(training_args.output_dir, "README.md")
        with open(readme_path, "a", encoding="utf-8") as f:
            f.write(
                "\n## GraphSignal Training Objective\n\n"
                f"- Loss mode: `{training_args.loss_mode}`\n"
                f"- Saliency loss: `{objective['saliency_loss_name']}` (`{training_args.saliency_loss_type}`)\n"
                f"- Saliency lambda: `{training_args.saliency_lambda}`\n"
                f"- Saliency temperature tau: `{training_args.saliency_alpha}`\n"
                f"- Numerical epsilon: `{training_args.saliency_eps}`\n"
                f"- Floor epsilon mode: `{training_args.saliency_floor_eps_mode}`\n"
                f"- Floor epsilon fixed value: `{training_args.saliency_floor_eps}`\n"
                f"- Floor logit epsilon: `{training_args.saliency_floor_logit_eps}`\n"
                f"- Floor epsilon quantile: `{training_args.saliency_floor_quantile}`\n"
                f"- Floor epsilon warmup steps: `{training_args.saliency_floor_warmup_steps}`\n"
                f"- Training data: `{data_args.data_path}`\n"
            )


if __name__ == "__main__":
    train()
