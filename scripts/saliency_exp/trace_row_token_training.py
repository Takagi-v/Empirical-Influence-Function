from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainerCallback

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SRC = ROOT / "src" / "train"
sys.path.insert(0, str(TRAIN_SRC))

from dataset import AnnotatedSFTDataset, DataCollatorForAnnotatedSFT  # noqa: E402
from loss import (  # noqa: E402
    _annotation_rows_from_pairs,
    build_contribution_rows,
    canonical_saliency_loss_type,
    compute_saliency_loss_from_rows,
    saliency_loss_display_name,
)
from train import AnnotatedSFTTrainer, DataArguments, ModelArguments, SFTTrainingArguments  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "annot_pairs":
            moved[key] = [x.to(device) for x in value]
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _trainable_params(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _grad_stats(
    ce_loss: torch.Tensor,
    sal_loss: torch.Tensor,
    params: list[torch.nn.Parameter],
    *,
    gamma: float,
) -> dict[str, float]:
    ce_grads = torch.autograd.grad(
        ce_loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    sal_grads = torch.autograd.grad(
        sal_loss,
        params,
        retain_graph=False,
        allow_unused=True,
    )

    ce_sq = torch.zeros((), device=ce_loss.device, dtype=torch.float32)
    sal_sq = torch.zeros((), device=ce_loss.device, dtype=torch.float32)
    total_sq = torch.zeros((), device=ce_loss.device, dtype=torch.float32)
    dot = torch.zeros((), device=ce_loss.device, dtype=torch.float32)

    for ce_g, sal_g in zip(ce_grads, sal_grads):
        if ce_g is None and sal_g is None:
            continue
        if ce_g is None:
            sal_f = sal_g.detach().float()
            sal_sq = sal_sq + sal_f.pow(2).sum()
            total_sq = total_sq + (float(gamma) * sal_f).pow(2).sum()
            continue
        if sal_g is None:
            ce_f = ce_g.detach().float()
            ce_sq = ce_sq + ce_f.pow(2).sum()
            total_sq = total_sq + ce_f.pow(2).sum()
            continue
        ce_f = ce_g.detach().float()
        sal_f = sal_g.detach().float()
        ce_sq = ce_sq + ce_f.pow(2).sum()
        sal_sq = sal_sq + sal_f.pow(2).sum()
        dot = dot + (ce_f * sal_f).sum()
        total_sq = total_sq + (ce_f + float(gamma) * sal_f).pow(2).sum()

    ce_norm = torch.sqrt(ce_sq).item()
    sal_norm = torch.sqrt(sal_sq).item()
    total_norm = torch.sqrt(total_sq).item()
    denom = ce_norm * sal_norm
    cosine = float(dot.item() / denom) if denom > 0 else 0.0
    return {
        "ce_grad_norm": float(ce_norm),
        "sal_grad_norm": float(sal_norm),
        "ce_sal_grad_cosine": float(cosine),
        "sample_total_grad_norm": float(total_norm),
    }


def _display_token(token: str) -> str:
    return token.replace("\n", "\\n").replace("\t", "\\t")


def _safe_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu())
    return float(value)


@dataclass
class TraceArguments:
    trace_output_dir: str = field(
        default="outputs/saliency_exp/row479_service_training_trace",
        metadata={"help": "Directory for JSONL/CSV/PNG diagnostics."},
    )
    trace_sample_index: int = field(
        default=479,
        metadata={"help": "Line/dataset index of the tracked sample."},
    )
    trace_query_token_index: int = field(
        default=251,
        metadata={"help": "Tracked query token index q. For row 479 this is the completion token 'Service'."},
    )
    trace_query_token_text: str = field(
        default="Service",
        metadata={"help": "Optional sanity-check string for the tracked query token decode."},
    )
    trace_top_k: int = field(default=20, metadata={"help": "Top-k causal source tokens to track."})
    trace_every_n_steps: int = field(default=1, metadata={"help": "Run diagnostics every N optimizer steps."})
    trace_before_train: bool = field(default=True, metadata={"help": "Also record the initial step-0 state."})
    trace_model_mode: str = field(
        default="eval",
        metadata={"help": "Model mode during diagnostics: eval for deterministic dropout-free probes, or train."},
    )
    trace_source_chunk_size: int = field(
        default=32,
        metadata={"help": "Source chunk size for contribution-row computation."},
    )
    trace_negative_denominator: str = field(
        default="sum",
        metadata={"help": "Diagnostic p denominator for negatives: sum matches current loss.py; mean matches the note with 1/|Nq|."},
    )
    trace_log_token_tables: bool = field(
        default=True,
        metadata={"help": "Print compact top-token tables to the training log."},
    )


class RowTokenTraceCallback(TrainerCallback):
    def __init__(
        self,
        *,
        tokenizer,
        sample_item: dict[str, torch.Tensor],
        trace_args: TraceArguments,
        saliency_alpha: float,
        saliency_eps: float,
        saliency_lambda: float,
        saliency_floor_eps: float,
        saliency_floor_eps_mode: str,
        saliency_floor_quantile: float,
        saliency_floor_ema_beta: float,
        saliency_floor_min_eps: float,
        saliency_floor_warmup_steps: int,
        saliency_floor_logit_eps: float | None,
        saliency_loss_type: str,
    ):
        self.tokenizer = tokenizer
        self.sample_item = sample_item
        self.trace_args = trace_args
        self.saliency_alpha = float(saliency_alpha)
        self.saliency_eps = float(saliency_eps)
        self.saliency_lambda = float(saliency_lambda)
        self.saliency_floor_eps = float(saliency_floor_eps)
        self.saliency_floor_eps_mode = saliency_floor_eps_mode
        self.saliency_floor_quantile = float(saliency_floor_quantile)
        self.saliency_floor_ema_beta = float(saliency_floor_ema_beta)
        self.saliency_floor_min_eps = float(saliency_floor_min_eps)
        self.saliency_floor_warmup_steps = int(saliency_floor_warmup_steps)
        self.saliency_floor_logit_eps = saliency_floor_logit_eps
        self.saliency_loss_type = canonical_saliency_loss_type(saliency_loss_type)
        self.trace_output_dir = Path(trace_args.trace_output_dir)
        self.trace_jsonl_path = self.trace_output_dir / "trace.jsonl"
        self.sample_csv_path = self.trace_output_dir / "sample_metrics.csv"
        self.token_csv_path = self.trace_output_dir / "token_summary.csv"
        self.summary_md_path = self.trace_output_dir / "README.md"
        self.streak_by_source: dict[int, int] = {}
        self.records: list[dict[str, Any]] = []
        self._wrote_header = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        self.trace_output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_jsonl_path.write_text("", encoding="utf-8")
        if self.trace_args.trace_before_train and model is not None:
            self._trace(model=model, step=0, stage="before_train")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or model is None:
            return
        every = max(1, int(self.trace_args.trace_every_n_steps))
        if int(state.global_step) % every != 0:
            return
        self._trace(model=model, step=int(state.global_step), stage="after_step")

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._write_csvs()
        self._write_plots()
        self._write_readme()

    def _trace(self, *, model, step: int, stage: str) -> None:
        was_training = model.training
        if self.trace_args.trace_model_mode.strip().lower() == "eval":
            model.eval()
        else:
            model.train()

        device = next(model.parameters()).device
        collator = DataCollatorForAnnotatedSFT(tokenizer=self.tokenizer)
        batch = collator([self.sample_item])
        batch = _move_batch_to_device(batch, device)
        params = _trainable_params(model)

        with torch.enable_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                output_attentions=True,
                output_hidden_states=True,
            )
            ce_loss = outputs.loss
            if ce_loss.dim() > 0:
                ce_loss = ce_loss.mean()

            attn_last = outputs.attentions[-1]
            if attn_last is None:
                raise RuntimeError("Need eager attention: outputs.attentions[-1] is None.")
            last_hidden_in = outputs.hidden_states[-2]
            B, T, _ = last_hidden_in.shape
            row_batch, row_qry, src_all, inv = _annotation_rows_from_pairs(
                batch["annot_pairs"],
                B=B,
                T=T,
                device=device,
            )
            C_rows = build_contribution_rows(
                model,
                last_hidden_in,
                attn_last,
                row_batch,
                row_qry,
                source_chunk_size=int(self.trace_args.trace_source_chunk_size),
            )
            sal_diag = compute_saliency_loss_from_rows(
                C_rows,
                row_batch,
                row_qry,
                src_all,
                inv,
                alpha=self.saliency_alpha,
                eps=self.saliency_eps,
                floor_eps=self.saliency_floor_eps,
                floor_eps_mode=self.saliency_floor_eps_mode,
                floor_eps_quantile=self.saliency_floor_quantile,
                floor_eps_ema_beta=self.saliency_floor_ema_beta,
                prev_floor_eps=None,
                floor_eps_min=self.saliency_floor_min_eps,
                floor_eps_step=max(1, int(step)),
                floor_eps_warmup_steps=self.saliency_floor_warmup_steps,
                floor_logit_eps=self.saliency_floor_logit_eps,
                loss_type=self.saliency_loss_type,
            )
            sal_loss = sal_diag.loss
            grad_stats = _grad_stats(
                ce_loss,
                sal_loss,
                params,
                gamma=self.saliency_lambda,
            )

        token_record = self._build_token_record(
            C_rows=C_rows.detach().float(),
            row_qry=row_qry.detach(),
            src_all=src_all.detach(),
            inv=inv.detach(),
            input_ids=batch["input_ids"][0].detach(),
            step=step,
        )

        lce = _safe_float(ce_loss)
        lsal = _safe_float(sal_loss)
        record = {
            "step": int(step),
            "stage": stage,
            "sample_index": int(self.trace_args.trace_sample_index),
            "query_token_index": int(self.trace_args.trace_query_token_index),
            "query_token_text": token_record["query_token_text"],
            "saliency_loss_type": self.saliency_loss_type,
            "saliency_loss_name": saliency_loss_display_name(self.saliency_loss_type),
            "saliency_lambda": self.saliency_lambda,
            "saliency_temperature_tau": self.saliency_alpha,
            "saliency_eps_num": self.saliency_eps,
            "saliency_floor_eps": self.saliency_floor_eps,
            "saliency_floor_logit_eps": self.saliency_floor_logit_eps,
            "trace_negative_denominator": self.trace_args.trace_negative_denominator,
            "sample_metrics": {
                "Lce": lce,
                "Lsal": lsal,
                "gamma_Lsal_over_Lce": float(self.saliency_lambda * lsal / max(lce, self.saliency_eps)),
                "Pbar_all_queries": sal_diag.avg_C,
                "Nbar_all_queries": sal_diag.avg_N,
                "ratio_all_queries": sal_diag.avg_ratio,
                **grad_stats,
            },
            "token_metrics": token_record,
        }
        self.records.append(record)
        with self.trace_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.trace_args.trace_log_token_tables:
            neg_names = " ".join(x["display"] for x in token_record["Nq_top20"])
            pos_names = " ".join(x["display"] for x in token_record["Pq_top20"])
            logger.info(
                "[trace row=%s q=%s %s step=%s] Lce=%.4g Lsal=%.4g gammaLsal/Lce=%.4g "
                "ce_g=%.4g sal_g=%.4g cos=%.4g total_g=%.4g Lq=%.4g Lq_norm=%.4g "
                "Pbar=%.4g Nbar=%.4g ratio=%.4g Ntop=[%s] Ptop=[%s]",
                self.trace_args.trace_sample_index,
                self.trace_args.trace_query_token_index,
                token_record["query_token_text"],
                step,
                lce,
                lsal,
                record["sample_metrics"]["gamma_Lsal_over_Lce"],
                grad_stats["ce_grad_norm"],
                grad_stats["sal_grad_norm"],
                grad_stats["ce_sal_grad_cosine"],
                grad_stats["sample_total_grad_norm"],
                token_record["Lq"],
                token_record["Lq_norm"],
                token_record["Pbar"],
                token_record["Nbar"],
                token_record["ratio"],
                neg_names,
                pos_names,
            )

        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
        else:
            model.eval()

    def _build_token_record(
        self,
        *,
        C_rows: torch.Tensor,
        row_qry: torch.Tensor,
        src_all: torch.Tensor,
        inv: torch.Tensor,
        input_ids: torch.Tensor,
        step: int,
    ) -> dict[str, Any]:
        q = int(self.trace_args.trace_query_token_index)
        q_matches = torch.nonzero(row_qry.cpu() == q, as_tuple=False).flatten()
        if q_matches.numel() == 0:
            raise RuntimeError(f"Tracked query token q={q} has no annotation row.")
        qrow = int(q_matches[0])
        T = int(C_rows.size(1))
        if not (0 <= q < T):
            raise RuntimeError(f"Tracked query token q={q} is out of range T={T}.")

        query_text = _display_token(self.tokenizer.decode([int(input_ids[q].cpu())]))
        expected = self.trace_args.trace_query_token_text
        if expected and expected not in query_text:
            logger.warning(
                "Tracked q token text mismatch: q=%s decoded=%r expected_contains=%r",
                q,
                query_text,
                expected,
            )

        edge_mask = inv.cpu() == qrow
        P_sources = sorted({int(x) for x in src_all.cpu()[edge_mask].tolist() if int(x) < q})
        P_set = set(P_sources)
        causal_sources = list(range(q))
        N_sources = [s for s in causal_sources if s not in P_set]
        C_q = C_rows[qrow].float()
        top_k = min(max(1, int(self.trace_args.trace_top_k)), q)
        top_sources = torch.topk(C_q[:q], k=top_k, largest=True).indices.cpu().tolist()
        top_sources = [int(x) for x in top_sources]

        N_top = [s for s in top_sources if s not in P_set]
        P_top = [s for s in top_sources if s in P_set]
        for src in list(self.streak_by_source):
            if src not in N_top:
                self.streak_by_source[src] = 0
        for src in N_top:
            self.streak_by_source[src] = self.streak_by_source.get(src, 0) + 1

        eps = float(self.saliency_eps)
        tau = max(float(self.saliency_alpha), eps)
        floor_logit = (
            float(self.saliency_floor_logit_eps)
            if self.saliency_floor_logit_eps is not None
            else math.log(float(self.saliency_floor_eps) + eps) / tau
        )

        logits = torch.log(C_q + eps) / tau
        neg_logits = torch.maximum(logits, torch.tensor(floor_logit, device=logits.device))
        neg_terms = {s: float(torch.exp(neg_logits[s]).cpu()) for s in N_sources}
        neg_den_raw = sum(neg_terms.values())
        use_mean_neg = self.trace_args.trace_negative_denominator.strip().lower() == "mean"
        neg_scale = max(1, len(N_sources)) if use_mean_neg else 1
        neg_den = neg_den_raw / neg_scale

        P_values = [float(C_q[s].cpu()) for s in P_sources]
        N_values = [float(C_q[s].cpu()) for s in N_sources]
        Pbar = sum(P_values) / max(1, len(P_values))
        Nbar = sum(N_values) / max(1, len(N_values))
        ratio = Pbar / max(Nbar, eps)

        pos_rows = []
        pos_losses = []
        for src in P_sources:
            exp_l = float(torch.exp(logits[src]).cpu())
            denom = neg_den + exp_l
            p = exp_l / max(denom, eps)
            loss_qs = -math.log(max(p, eps))
            pos_losses.append(loss_qs)
            if src in P_top:
                pos_rows.append({
                    "source": src,
                    "token_text": _display_token(self.tokenizer.decode([int(input_ids[src].cpu())])),
                    "display": f"{_display_token(self.tokenizer.decode([int(input_ids[src].cpu())]))}$",
                    "Cqs": float(C_q[src].cpu()),
                    "lqs": float(logits[src].cpu()),
                    "exp_lqs": exp_l,
                    "pqs": p,
                    "loss_qs": loss_qs,
                    "rank": top_sources.index(src) + 1,
                })

        neg_rows = []
        pos_denoms = []
        for pos in P_sources:
            exp_l_pos = float(torch.exp(logits[pos]).cpu())
            pos_denoms.append((pos, neg_den + exp_l_pos))
        for src in N_top:
            token_text = _display_token(self.tokenizer.decode([int(input_ids[src].cpu())]))
            neg_contrib = neg_terms[src] / neg_scale
            p_by_pos = []
            for pos, denom in pos_denoms:
                p_by_pos.append({
                    "positive_source": int(pos),
                    "positive_token_text": _display_token(self.tokenizer.decode([int(input_ids[pos].cpu())])),
                    "pqs": float(neg_contrib / max(denom, eps)),
                })
            p_mean = (
                sum(x["pqs"] for x in p_by_pos) / len(p_by_pos)
                if p_by_pos else 0.0
            )
            sticky = self.streak_by_source.get(src, 0) >= 5
            neg_rows.append({
                "source": src,
                "token_text": token_text,
                "display": f"{token_text}{'*' if sticky else ''}",
                "sticky_5_steps": bool(sticky),
                "top20_streak": int(self.streak_by_source.get(src, 0)),
                "Cqs": float(C_q[src].cpu()),
                "lqs": float(logits[src].cpu()),
                "exp_max_lqs_floor": float(neg_terms[src]),
                "pqs_mean_over_P": float(p_mean),
                "pqs_by_positive": p_by_pos,
                "rank": top_sources.index(src) + 1,
            })

        Lq = sum(pos_losses) / max(1, len(pos_losses))
        Lq_norm = Lq / max(math.log(1 + max(1, len(N_sources))), eps)
        return {
            "step": int(step),
            "query": q,
            "query_token_text": query_text,
            "num_P": len(P_sources),
            "num_N": len(N_sources),
            "top_k": top_k,
            "P_sources": P_sources,
            "N_top20_sources": N_top,
            "P_top20_sources": P_top,
            "Pbar": float(Pbar),
            "Nbar": float(Nbar),
            "ratio": float(ratio),
            "Lq": float(Lq),
            "Lq_norm": float(Lq_norm),
            "negative_floor_logit": float(floor_logit),
            "negative_denominator_mode": self.trace_args.trace_negative_denominator,
            "negative_denominator_value": float(neg_den),
            "Nq_top20": neg_rows,
            "Pq_top20": pos_rows,
        }

    def _write_csvs(self) -> None:
        if not self.records:
            return
        with self.sample_csv_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "step",
                "stage",
                "Lce",
                "Lsal",
                "gamma_Lsal_over_Lce",
                "ce_grad_norm",
                "sal_grad_norm",
                "ce_sal_grad_cosine",
                "sample_total_grad_norm",
                "Pbar_all_queries",
                "Nbar_all_queries",
                "ratio_all_queries",
                "Lq",
                "Lq_norm",
                "Pbar",
                "Nbar",
                "ratio",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.records:
                sm = rec["sample_metrics"]
                tm = rec["token_metrics"]
                writer.writerow({
                    "step": rec["step"],
                    "stage": rec["stage"],
                    "Lce": sm["Lce"],
                    "Lsal": sm["Lsal"],
                    "gamma_Lsal_over_Lce": sm["gamma_Lsal_over_Lce"],
                    "ce_grad_norm": sm["ce_grad_norm"],
                    "sal_grad_norm": sm["sal_grad_norm"],
                    "ce_sal_grad_cosine": sm["ce_sal_grad_cosine"],
                    "sample_total_grad_norm": sm["sample_total_grad_norm"],
                    "Pbar_all_queries": sm["Pbar_all_queries"],
                    "Nbar_all_queries": sm["Nbar_all_queries"],
                    "ratio_all_queries": sm["ratio_all_queries"],
                    "Lq": tm["Lq"],
                    "Lq_norm": tm["Lq_norm"],
                    "Pbar": tm["Pbar"],
                    "Nbar": tm["Nbar"],
                    "ratio": tm["ratio"],
                })

        with self.token_csv_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "step",
                "kind",
                "source",
                "rank",
                "display",
                "Cqs",
                "lqs",
                "exp_value",
                "pqs",
                "sticky_5_steps",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.records:
                step = rec["step"]
                tm = rec["token_metrics"]
                for row in tm["Nq_top20"]:
                    writer.writerow({
                        "step": step,
                        "kind": "N",
                        "source": row["source"],
                        "rank": row["rank"],
                        "display": row["display"],
                        "Cqs": row["Cqs"],
                        "lqs": row["lqs"],
                        "exp_value": row["exp_max_lqs_floor"],
                        "pqs": row["pqs_mean_over_P"],
                        "sticky_5_steps": row["sticky_5_steps"],
                    })
                for row in tm["Pq_top20"]:
                    writer.writerow({
                        "step": step,
                        "kind": "P",
                        "source": row["source"],
                        "rank": row["rank"],
                        "display": row["display"],
                        "Cqs": row["Cqs"],
                        "lqs": row["lqs"],
                        "exp_value": row["exp_lqs"],
                        "pqs": row["pqs"],
                        "sticky_5_steps": "",
                    })

    def _write_plots(self) -> None:
        if not self.records:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [r["step"] for r in self.records]
        sample = [r["sample_metrics"] for r in self.records]
        token = [r["token_metrics"] for r in self.records]

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        axes[0, 0].plot(steps, [x["Lce"] for x in sample], label="Lce")
        axes[0, 0].plot(steps, [x["Lsal"] for x in sample], label="Lsal")
        axes[0, 0].plot(steps, [x["gamma_Lsal_over_Lce"] for x in sample], label="gamma*Lsal/Lce")
        axes[0, 0].legend()
        axes[0, 0].set_title("Tracked Sample Loss")

        axes[0, 1].plot(steps, [x["ce_grad_norm"] for x in sample], label="CE grad norm")
        axes[0, 1].plot(steps, [x["sal_grad_norm"] for x in sample], label="SAL grad norm")
        axes[0, 1].plot(steps, [x["sample_total_grad_norm"] for x in sample], label="CE+gamma SAL grad norm")
        axes[0, 1].legend()
        axes[0, 1].set_title("Tracked Sample Grad Norms")

        axes[1, 0].plot(steps, [x["ce_sal_grad_cosine"] for x in sample], label="cos(CE, SAL)")
        axes[1, 0].axhline(0, color="black", linewidth=0.8)
        axes[1, 0].legend()
        axes[1, 0].set_title("Gradient Alignment")

        axes[1, 1].plot(steps, [x["Pbar"] for x in token], label="Pbar(q)")
        axes[1, 1].plot(steps, [x["Nbar"] for x in token], label="Nbar(q)")
        axes[1, 1].plot(steps, [x["ratio"] for x in token], label="Pbar/Nbar")
        axes[1, 1].legend()
        axes[1, 1].set_title("Tracked Token Saliency Split")
        fig.tight_layout()
        fig.savefig(self.trace_output_dir / "sample_and_grad_metrics.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(steps, [x["Lq"] for x in token], label="Lq")
        axes[0].plot(steps, [x["Lq_norm"] for x in token], label="Lq_norm")
        axes[0].legend()
        axes[0].set_title("Tracked Query Loss")
        axes[1].plot(steps, [len(x["P_top20_sources"]) for x in token], label="|Pq cap Top20|")
        axes[1].plot(steps, [len(x["N_top20_sources"]) for x in token], label="|Nq cap Top20|")
        axes[1].legend()
        axes[1].set_title("Top-20 Composition")
        fig.tight_layout()
        fig.savefig(self.trace_output_dir / "token_metrics.png", dpi=180)
        plt.close(fig)

    def _write_readme(self) -> None:
        text = f"""# Row/Token Saliency Training Trace

Tracked sample: row `{self.trace_args.trace_sample_index}`

Tracked query token q: `#{self.trace_args.trace_query_token_index}` decoded as `{self.records[-1]['token_metrics']['query_token_text'] if self.records else ''}`

Saliency loss: `{saliency_loss_display_name(self.saliency_loss_type)}` (`{self.saliency_loss_type}`)

Temperature tau: `{self.saliency_alpha}`

Numerical eps: `{self.saliency_eps}`

Floor logit eps: `{self.saliency_floor_logit_eps}`

Diagnostic negative denominator mode: `{self.trace_args.trace_negative_denominator}`

## Files

- `trace.jsonl`: full per-step nested diagnostics.
- `sample_metrics.csv`: one row per traced step for sample-level loss/gradient and q-level aggregate metrics.
- `token_summary.csv`: one row per top-20 token per traced step.
- `sample_and_grad_metrics.png`: loss, gradient norm, cosine, and P/N curves.
- `token_metrics.png`: q-level loss and top-20 composition curves.

## Metric Notes

- `Pq` is the annotated source set for the tracked query q.
- `Nq` is the causal non-annotated source set for q.
- `Tq20` is the top-20 causal source set ranked by current saliency score `Cqs`.
- Negative top tokens marked with `*` have stayed in `Nq cap Tq20` for at least 5 consecutive traced steps.
- Positive top tokens marked with `$` belong to `Pq cap Tq20`.
- `Pbar` and `Nbar` are the mean saliency scores over all sources in Pq and Nq.
- `Lq` is the current softmax-margin multi-positive NLL for this q.
- `Lq_norm = Lq / log(1 + |Nq|)` provides a scale-normalized q loss.
- Negative-token `pqs_mean_over_P` is the mean denominator contribution of that negative source across all positive sources.
"""
        self.summary_md_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, SFTTrainingArguments, TraceArguments)
    )
    model_args, data_args, training_args, trace_args = parser.parse_args_into_dataclasses()
    training_args.saliency_loss_type = canonical_saliency_loss_type(training_args.saliency_loss_type)
    training_args.remove_unused_columns = False

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

    train_dataset = AnnotatedSFTDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_len=data_args.max_len,
        language=data_args.language,
    )
    if not (0 <= trace_args.trace_sample_index < len(train_dataset)):
        raise ValueError(
            f"trace_sample_index={trace_args.trace_sample_index} is out of range for "
            f"loaded dataset length {len(train_dataset)}"
        )
    sample_item = train_dataset[int(trace_args.trace_sample_index)]
    input_ids = sample_item["input_ids"]
    q = int(trace_args.trace_query_token_index)
    if not (0 <= q < int(input_ids.numel())):
        raise ValueError(f"trace_query_token_index={q} is out of range for sample length {input_ids.numel()}")
    q_text = _display_token(tokenizer.decode([int(input_ids[q])]))
    logger.info(
        "Tracing row=%s q=%s token=%r len=%s annotations=%s",
        trace_args.trace_sample_index,
        q,
        q_text,
        int(input_ids.numel()),
        int(sample_item["annot_pairs"].size(0)),
    )

    if model_args.use_flash_attention:
        logger.warning("Flash attention hides attention probabilities; using eager attention for saliency trace.")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if model_args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    callback = RowTokenTraceCallback(
        tokenizer=tokenizer,
        sample_item=sample_item,
        trace_args=trace_args,
        saliency_alpha=training_args.saliency_alpha,
        saliency_eps=training_args.saliency_eps,
        saliency_lambda=training_args.saliency_lambda,
        saliency_floor_eps=training_args.saliency_floor_eps,
        saliency_floor_eps_mode=training_args.saliency_floor_eps_mode,
        saliency_floor_quantile=training_args.saliency_floor_quantile,
        saliency_floor_ema_beta=training_args.saliency_floor_ema_beta,
        saliency_floor_min_eps=training_args.saliency_floor_min_eps,
        saliency_floor_warmup_steps=training_args.saliency_floor_warmup_steps,
        saliency_floor_logit_eps=training_args.saliency_floor_logit_eps,
        saliency_loss_type=training_args.saliency_loss_type,
    )

    trainer = AnnotatedSFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        data_collator=DataCollatorForAnnotatedSFT(tokenizer=tokenizer),
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
        callbacks=[callback],
    )

    logger.info(
        "Training with trace: loss_mode=%s saliency=%s lambda=%s tau=%s floor_logit_eps=%s trace_dir=%s",
        training_args.loss_mode,
        saliency_loss_display_name(training_args.saliency_loss_type),
        training_args.saliency_lambda,
        training_args.saliency_alpha,
        training_args.saliency_floor_logit_eps,
        trace_args.trace_output_dir,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    if trainer.is_world_process_zero():
        os.makedirs(training_args.output_dir, exist_ok=True)
        config = {
            "data_path": data_args.data_path,
            "model_name_or_path": model_args.model_name_or_path,
            "output_dir": training_args.output_dir,
            "loss_mode": training_args.loss_mode,
            "saliency_loss_type": training_args.saliency_loss_type,
            "saliency_loss_name": saliency_loss_display_name(training_args.saliency_loss_type),
            "saliency_lambda": training_args.saliency_lambda,
            "saliency_temperature_tau": training_args.saliency_alpha,
            "saliency_eps_num": training_args.saliency_eps,
            "saliency_floor_eps": training_args.saliency_floor_eps,
            "saliency_floor_logit_eps": training_args.saliency_floor_logit_eps,
            "trace_output_dir": trace_args.trace_output_dir,
            "trace_sample_index": trace_args.trace_sample_index,
            "trace_query_token_index": trace_args.trace_query_token_index,
            "trace_query_token_text": q_text,
        }
        with open(os.path.join(training_args.output_dir, "row_token_trace_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
