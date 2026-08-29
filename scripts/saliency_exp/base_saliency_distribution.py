#!/usr/bin/env python3
"""Profile the initial teacher-forcing saliency distribution of a base model.

The script runs the base model over annotated SFT samples, computes last-layer
contribution saliency rows with the same definition used by training, and
streams summary statistics for all causal source/query pairs s < q.

By default it keeps exact count/mean/std/min/max and uses a reservoir sample per
bucket for quantiles and histograms. Set --max_values_per_bucket 0 if you want
exact quantiles/histograms and have enough host memory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRAIN_DIR = ROOT / "src" / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from dataset import AnnotatedSFTDataset, IGNORE_INDEX  # noqa: E402
from loss import _unwrap_to_decoder_stack  # noqa: E402


DEFAULT_MODEL = ROOT / "/mnt/nvme0n1/wenhao/models/Empirical-Influence-Function/Qwen2.5-Coder-7B-Instruct"
DEFAULT_DATA = ROOT / "data/go_single/train_data/go_single_train_v2_graphsignal_500_compact.json"
DEFAULT_OUT_DIR = ROOT / "outputs/saliency_exp/base_go_single_500_saliency_distribution"


def parse_quantiles(text: str) -> list[float]:
    out: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value > 1.0:
            value /= 100.0
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"quantile must be in [0, 1] or [0, 100], got {part!r}")
        out.append(value)
    if not out:
        raise ValueError("at least one quantile is required")
    return out


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def first_completion_index(labels: torch.Tensor) -> int:
    hits = torch.nonzero(labels.ne(IGNORE_INDEX), as_tuple=False)
    if hits.numel() == 0:
        return int(labels.numel())
    return int(hits[0].item())


def normalize_pairs(pairs: torch.Tensor, seq_len: int) -> dict[int, list[int]]:
    by_q: dict[int, set[int]] = {}
    if pairs.numel() == 0:
        return {}
    for src_raw, dst_raw in pairs.detach().cpu().tolist():
        src = int(min(src_raw, dst_raw))
        dst = int(max(src_raw, dst_raw))
        if 0 <= src < dst < seq_len:
            by_q.setdefault(dst, set()).add(src)
    return {q: sorted(srcs) for q, srcs in by_q.items()}


def quantile_key(q: float) -> str:
    if q in {0.0, 1.0}:
        return f"p{int(q * 100):03d}"
    pct = q * 100.0
    return f"p{pct:g}".replace(".", "_")


@dataclass
class StreamingBucket:
    name: str
    max_values: int
    generator: torch.Generator
    count: int = 0
    sum_value: float = 0.0
    sum_sq_value: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf
    _chunks: list[torch.Tensor] = field(default_factory=list)
    _reservoir: torch.Tensor | None = None
    _reservoir_size: int = 0

    def add(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        vals = values.detach().float().reshape(-1).cpu()
        n = int(vals.numel())
        self.count += n
        self.sum_value += float(vals.sum().item())
        self.sum_sq_value += float(vals.square().sum().item())
        self.min_value = min(self.min_value, float(vals.min().item()))
        self.max_value = max(self.max_value, float(vals.max().item()))

        if self.max_values <= 0:
            self._chunks.append(vals)
            return

        if self._reservoir is None:
            self._reservoir = torch.empty(self.max_values, dtype=torch.float32)

        if self._reservoir_size < self.max_values:
            take = min(self.max_values - self._reservoir_size, n)
            self._reservoir[self._reservoir_size:self._reservoir_size + take] = vals[:take]
            self._reservoir_size += take
            vals = vals[take:]
            n = int(vals.numel())
            if n == 0:
                return

        # Vectorized reservoir sampling for the remaining new values. The exact
        # count before this chunk is current count minus remaining chunk length.
        seen_before = self.count - n
        denom = torch.arange(
            seen_before + 1,
            seen_before + n + 1,
            dtype=torch.float32,
        )
        keep = torch.rand(n, generator=self.generator) < (float(self.max_values) / denom)
        if bool(keep.any()):
            kept = vals[keep]
            replace_at = torch.randint(
                self.max_values,
                (int(kept.numel()),),
                generator=self.generator,
            )
            self._reservoir[replace_at] = kept

    def values_for_distribution(self) -> torch.Tensor:
        if self.max_values <= 0:
            if not self._chunks:
                return torch.empty(0, dtype=torch.float32)
            return torch.cat(self._chunks).float()
        if self._reservoir is None or self._reservoir_size == 0:
            return torch.empty(0, dtype=torch.float32)
        return self._reservoir[:self._reservoir_size].float()

    def summary(self, quantiles: list[float]) -> dict[str, Any]:
        dist_values = self.values_for_distribution()
        sampled = self.max_values > 0 and self.count > int(dist_values.numel())
        if self.count == 0:
            out: dict[str, Any] = {
                "n_values": 0,
                "n_values_used_for_quantiles": 0,
                "sampled_quantiles": False,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
            out.update({quantile_key(q): 0.0 for q in quantiles})
            return out

        mean = self.sum_value / self.count
        variance = max(self.sum_sq_value / self.count - mean * mean, 0.0)
        out = {
            "n_values": int(self.count),
            "n_values_used_for_quantiles": int(dist_values.numel()),
            "sampled_quantiles": bool(sampled),
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.min_value,
            "max": self.max_value,
        }
        if dist_values.numel():
            qs = torch.tensor(quantiles, dtype=torch.float32)
            qvals = torch.quantile(dist_values, qs)
            out.update({quantile_key(q): float(v.item()) for q, v in zip(quantiles, qvals)})
        else:
            out.update({quantile_key(q): 0.0 for q in quantiles})
        return out

    def histogram(self, bins: int, *, log10: bool = False, eps: float = 1e-12) -> dict[str, Any]:
        values = self.values_for_distribution()
        if values.numel() == 0:
            return {"bin_edges": [], "counts": [], "density": [], "log10": log10}
        vals = torch.log10(values.clamp_min(eps)) if log10 else values
        if float(vals.min()) == float(vals.max()):
            center = float(vals.min())
            edges = torch.tensor([center - 0.5, center + 0.5], dtype=torch.float32)
            counts = torch.tensor([int(vals.numel())], dtype=torch.float32)
        else:
            counts = torch.histc(vals, bins=bins, min=float(vals.min()), max=float(vals.max()))
            edges = torch.linspace(float(vals.min()), float(vals.max()), bins + 1)
        density = counts / counts.sum().clamp_min(1.0)
        return {
            "bin_edges": [float(x) for x in edges.tolist()],
            "counts": [int(x) for x in counts.to(torch.long).tolist()],
            "density": [float(x) for x in density.tolist()],
            "log10": log10,
            "n_values_used": int(values.numel()),
        }


def build_buckets(names: list[str], max_values: int, seed: int) -> dict[str, StreamingBucket]:
    return {
        name: StreamingBucket(
            name=name,
            max_values=max_values,
            generator=torch.Generator().manual_seed(seed + i * 9973),
        )
        for i, name in enumerate(names)
    }


def add_masked(bucket: StreamingBucket, rows: torch.Tensor, mask: torch.Tensor) -> None:
    if bool(mask.any()):
        bucket.add(rows[mask])


def iter_contribution_row_chunks(
    model: Any,
    last_hidden_in: torch.Tensor,
    attn_probs: torch.Tensor,
    *,
    query_chunk_size: int,
    source_chunk_size: int,
):
    """Yield (row_qry, C_rows) chunks without recomputing transformed values."""
    decoder = _unwrap_to_decoder_stack(model)
    layer = decoder.layers[-1]
    self_attn = layer.self_attn

    B, T, D = last_hidden_in.shape
    if B != 1:
        raise ValueError("base_saliency_distribution.py currently processes one sample at a time")
    H = attn_probs.size(1)
    device = last_hidden_in.device
    dtype = last_hidden_in.dtype

    head_dim = getattr(self_attn, "head_dim", None) or (self_attn.q_proj.weight.shape[0] // H)
    v_out = self_attn.v_proj.weight.shape[0]
    num_kv_heads = v_out // head_dim
    if H % num_kv_heads != 0:
        raise ValueError(f"H={H} not divisible by num_kv_heads={num_kv_heads}")
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

    attn_f = attn_probs.float()
    eps_rms = getattr(layer.input_layernorm, "variance_epsilon", 1e-6)

    for q_start in range(1, T, query_chunk_size):
        q_stop = min(T, q_start + query_chunk_size)
        row_qry = torch.arange(q_start, q_stop, device=device, dtype=torch.long)
        row_batch = torch.zeros_like(row_qry)
        Q = int(row_qry.numel())
        query_hidden = last_hidden_in.float()[row_batch, row_qry, :]
        sigma_q = query_hidden.pow(2).mean(dim=-1).add(eps_rms).sqrt().clamp_min(1e-12)

        row_chunks = []
        for s_start in range(0, T, source_chunk_size):
            s_stop = min(T, s_start + source_chunk_size)
            attn_chunk = attn_f[row_batch, :, row_qry, s_start:s_stop]
            transformed_chunk = transformed[row_batch, :, s_start:s_stop, :]
            contrib = torch.einsum("qhs,qhso->qso", attn_chunk, transformed_chunk)

            diag_mask = (row_qry >= s_start) & (row_qry < s_stop)
            if bool(diag_mask.any()):
                local = row_qry[diag_mask] - s_start
                contrib[diag_mask, local, :] = contrib[diag_mask, local, :] + query_hidden[diag_mask]

            contrib = contrib / sigma_q.view(Q, 1, 1)
            row_chunks.append(contrib.norm(dim=-1, p=2).to(dtype))

        yield row_qry, torch.cat(row_chunks, dim=1)


def plot_histograms(histograms: dict[str, Any], output_path: Path, *, log10: bool, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preferred = ["all_causal", "annotated", "non_annotated", "prompt_to_completion", "completion_to_completion"]
    names = [name for name in preferred if name in histograms]
    if not names:
        names = list(histograms)

    plt.figure(figsize=(11, 6))
    for name in names:
        hist = histograms[name]["log10" if log10 else "linear"]
        edges = hist.get("bin_edges", [])
        density = hist.get("density", [])
        if len(edges) < 2 or not density:
            continue
        centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(density))]
        plt.plot(centers, density, label=name, linewidth=1.6)
    plt.xlabel("log10(saliency)" if log10 else "saliency")
    plt.ylabel("frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default=str(DEFAULT_MODEL))
    parser.add_argument("--data_path", default=str(DEFAULT_DATA))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all loaded samples.")
    parser.add_argument("--max_len", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query_chunk_size", type=int, default=64)
    parser.add_argument("--source_chunk_size", type=int, default=8)
    parser.add_argument("--max_values_per_bucket", type=int, default=2_000_000)
    parser.add_argument("--quantiles", default="0,0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99,1.0")
    parser.add_argument("--hist_bins", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--write_sample_stats", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quantiles = parse_quantiles(args.quantiles)
    dtype = dtype_from_name(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        padding_side="right",
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = AnnotatedSFTDataset(args.data_path, tokenizer=tokenizer, max_len=args.max_len)
    if args.max_samples and args.max_samples > 0:
        dataset.items = dataset.items[:args.max_samples]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False
    model.to(device)
    model.eval()

    bucket_names = [
        "all_causal",
        "annotated",
        "non_annotated",
        "prompt_to_prompt",
        "prompt_to_completion",
        "completion_to_completion",
    ]
    buckets = build_buckets(bucket_names, args.max_values_per_bucket, args.seed)

    sample_stats_path = output_dir / "sample_stats.jsonl"
    sample_stats_file = sample_stats_path.open("w", encoding="utf-8") if args.write_sample_stats else None

    metadata = {
        "model_name_or_path": args.model_name_or_path,
        "data_path": args.data_path,
        "num_loaded_samples": len(dataset),
        "max_samples": args.max_samples,
        "max_len": args.max_len,
        "dtype": args.dtype,
        "device": str(device),
        "query_chunk_size": args.query_chunk_size,
        "source_chunk_size": args.source_chunk_size,
        "max_values_per_bucket": args.max_values_per_bucket,
        "quantiles": quantiles,
        "hist_bins": args.hist_bins,
        "saliency_definition": "Last-layer contribution rows from src/train/loss.py::build_contribution_rows; causal values use source index s < query index q.",
    }

    total_annotation_edges = 0
    total_query_rows = 0

    try:
        with torch.inference_mode():
            for sample_idx, item in enumerate(dataset):
                input_ids = item["input_ids"].to(device).unsqueeze(0)
                labels = item["labels"]
                seq_len = int(input_ids.size(1))
                completion_start = first_completion_index(labels)
                annot_by_q = normalize_pairs(item["annot_pairs"], seq_len)
                total_annotation_edges += sum(len(v) for v in annot_by_q.values())

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids, device=device),
                    output_attentions=True,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                attn_last = outputs.attentions[-1]
                last_hidden_in = outputs.hidden_states[-2]

                sample_bucket_names = ["all_causal", "annotated", "non_annotated"]
                sample_buckets = build_buckets(sample_bucket_names, 0, args.seed + sample_idx * 104729)

                for row_qry, C_rows in iter_contribution_row_chunks(
                    model,
                    last_hidden_in,
                    attn_last,
                    query_chunk_size=args.query_chunk_size,
                    source_chunk_size=args.source_chunk_size,
                ):
                    C_rows = C_rows.float()
                    total_query_rows += int(row_qry.numel())

                    src_idx = torch.arange(seq_len, device=device).unsqueeze(0)
                    q_col = row_qry.unsqueeze(1)
                    causal_mask = src_idx < q_col

                    annot_mask = torch.zeros_like(causal_mask, dtype=torch.bool)
                    for local_i, q in enumerate(row_qry.detach().cpu().tolist()):
                        srcs = annot_by_q.get(int(q))
                        if srcs:
                            annot_mask[local_i, torch.tensor(srcs, device=device, dtype=torch.long)] = True
                    annot_mask &= causal_mask
                    non_annot_mask = causal_mask & (~annot_mask)

                    q_prompt = row_qry < completion_start
                    src_prompt = src_idx < completion_start
                    prompt_to_prompt = causal_mask & q_prompt.unsqueeze(1) & src_prompt
                    prompt_to_completion = causal_mask & (~q_prompt).unsqueeze(1) & src_prompt
                    completion_to_completion = causal_mask & (~q_prompt).unsqueeze(1) & (~src_prompt)

                    add_masked(buckets["all_causal"], C_rows, causal_mask)
                    add_masked(buckets["annotated"], C_rows, annot_mask)
                    add_masked(buckets["non_annotated"], C_rows, non_annot_mask)
                    add_masked(buckets["prompt_to_prompt"], C_rows, prompt_to_prompt)
                    add_masked(buckets["prompt_to_completion"], C_rows, prompt_to_completion)
                    add_masked(buckets["completion_to_completion"], C_rows, completion_to_completion)

                    if sample_stats_file is not None:
                        add_masked(sample_buckets["all_causal"], C_rows, causal_mask)
                        add_masked(sample_buckets["annotated"], C_rows, annot_mask)
                        add_masked(sample_buckets["non_annotated"], C_rows, non_annot_mask)

                    del C_rows

                if sample_stats_file is not None:
                    sample_stats_file.write(json.dumps({
                        "sample_index": sample_idx,
                        "seq_len": seq_len,
                        "completion_start": completion_start,
                        "num_annotation_edges": sum(len(v) for v in annot_by_q.values()),
                        "buckets": {
                            name: bucket.summary([0.25, 0.50, 0.75])
                            for name, bucket in sample_buckets.items()
                        },
                    }, ensure_ascii=False) + "\n")
                    sample_stats_file.flush()

                if args.progress_every > 0 and (sample_idx + 1) % args.progress_every == 0:
                    print(
                        f"[{sample_idx + 1}/{len(dataset)}] "
                        f"seq_len={seq_len} completion_start={completion_start} "
                        f"all_causal_values={buckets['all_causal'].count}",
                        flush=True,
                    )

                del outputs, attn_last, last_hidden_in
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        if sample_stats_file is not None:
            sample_stats_file.close()

    summary = {
        "metadata": metadata,
        "num_samples_processed": len(dataset),
        "num_query_rows_processed": total_query_rows,
        "num_annotation_edges": total_annotation_edges,
        "buckets": {
            name: bucket.summary(quantiles)
            for name, bucket in buckets.items()
        },
    }

    histograms = {
        name: {
            "linear": bucket.histogram(args.hist_bins, log10=False),
            "log10": bucket.histogram(args.hist_bins, log10=True),
        }
        for name, bucket in buckets.items()
    }

    summary_path = output_dir / "summary.json"
    hist_path = output_dir / "histograms.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    hist_path.write_text(json.dumps(histograms, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_histograms(histograms, output_dir / "hist_linear.png", log10=False, title="Base saliency distribution")
    plot_histograms(histograms, output_dir / "hist_log10.png", log10=True, title="Base saliency distribution, log10 scale")

    print(f"saved summary -> {summary_path}")
    print(f"saved histograms -> {hist_path}")
    print(f"saved linear histogram -> {output_dir / 'hist_linear.png'}")
    print(f"saved log histogram -> {output_dir / 'hist_log10.png'}")


if __name__ == "__main__":
    main()
