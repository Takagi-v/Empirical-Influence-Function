#!/usr/bin/env python
"""Evaluate CodeSearchNet Go internal completion data.

This evaluator is intentionally separate from the MCEval-style Go evaluator:
CodeSearchNet internal rows do not contain unit tests, so pass@k is an exact
normalized completion match.  Saliency alignment is teacher-forcing alignment
against annotation edges using the same last-layer contribution rows as
src/train/loss.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IGNORE_INDEX = -100

DEFAULT_ANNOTATED = Path("data/go_single/eval_data/codesearchnet_go_test_1000_graphsignal_500_compact.json")
DEFAULT_EVAL = Path("data/go_single/eval_data/codesearchnet_go_test_1000_chatml.jsonl")
DEFAULT_OUTPUT = Path("outputs/go_singleline_fim_exp/eval_results/codesearchnet_go_internal_results.jsonl")
DEFAULT_SUMMARY = Path("outputs/go_singleline_fim_exp/eval_results/codesearchnet_go_internal_summary.json")
DEFAULT_TABLE = Path("outputs/go_singleline_fim_exp/eval_results/codesearchnet_go_internal_table.md")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def resolve_annotated_path(path: Path) -> tuple[Path, dict[str, str]]:
    if path.name.endswith(".failures.json"):
        failures = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(failures, dict):
            raise ValueError(f"Expected a JSON object in failures file: {path}")
        base = path.with_name(path.name[: -len(".failures.json")])
        if not base.exists():
            raise FileNotFoundError(f"Could not find compact data next to failures file: {base}")
        return base, {str(k): str(v) for k, v in failures.items()}
    return path, {}


def first_completion_idx(labels: list[int]) -> int | None:
    return next((i for i, value in enumerate(labels) if int(value) != IGNORE_INDEX), None)


def normalize_edge(edge: dict[str, Any]) -> tuple[int, int, str]:
    src = int(edge.get("src", edge.get("source", edge.get("token_i_idx", -1))))
    dst = int(edge.get("dst", edge.get("target", edge.get("token_j_idx", -1))))
    subtype = str(edge.get("subtype", edge.get("type", "")) or "")
    return src, dst, subtype


def causal_edges_by_target(
    row: dict[str, Any],
    seq_len: int,
    completion_start: int,
    target_scope: str,
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in row.get("attention_edges", []):
        src, dst, subtype = normalize_edge(edge)
        if not (0 <= src < dst < seq_len):
            continue
        if target_scope == "completion" and dst < completion_start:
            continue
        if target_scope == "prompt" and dst >= completion_start:
            continue
        grouped[dst].append({"src": src, "dst": dst, "subtype": subtype})
    return grouped


def load_predictions(path: Path) -> dict[str, list[str]]:
    pred_map: dict[str, list[str]] = {}
    for row in iter_jsonl(path):
        uid = str(row.get("uid") or "")
        if not uid:
            continue
        values = row.get("predictions")
        if values is None and "prediction" in row:
            values = [row.get("prediction")]
        if values is None:
            values = row.get("raw_generation")
        if values is None:
            continue
        if isinstance(values, str):
            values = [values]
        out = []
        for item in values:
            if isinstance(item, dict):
                text = item.get("text") or item.get("prediction") or item.get("content") or ""
            else:
                text = item
            if text is not None:
                out.append(str(text))
        pred_map[uid] = out
    return pred_map


def sanitize_prediction(text: str) -> str:
    try:
        from scripts.benchmark.eval_judges import sanitize_prediction as _sanitize

        return _sanitize(text)
    except Exception:
        return text


def normalize_completion(text: str) -> str:
    return sanitize_prediction(text).strip()


def import_codebleu():
    try:
        from codebleu import calc_codebleu  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CodeBLEU requires the `codebleu` package in eif-bench. "
            "Activate the env and install it if needed."
        ) from exc
    return calc_codebleu


def empty_codebleu_bucket() -> dict[str, float]:
    return {
        "n_total": 0,
        "n_supported": 0,
        "n_failed": 0,
        "codebleu_sum": 0.0,
        "ngram_match_score_sum": 0.0,
        "weighted_ngram_match_score_sum": 0.0,
        "syntax_match_score_sum": 0.0,
        "dataflow_match_score_sum": 0.0,
    }


def add_codebleu_score(bucket: dict[str, float], score: dict[str, Any]) -> None:
    bucket["n_supported"] += 1
    for key in ("codebleu", "ngram_match_score", "weighted_ngram_match_score", "syntax_match_score", "dataflow_match_score"):
        bucket[f"{key}_sum"] += float(score.get(key, 0.0))


def finalize_codebleu_bucket(bucket: dict[str, float]) -> dict[str, Any]:
    n_supported = int(bucket["n_supported"])
    out: dict[str, Any] = {
        "n_total": int(bucket["n_total"]),
        "n_supported": n_supported,
        "n_failed": int(bucket["n_failed"]),
    }
    for key in ("codebleu", "ngram_match_score", "weighted_ngram_match_score", "syntax_match_score", "dataflow_match_score"):
        out[key] = bucket[f"{key}_sum"] / n_supported if n_supported else 0.0
    return out


def evaluate_completion(
    rows: list[dict[str, Any]],
    eval_by_uid: dict[str, dict[str, Any]],
    pred_map: dict[str, list[str]],
    *,
    pass_k: int,
    compute_codebleu: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calc_codebleu = import_codebleu() if compute_codebleu else None
    cb_bucket = empty_codebleu_bucket()
    results = []
    n_with_target = 0
    n_missing_target = 0
    n_missing_pred = 0
    pass1 = 0
    passk = 0

    for row_index, ann_row in enumerate(rows):
        uid = str(ann_row.get("uid") or "")
        eval_row = eval_by_uid.get(uid)
        if eval_row is None or "target" not in eval_row:
            n_missing_target += 1
            results.append({"uid": uid, "row_index": row_index, "status": "missing_eval_target"})
            continue

        n_with_target += 1
        target = str(eval_row.get("target", ""))
        target_norm = normalize_completion(target)
        preds = pred_map.get(uid, [])[: max(1, int(pass_k))]
        if not preds:
            n_missing_pred += 1
            results.append({"uid": uid, "row_index": row_index, "status": "missing_prediction", "pass1": False, "passk": False})
            continue

        clean_preds = [sanitize_prediction(str(pred)) for pred in preds]
        norm_preds = [normalize_completion(pred) for pred in clean_preds]
        sample_pass1 = bool(norm_preds and norm_preds[0] == target_norm)
        sample_passk = any(pred == target_norm for pred in norm_preds[: max(1, int(pass_k))])
        pass1 += int(sample_pass1)
        passk += int(sample_passk)

        greedy_codebleu = None
        if calc_codebleu is not None:
            cb_bucket["n_total"] += 1
            try:
                gold_full_code = str(eval_row.get("full_code") or (str(eval_row.get("prefix", "")) + target + str(eval_row.get("suffix", ""))))
                pred_full_code = str(eval_row.get("prefix", "")) + clean_preds[0] + str(eval_row.get("suffix", ""))
                greedy_codebleu = calc_codebleu(
                    references=[gold_full_code],
                    predictions=[pred_full_code],
                    lang="go",
                )
                add_codebleu_score(cb_bucket, greedy_codebleu)
            except Exception as exc:
                cb_bucket["n_failed"] += 1
                greedy_codebleu = {"error": f"{type(exc).__name__}: {exc}"}

        results.append({
            "uid": uid,
            "row_index": row_index,
            "source_row_index": ann_row.get("annotation_meta", {}).get("source_row_index"),
            "target_kind": eval_row.get("target_kind"),
            "status": "ok",
            "pass1": sample_pass1,
            "passk": sample_passk,
            "num_predictions": len(preds),
            "greedy_prediction": clean_preds[0],
            "target": target,
            "greedy_codebleu": greedy_codebleu,
        })

    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "n_with_target": n_with_target,
        "n_missing_target": n_missing_target,
        "n_missing_prediction": n_missing_pred,
        "pass@1": pass1 / n_with_target if n_with_target else 0.0,
        f"pass@{pass_k}": passk / n_with_target if n_with_target else 0.0,
        "n_pass@1": pass1,
        f"n_pass@{pass_k}": passk,
        "pass_definition": "exact normalized completion match against CodeSearchNet target; no unit tests are available for this internal split.",
    }
    if compute_codebleu:
        summary["codebleu"] = finalize_codebleu_bucket(cb_bucket)
        summary["codebleu_scope"] = "full_code = prefix + completion + suffix; this keeps oracle CodeBLEU well-defined for Go."
    return results, summary


def dtype_from_name(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_model(path_or_repo: str, device: str, dtype_name: str, local_files_only: bool):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = dtype_from_name(dtype_name)
    path = Path(path_or_repo)
    if path.exists() and (path / "adapter_config.json").exists():
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(path_or_repo, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model = PeftModel.from_pretrained(model, path_or_repo, is_trainable=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path_or_repo,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    model.config.use_cache = False
    model.to(device)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def rank_metrics(scores: Any, annot_sources: set[int], q: int, k: int) -> dict[str, float | int]:
    import torch

    k_eff = min(max(1, int(k)), max(0, int(q)))
    if k_eff == 0 or not annot_sources:
        return {"hits": 0, "recall": 0.0, "precision": 0.0, "ap": 0.0}
    ranked = torch.argsort(scores[:q].detach().float().cpu(), descending=True).tolist()[:k_eff]
    hits = 0
    precision_sum = 0.0
    for rank, idx in enumerate(ranked, start=1):
        if int(idx) in annot_sources:
            hits += 1
            precision_sum += hits / rank
    return {
        "hits": int(hits),
        "recall": hits / max(1, len(annot_sources)),
        "precision": hits / max(1, k_eff),
        "ap": precision_sum / max(1, len(annot_sources)),
    }


def select_saliency_targets(
    row: dict[str, Any],
    *,
    target_scope: str,
    max_queries: int,
    min_annotation_sources: int,
) -> tuple[int | None, dict[int, list[dict[str, Any]]], list[int]]:
    input_ids = [int(x) for x in row.get("input_ids", [])]
    labels = [int(x) for x in row.get("label", row.get("labels", []))]
    if not input_ids or len(input_ids) != len(labels):
        return None, {}, []
    completion_start = first_completion_idx(labels)
    if completion_start is None:
        return None, {}, []
    edges_by_q = causal_edges_by_target(row, len(input_ids), completion_start, target_scope)
    targets = []
    for q, edges in edges_by_q.items():
        srcs = {int(edge["src"]) for edge in edges if int(edge["src"]) < int(q)}
        if len(srcs) >= min_annotation_sources and int(q) > 0:
            targets.append(int(q))
    targets.sort(key=lambda q: (len({int(e["src"]) for e in edges_by_q[q]}), q), reverse=True)
    if max_queries > 0:
        targets = targets[:max_queries]
    return completion_start, edges_by_q, targets


def evaluate_saliency_alignment(
    rows: list[dict[str, Any]],
    *,
    model_name_or_path: str,
    device: str,
    dtype: str,
    local_files_only: bool,
    top_k: int,
    target_scope: str,
    max_samples: int,
    max_queries_per_sample: int,
    min_annotation_sources: int,
    source_chunk_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from src.train.loss import build_contribution_rows

    model = load_model(model_name_or_path, device, dtype, local_files_only)
    saliency_results = []
    recalls = []
    precisions = []
    aps = []
    n_queries = 0
    n_skipped = 0
    chosen_rows = rows[: max_samples if max_samples > 0 else len(rows)]

    for row_index, row in enumerate(chosen_rows):
        input_ids = [int(x) for x in row.get("input_ids", [])]
        _, edges_by_q, target_qs = select_saliency_targets(
            row,
            target_scope=target_scope,
            max_queries=max_queries_per_sample,
            min_annotation_sources=min_annotation_sources,
        )
        if not target_qs:
            n_skipped += 1
            continue

        input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_t, device=device)
        row_batch = torch.zeros(len(target_qs), dtype=torch.long, device=device)
        row_qry = torch.tensor(target_qs, dtype=torch.long, device=device)
        with torch.inference_mode():
            outputs = model(
                input_ids=input_t,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            c_rows = build_contribution_rows(
                model,
                outputs.hidden_states[-2],
                outputs.attentions[-1],
                row_batch,
                row_qry,
                source_chunk_size=max(1, int(source_chunk_size)),
            ).detach().cpu()

        per_query = []
        for row_id, q in enumerate(target_qs):
            annot_sources = {int(edge["src"]) for edge in edges_by_q.get(int(q), []) if int(edge["src"]) < int(q)}
            metrics = rank_metrics(c_rows[row_id], annot_sources, int(q), top_k)
            recalls.append(float(metrics["recall"]))
            precisions.append(float(metrics["precision"]))
            aps.append(float(metrics["ap"]))
            n_queries += 1
            per_query.append({
                "query": int(q),
                "num_annotation_sources": len(annot_sources),
                f"recall@{top_k}": metrics["recall"],
                f"precision@{top_k}": metrics["precision"],
                f"mAP@{top_k}": metrics["ap"],
            })

        saliency_results.append({
            "uid": row.get("uid"),
            "row_index": row_index,
            "source_row_index": row.get("annotation_meta", {}).get("source_row_index"),
            "num_queries": len(per_query),
            "queries": per_query,
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "model_name_or_path": model_name_or_path,
        "target_scope": target_scope,
        "top_k": int(top_k),
        "n_samples_input": len(rows),
        "n_samples_evaluated": len(saliency_results),
        "n_samples_skipped": n_skipped,
        "n_queries": n_queries,
        f"recall@{top_k}": mean(recalls) if recalls else 0.0,
        f"precision@{top_k}": mean(precisions) if precisions else 0.0,
        f"mAP@{top_k}": mean(aps) if aps else 0.0,
        "metric_definition": "For each annotated query token, rank causal source tokens s<q by teacher-forcing saliency; AP is normalized by all annotation sources.",
    }
    return saliency_results, summary


def markdown_table(summary: dict[str, Any], pass_k: int, top_k: int) -> str:
    comp = summary.get("completion", {})
    cb = comp.get("codebleu", {})
    sal = summary.get("saliency_alignment")
    sal_cells = ["NA", "NA", "NA"]
    if isinstance(sal, dict):
        sal_cells = [
            f"{float(sal.get(f'recall@{top_k}', 0.0)):.4f}",
            f"{float(sal.get(f'precision@{top_k}', 0.0)):.4f}",
            f"{float(sal.get(f'mAP@{top_k}', 0.0)):.4f}",
        ]
    row = [
        summary.get("run_name", "eval"),
        str(comp.get("n_with_target", 0)),
        f"{float(comp.get('pass@1', 0.0)):.4f}",
        f"{float(comp.get(f'pass@{pass_k}', 0.0)):.4f}",
        f"{float(cb.get('codebleu', 0.0)):.4f}" if cb else "NA",
        *sal_cells,
    ]
    return "\n".join([
        f"| run | n | pass@1 | pass@{pass_k} | codebleu | recall@{top_k} | precision@{top_k} | mAP@{top_k} |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| " + " | ".join(row) + " |",
        "",
    ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotated-data", type=Path, default=DEFAULT_ANNOTATED)
    p.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--predictions", type=Path, default=None, help="JSONL with uid and prediction/predictions/raw_generation.")
    p.add_argument("--oracle", action="store_true", help="Use the gold target as the prediction.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--table-md", type=Path, default=DEFAULT_TABLE)
    p.add_argument("--run-name", default="codesearchnet_go_internal")
    p.add_argument("--pass-k", type=int, default=10)
    p.add_argument("--max-samples", type=int, default=0, help="Limit rows for quick checks; 0 means all annotated rows.")
    p.add_argument("--compute-codebleu", action="store_true")
    p.add_argument("--skip-saliency", action="store_true")
    p.add_argument("--model_name_or_path", default="", help="Model/adapter path for teacher-forcing saliency alignment.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--local_files_only", type=int, default=1)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--target-scope", choices=["completion", "prompt", "all"], default="completion")
    p.add_argument("--max-saliency-samples", type=int, default=0, help="Limit saliency rows only; 0 follows --max-samples/all.")
    p.add_argument("--max-saliency-queries-per-sample", type=int, default=32)
    p.add_argument("--min-annotation-sources", type=int, default=1)
    p.add_argument("--source-chunk-size", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    annotated_path, failures = resolve_annotated_path(args.annotated_data)
    rows = iter_jsonl(annotated_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    eval_rows = iter_jsonl(args.eval_data)
    eval_by_uid = {str(row.get("uid")): row for row in eval_rows if row.get("uid")}

    if args.oracle:
        pred_map = {
            str(row["uid"]): [str(eval_by_uid[str(row["uid"])].get("target", ""))]
            for row in rows
            if row.get("uid") and str(row.get("uid")) in eval_by_uid
        }
    elif args.predictions is not None:
        pred_map = load_predictions(args.predictions)
    else:
        raise ValueError("Provide --predictions or use --oracle.")

    completion_results, completion_summary = evaluate_completion(
        rows,
        eval_by_uid,
        pred_map,
        pass_k=int(args.pass_k),
        compute_codebleu=bool(args.compute_codebleu),
    )

    saliency_results = None
    saliency_summary = None
    if not args.skip_saliency:
        if not args.model_name_or_path:
            raise ValueError("Saliency alignment requires --model_name_or_path, or pass --skip-saliency.")
        max_saliency_samples = int(args.max_saliency_samples or args.max_samples or 0)
        saliency_results, saliency_summary = evaluate_saliency_alignment(
            rows,
            model_name_or_path=str(args.model_name_or_path),
            device=str(args.device),
            dtype=str(args.dtype),
            local_files_only=bool(args.local_files_only),
            top_k=int(args.top_k),
            target_scope=str(args.target_scope),
            max_samples=max_saliency_samples,
            max_queries_per_sample=int(args.max_saliency_queries_per_sample),
            min_annotation_sources=int(args.min_annotation_sources),
            source_chunk_size=int(args.source_chunk_size),
        )

    result_by_uid = {str(row.get("uid")): row for row in completion_results}
    if saliency_results is not None:
        for sal_row in saliency_results:
            uid = str(sal_row.get("uid"))
            result_by_uid.setdefault(uid, {"uid": uid})["saliency_alignment"] = sal_row
    write_jsonl(args.output, list(result_by_uid.values()))

    summary: dict[str, Any] = {
        "run_name": args.run_name,
        "annotated_data": str(annotated_path),
        "requested_annotated_data": str(args.annotated_data),
        "eval_data": str(args.eval_data),
        "n_annotation_failures_file": len(failures),
        "completion": completion_summary,
        "saliency_alignment": saliency_summary,
    }
    write_json(args.summary, summary)
    args.table_md.parent.mkdir(parents=True, exist_ok=True)
    args.table_md.write_text(markdown_table(summary, int(args.pass_k), int(args.top_k)), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
