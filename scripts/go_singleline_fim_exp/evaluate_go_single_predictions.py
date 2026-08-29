#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark.eval_judges import judge_mceval, sanitize_prediction  # noqa: E402
from scripts.go_singleline_fim_exp.go_single_pipeline import iter_jsonl, write_jsonl  # noqa: E402


def _import_codebleu():
    try:
        from codebleu import calc_codebleu  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CodeBLEU support requires the `codebleu` package. Install it in the eif-bench env, "
            "for example: `.micromamba/envs/eif-bench/bin/python -m pip install codebleu`."
        ) from exc
    return calc_codebleu


def _codebleu_lang(language: str) -> str | None:
    return {"go": "go"}.get(language.lower())


def _empty_codebleu_bucket() -> dict[str, float]:
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


def _add_codebleu_score(bucket: dict[str, float], score: dict[str, Any]) -> None:
    bucket["n_supported"] += 1
    for key in ("codebleu", "ngram_match_score", "weighted_ngram_match_score", "syntax_match_score", "dataflow_match_score"):
        bucket[f"{key}_sum"] += float(score.get(key, 0.0))


def _finalize_codebleu_bucket(bucket: dict[str, float]) -> dict[str, Any]:
    n_supported = int(bucket["n_supported"])
    out = {
        "n_total": int(bucket["n_total"]),
        "n_supported": n_supported,
        "n_failed": int(bucket["n_failed"]),
    }
    for key in ("codebleu", "ngram_match_score", "weighted_ngram_match_score", "syntax_match_score", "dataflow_match_score"):
        out[key] = bucket[f"{key}_sum"] / n_supported if n_supported else 0.0
    return out



def load_predictions(path: Path) -> dict[str, list[str]]:
    preds: dict[str, list[str]] = {}
    for row in iter_jsonl(path):
        uid = str(row.get("uid") or "")
        if not uid:
            continue
        values = row.get("predictions")
        if values is None:
            values = row.get("raw_generation")
        if values is None and "prediction" in row:
            values = [row.get("prediction")]
        if values is None:
            continue
        if isinstance(values, str):
            values = [values]
        out: list[str] = []
        for item in values:
            if isinstance(item, dict):
                text = item.get("text") or item.get("prediction") or item.get("content") or ""
            else:
                text = item
            if text is None:
                continue
            out.append(str(text))
        preds[uid] = out
    return preds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate GoSingle benchmarkV2 predictions with MCEval-derived Go tests.")
    p.add_argument("--eval-data", type=Path, default=Path("data/go_single/eval_data/mceval_go_single_v2_canonical.jsonl"))
    p.add_argument("--predictions", type=Path, required=True, help="JSONL with uid + prediction/predictions/raw_generation.")
    p.add_argument("--output", type=Path, default=Path("outputs/go_singleline_fim_exp/eval_results/go_single_eval_results.jsonl"))
    p.add_argument("--summary", type=Path, default=Path("outputs/go_singleline_fim_exp/eval_results/go_single_eval_summary.json"))
    p.add_argument("--timeout-sec", type=int, default=10)
    p.add_argument("--pass-k", type=int, default=10)
    p.add_argument("--compute-codebleu", action="store_true", help="Compute greedy CodeBLEU against target; requires codebleu package.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred_map = load_predictions(args.predictions)
    calc_codebleu = _import_codebleu() if args.compute_codebleu else None
    cb_bucket = _empty_codebleu_bucket()

    results: list[dict[str, Any]] = []
    n_total = 0
    n_missing = 0
    n_supported = 0
    n_unsupported = 0
    pass1 = 0
    passk = 0

    for row in iter_jsonl(args.eval_data):
        n_total += 1
        uid = str(row.get("uid"))
        preds = pred_map.get(uid, [])[: max(1, args.pass_k)]
        if not preds:
            n_missing += 1
            results.append({"uid": uid, "status": "missing_prediction", "pass1": False, "passk": False})
            continue

        judge_payload = row.get("judge_payload", {}) or {}
        if judge_payload.get("kind") != "mceval_go_test":
            results.append({"uid": uid, "status": "unsupported_judge", "pass1": None, "passk": None})
            continue
        judge_prefix_for_cb = str(judge_payload.get("judge_prefix", row["prefix"]))
        judge_suffix_for_cb = str(judge_payload.get("judge_suffix", row["suffix"]))
        gold_full_code_for_cb = judge_prefix_for_cb + str(row.get("target", "")) + judge_suffix_for_cb

        candidate_results: list[dict[str, Any]] = []
        sample_pass1 = False
        sample_passk = False
        for i, pred in enumerate(preds):
            clean_pred = sanitize_prediction(pred).strip("\n")
            judge_prefix = str(judge_payload.get("judge_prefix", row["prefix"]))
            judge_suffix = str(judge_payload.get("judge_suffix", row["suffix"]))
            full_code = judge_prefix + clean_pred + judge_suffix
            ok, status, err = judge_mceval("go", full_code, row, timeout_sec=args.timeout_sec)
            cand_obj = {
                "index": i,
                "prediction": clean_pred,
                "pass": ok,
                "status": status,
                "error_summary": err,
            }
            candidate_results.append(cand_obj)
            if i == 0 and ok is True:
                sample_pass1 = True
            if ok is True:
                sample_passk = True

        sample_unsupported = all(c.get("pass") is None for c in candidate_results)
        if sample_unsupported:
            n_unsupported += 1
        else:
            n_supported += 1
            if sample_pass1:
                pass1 += 1
            if sample_passk:
                passk += 1

        greedy_codebleu = None
        if calc_codebleu is not None:
            cb_bucket["n_total"] += 1
            cb_lang = _codebleu_lang("go")
            try:
                greedy_codebleu = calc_codebleu(
                    references=[gold_full_code_for_cb],
                    predictions=[judge_prefix_for_cb + candidate_results[0]["prediction"] + judge_suffix_for_cb],
                    lang=cb_lang,
                )
                _add_codebleu_score(cb_bucket, greedy_codebleu)
            except Exception as exc:
                cb_bucket["n_failed"] += 1
                greedy_codebleu = {"error": f"{type(exc).__name__}: {exc}"}

        results.append({
            "uid": uid,
            "target": row.get("target"),
            "target_kind": row.get("target_kind"),
            "pass1": None if sample_unsupported else sample_pass1,
            "passk": None if sample_unsupported else sample_passk,
            "candidates": candidate_results,
            "greedy_codebleu": greedy_codebleu,
        })

    write_jsonl(args.output, results)
    summary: dict[str, Any] = {
        "n_total": n_total,
        "n_missing_prediction": n_missing,
        "n_supported": n_supported,
        "n_unsupported": n_unsupported,
        "pass1": pass1 / n_supported if n_supported else 0.0,
        f"pass@{args.pass_k}": passk / n_supported if n_supported else 0.0,
        "n_pass1": pass1,
        "n_passk": passk,
    }
    if args.compute_codebleu:
        summary["codebleu"] = _finalize_codebleu_bucket(cb_bucket)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
