#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The Go tarball install path is not always visible to non-interactive shells.
_go_bin = Path("/usr/local/go/bin")
if _go_bin.exists():
    os.environ["PATH"] = f"{_go_bin}:{os.environ.get('PATH', '')}"

from scripts.benchmark.eval_judges import judge_mceval, sanitize_prediction  # noqa: E402
from scripts.go_singleline_fim_exp.evaluate_go_single_predictions import (  # noqa: E402
    _add_codebleu_score,
    _codebleu_lang,
    _empty_codebleu_bucket,
    _finalize_codebleu_bucket,
    _import_codebleu,
)
from scripts.go_singleline_fim_exp.go_single_pipeline import iter_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle-evaluate GoSingle MCEval-derived eval data with gold targets.")
    p.add_argument("--eval-data", type=Path, default=Path("data/go_single/eval_data/mceval_go_single_v2_canonical.jsonl"))
    p.add_argument("--output", type=Path, default=Path("outputs/go_singleline_fim_exp/eval_results/oracle/mceval_go_single_v2_oracle_results.jsonl"))
    p.add_argument("--summary", type=Path, default=Path("outputs/go_singleline_fim_exp/eval_results/oracle/mceval_go_single_v2_oracle_summary.json"))
    p.add_argument("--table-md", type=Path, default=Path("outputs/go_singleline_fim_exp/eval_results/oracle/mceval_go_single_v2_oracle_table.md"))
    p.add_argument("--table-csv", type=Path, default=Path("outputs/go_singleline_fim_exp/eval_results/oracle/mceval_go_single_v2_oracle_table.csv"))
    p.add_argument("--timeout-sec", type=int, default=10)
    p.add_argument("--compute-codebleu", action="store_true", help="Compute CodeBLEU(target, target); requires codebleu package.")
    return p.parse_args()


def fmt_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def make_table_rows(summary: dict[str, Any], by_kind: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overall = {
        "Group": "overall",
        "N": summary["n_total"],
        "N_supported": summary["n_supported"],
        "N_unsupported": summary["n_unsupported"],
        "Pass@1": fmt_float(summary["pass1"]),
        "CodeBLEU": fmt_float(summary.get("codebleu", {}).get("codebleu") if summary.get("codebleu") else None),
    }
    rows.append(overall)
    for kind, obj in sorted(by_kind.items()):
        cb = obj.get("codebleu") or {}
        rows.append({
            "Group": kind,
            "N": obj["n_total"],
            "N_supported": obj["n_supported"],
            "N_unsupported": obj["n_unsupported"],
            "Pass@1": fmt_float(obj["pass1"]),
            "CodeBLEU": fmt_float(cb.get("codebleu") if cb else None),
        })
    return rows


def write_table_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Group", "N", "N_supported", "N_unsupported", "Pass@1", "CodeBLEU"]
    lines = ["# GoSingle MCEval Oracle", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Group", "N", "N_supported", "N_unsupported", "Pass@1", "CodeBLEU"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    calc_codebleu = _import_codebleu() if args.compute_codebleu else None
    cb_overall = _empty_codebleu_bucket()
    cb_by_kind: dict[str, dict[str, float]] = defaultdict(_empty_codebleu_bucket)

    results: list[dict[str, Any]] = []
    total = 0
    supported = 0
    unsupported = 0
    passed = 0
    by_kind_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in iter_jsonl(args.eval_data):
        total += 1
        uid = str(row.get("uid"))
        kind = str(row.get("target_kind") or "unknown")
        target = sanitize_prediction(str(row.get("target", ""))).strip("\n")
        judge_payload = row.get("judge_payload", {}) or {}
        if judge_payload.get("kind") != "mceval_go_test":
            unsupported += 1
            by_kind_counts[kind]["unsupported"] += 1
            results.append({"uid": uid, "target_kind": kind, "pass": None, "status": "unsupported_judge"})
            continue

        judge_prefix = str(judge_payload.get("judge_prefix", row.get("prefix", "")))
        judge_suffix = str(judge_payload.get("judge_suffix", row.get("suffix", "")))
        full_code = judge_prefix + target + judge_suffix
        ok, status, err = judge_mceval("go", full_code, row, timeout_sec=args.timeout_sec)

        if ok is None:
            unsupported += 1
            by_kind_counts[kind]["unsupported"] += 1
        else:
            supported += 1
            by_kind_counts[kind]["supported"] += 1
            if ok:
                passed += 1
                by_kind_counts[kind]["passed"] += 1

        oracle_codebleu = None
        if calc_codebleu is not None:
            cb_overall["n_total"] += 1
            cb_by_kind[kind]["n_total"] += 1
            try:
                oracle_codebleu = calc_codebleu(
                    references=[full_code],
                    predictions=[full_code],
                    lang=_codebleu_lang("go"),
                )
                _add_codebleu_score(cb_overall, oracle_codebleu)
                _add_codebleu_score(cb_by_kind[kind], oracle_codebleu)
            except Exception as exc:
                cb_overall["n_failed"] += 1
                cb_by_kind[kind]["n_failed"] += 1
                oracle_codebleu = {"error": f"{type(exc).__name__}: {exc}"}

        results.append({
            "uid": uid,
            "target_kind": kind,
            "target": target,
            "pass": ok,
            "status": status,
            "error_summary": err,
            "oracle_codebleu": oracle_codebleu,
        })

    write_jsonl(args.output, results)

    summary: dict[str, Any] = {
        "eval_data": str(args.eval_data),
        "n_total": total,
        "n_supported": supported,
        "n_unsupported": unsupported,
        "n_passed": passed,
        "pass1": passed / supported if supported else 0.0,
    }
    if args.compute_codebleu:
        summary["codebleu"] = _finalize_codebleu_bucket(cb_overall)

    by_kind_summary: dict[str, dict[str, Any]] = {}
    for kind, counts in sorted(by_kind_counts.items()):
        kind_supported = counts["supported"]
        kind_passed = counts["passed"]
        obj: dict[str, Any] = {
            "n_total": kind_supported + counts["unsupported"],
            "n_supported": kind_supported,
            "n_unsupported": counts["unsupported"],
            "n_passed": kind_passed,
            "pass1": kind_passed / kind_supported if kind_supported else 0.0,
        }
        if args.compute_codebleu:
            obj["codebleu"] = _finalize_codebleu_bucket(cb_by_kind[kind])
        by_kind_summary[kind] = obj
    summary["by_target_kind"] = by_kind_summary

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    table_rows = make_table_rows(summary, by_kind_summary)
    write_table_md(args.table_md, table_rows)
    write_table_csv(args.table_csv, table_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[oracle] results: {args.output}")
    print(f"[oracle] summary: {args.summary}")
    print(f"[oracle] table_md: {args.table_md}")
    print(f"[oracle] table_csv: {args.table_csv}")


if __name__ == "__main__":
    main()
