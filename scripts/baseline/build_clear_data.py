#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baseline.clear import apply_clear_scores
from src.baseline.common import load_score_rows, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply CLEAR-style filtering/correction from precomputed scores.")
    parser.add_argument("--input", required=True, help="Input compact/canonical JSONL.")
    parser.add_argument("--scores", required=True, help="JSONL with uid/task_id/id and confidence fields.")
    parser.add_argument("--output", required=True, help="Output curated JSONL.")
    parser.add_argument("--report", default="", help="Optional report JSON path.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Drop sample if confidence <= gamma.")
    parser.add_argument("--eta", type=float, default=0.8, help="Replace response if candidate_better_score > eta.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Confidence mix weight for observed consistency.")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    samples = read_jsonl(args.input, max_rows=args.max_rows)
    score_rows = load_score_rows(args.scores)
    cleaned, report = apply_clear_scores(samples, score_rows, gamma=args.gamma, eta=args.eta, alpha=args.alpha)
    n = write_jsonl(args.output, cleaned)

    report.update({"input": args.input, "scores": args.scores, "output": args.output, "output_rows": n})
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

