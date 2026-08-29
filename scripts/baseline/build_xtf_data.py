#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baseline.common import load_score_rows, read_jsonl, write_jsonl
from src.baseline.xtf import apply_xtf_from_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply XTF-style token masking to compact SFT data.")
    parser.add_argument("--input", required=True, help="Input compact JSONL with input_ids + label/labels.")
    parser.add_argument("--scores", required=True, help="JSONL with uid/task_id/id and ri_scores/pcp_probs/tr_scores.")
    parser.add_argument("--output", required=True, help="Output cleaned compact JSONL.")
    parser.add_argument("--report", default="", help="Optional report JSON path.")
    parser.add_argument("--pcp-threshold", type=float, default=0.95)
    parser.add_argument("--tr-percentile", type=float, default=10.0)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    samples = read_jsonl(args.input, max_rows=args.max_rows)
    score_rows = load_score_rows(args.scores)
    cleaned, report = apply_xtf_from_scores(
        samples,
        score_rows,
        pcp_threshold=args.pcp_threshold,
        tr_percentile=args.tr_percentile,
    )
    n = write_jsonl(args.output, cleaned)

    report.update({"input": args.input, "scores": args.scores, "output": args.output, "output_rows": n})
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

