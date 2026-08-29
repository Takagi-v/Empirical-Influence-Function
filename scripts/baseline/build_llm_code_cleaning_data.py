#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baseline.common import load_score_rows, read_jsonl, write_jsonl
from src.baseline.llm_code_cleaning import apply_llm_cleaning_rewrites


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply precomputed LLM-assisted code-cleaning rewrites. "
            "The expensive LLM rewriting step should be generated separately by the benchmark owner."
        )
    )
    parser.add_argument("--input", required=True, help="Input compact/canonical JSONL.")
    parser.add_argument("--rewrites", required=True, help="JSONL with uid/task_id/id and cleaned_response.")
    parser.add_argument("--output", required=True, help="Output JSONL with rewritten response fields.")
    parser.add_argument("--report", default="", help="Optional report JSON path.")
    parser.add_argument("--rewrite-key", default="cleaned_response")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    samples = read_jsonl(args.input, max_rows=args.max_rows)
    rewrite_rows = load_score_rows(args.rewrites)
    cleaned, report = apply_llm_cleaning_rewrites(samples, rewrite_rows, rewrite_key=args.rewrite_key)
    n = write_jsonl(args.output, cleaned)

    report.update({"input": args.input, "rewrites": args.rewrites, "output": args.output, "output_rows": n})
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

