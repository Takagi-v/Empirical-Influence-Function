#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.go_singleline_fim_exp.go_single_pipeline import (  # noqa: E402
    BuildStats,
    build_codesearchnet_candidates,
    build_mceval_candidates,
    render_chatml,
    select_train_subset,
    write_jsonl,
    filter_by_function_lines,
    write_report,
    write_samples_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Go single-statement ChatML benchmarkV2 data.")
    p.add_argument("--codesearchnet-dir", type=Path, default=Path("/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/raw/codesearchnet/go/final/jsonl/unzip"))
    p.add_argument("--codesearchnet-glob", default="go_train_*.jsonl")
    p.add_argument("--mceval-root", type=Path, default=Path("/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/raw/mceval"))
    p.add_argument("--train-output", type=Path, default=Path("data/go_single/train_data/go_single_train_v2_canonical.jsonl"))
    p.add_argument("--train-chatml-output", type=Path, default=Path("data/go_single/train_data/go_single_train_v2_chatml.jsonl"))
    p.add_argument("--eval-output", type=Path, default=Path("data/go_single/eval_data/mceval_go_single_v2_canonical.jsonl"))
    p.add_argument("--eval-chatml-output", type=Path, default=Path("data/go_single/eval_data/mceval_go_single_v2_chatml.jsonl"))
    p.add_argument("--report", type=Path, default=Path("outputs/go_singleline_fim_exp/reports/go_single_v2_build_report.md"))
    p.add_argument("--samples-report", type=Path, default=Path("outputs/go_singleline_fim_exp/reports/go_single_v2_samples.md"))
    p.add_argument("--num-train", type=int, default=10000)
    p.add_argument("--max-train-rows", type=int, default=0, help="Preview limit for CodeSearchNet rows; 0 means all rows.")
    p.add_argument("--eval-per-task", type=int, default=3)
    p.add_argument("--sample-report-size", type=int, default=60)
    p.add_argument("--max-function-lines", type=int, default=0, help="Hard cap for function non-empty line count; 0 disables.")
    p.add_argument("--train-line-quantile", type=float, default=1.0, help="Keep train samples with function lines <= this quantile, e.g. 0.5 for median.")
    p.add_argument("--eval-line-quantile", type=float, default=1.0, help="Keep eval samples with function lines <= this quantile, e.g. 0.5 for median.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_paths = sorted(args.codesearchnet_dir.glob(args.codesearchnet_glob))
    if not train_paths:
        raise SystemExit(f"No CodeSearchNet files matched: {args.codesearchnet_dir}/{args.codesearchnet_glob}")

    train_stats = BuildStats()
    eval_stats = BuildStats()
    max_rows = args.max_train_rows if args.max_train_rows > 0 else None

    print(f"[train] reading {len(train_paths)} files from {args.codesearchnet_dir}")
    train_candidates = build_codesearchnet_candidates(train_paths, max_rows=max_rows, stats=train_stats)
    train_candidates, train_line_filter = filter_by_function_lines(
        train_candidates,
        max_lines=args.max_function_lines,
        quantile=args.train_line_quantile,
    )
    train_samples = select_train_subset(train_candidates, args.num_train, args.seed)
    train_chatml = [render_chatml(s) for s in train_samples]

    print(f"[eval] reading MCEval from {args.mceval_root}")
    eval_samples = build_mceval_candidates(args.mceval_root, per_task=args.eval_per_task, stats=eval_stats)
    eval_samples, eval_line_filter = filter_by_function_lines(
        eval_samples,
        max_lines=args.max_function_lines,
        quantile=args.eval_line_quantile,
    )
    eval_chatml = [render_chatml(s) for s in eval_samples]

    n_train = write_jsonl(args.train_output, train_samples)
    n_train_chatml = write_jsonl(args.train_chatml_output, train_chatml)
    n_eval = write_jsonl(args.eval_output, eval_samples)
    n_eval_chatml = write_jsonl(args.eval_chatml_output, eval_chatml)

    write_report(
        args.report,
        title="GoSingle BenchmarkV2 Build Report",
        train_stats=train_stats,
        eval_stats=eval_stats,
        train_samples=train_samples,
        eval_samples=eval_samples,
        train_line_filter=train_line_filter,
        eval_line_filter=eval_line_filter,
    )
    write_samples_report(
        args.samples_report,
        train_samples + eval_samples,
        limit=args.sample_report_size,
        seed=args.seed,
    )

    print(f"[done] train canonical: {n_train} -> {args.train_output}")
    print(f"[done] train chatml:    {n_train_chatml} -> {args.train_chatml_output}")
    print(f"[done] eval canonical:  {n_eval} -> {args.eval_output}")
    print(f"[done] eval chatml:     {n_eval_chatml} -> {args.eval_chatml_output}")
    print(f"[done] report:          {args.report}")
    print(f"[done] samples:         {args.samples_report}")


if __name__ == "__main__":
    main()
