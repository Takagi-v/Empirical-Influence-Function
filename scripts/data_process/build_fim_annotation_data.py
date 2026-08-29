#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_process.pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build canonical/ChatML FIM data and compact annotation data from raw JSONL."
    )
    p.add_argument("--input", required=True, type=Path, help="Raw/canonical/ChatML JSONL input path.")
    p.add_argument("--output-dir", required=True, type=Path, help="Directory for canonical/chatml/compact/report outputs.")
    p.add_argument("--model-path", required=True, help="Tokenizer/base model path used for Qwen token mapping.")
    p.add_argument("--annotate-model", required=True, help="OpenAI-compatible annotation model name or Huawei model id.")
    p.add_argument("--language", default="auto", help="auto, go, java, python, ...")
    p.add_argument("--source-dataset", default="auto")
    p.add_argument("--run-name", default="fim_annotation")
    p.add_argument("--max-rows", type=int, default=0, help="Read at most N raw rows; 0 means all.")
    p.add_argument("--max-accepted-rows", type=int, default=0, help="Stop after N accepted rows; 0 means all.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--model-max-length", type=int, default=4096)
    p.add_argument("--max-teacher-edges", type=int, default=64)
    p.add_argument("--structural-only", action="store_true", help="Do not call external LLM; keep deterministic structural edges only.")
    p.add_argument("--skip-annotation", action="store_true", help="Only write canonical/chatml/report; do not annotate.")
    p.add_argument("--force", action="store_true", help="Ignore existing annotation cache for this run.")
    p.add_argument("--no-strip-cjk-comments", action="store_true")
    p.add_argument("--max-target-nonempty-lines", type=int, default=10)
    p.add_argument("--max-target-rough-tokens", type=int, default=192)
    p.add_argument("--max-target-chars", type=int, default=1024)
    p.add_argument("--flush-every", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(PipelineConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        model_path=args.model_path,
        annotate_model=args.annotate_model,
        language=args.language,
        source_dataset=args.source_dataset,
        run_name=args.run_name,
        max_rows=args.max_rows,
        max_accepted_rows=args.max_accepted_rows,
        num_workers=args.num_workers,
        model_max_length=args.model_max_length,
        max_teacher_edges=args.max_teacher_edges,
        use_llm=not args.structural_only,
        skip_annotation=args.skip_annotation,
        force=args.force,
        strip_cjk_comments=not args.no_strip_cjk_comments,
        max_target_nonempty_lines=args.max_target_nonempty_lines,
        max_target_rough_tokens=args.max_target_rough_tokens,
        max_target_chars=args.max_target_chars,
        flush_every=args.flush_every,
    ))
    print(f"canonical={result.canonical_path}")
    print(f"chatml={result.chatml_path}")
    print(f"compact={result.compact_path}")
    print(f"report={result.report_path}")
    print(f"accepted={result.accepted} rejected={result.rejected} annotated={result.annotated}")


if __name__ == "__main__":
    main()
