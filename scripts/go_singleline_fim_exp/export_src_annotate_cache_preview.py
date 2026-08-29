#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.go_singleline_fim_exp.annotate_chatml_with_src_annotate import load_jsonl, load_seed_compact, row_key


def load_annotation_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return cache
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get("key")
            record = obj.get("record")
            if key and isinstance(record, dict):
                cache[str(key)] = record
    return cache


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} rows to {p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export completed src/annotate cache rows as aligned raw + compact preview files."
    )
    parser.add_argument("--input_path", required=True, help="Original raw/chatml JSONL used by annotation.")
    parser.add_argument("--annotation_cache_path", required=True, help="*.annotation_cache.jsonl from annotation run.")
    parser.add_argument("--output_path", required=True, help="Preview compact JSONL output.")
    parser.add_argument("--raw_output_path", required=True, help="Raw JSONL rows aligned with the preview compact output.")
    parser.add_argument("--seed_compact_path", default="", help="Optional existing compact file to reuse by uid.")
    parser.add_argument("--max_rows", type=int, default=0, help="Limit exported rows; 0 means all cached/seeded rows.")
    args = parser.parse_args()

    raw_rows = load_jsonl(args.input_path)
    cache = load_annotation_cache(args.annotation_cache_path)
    seed = load_seed_compact(args.seed_compact_path) if args.seed_compact_path else {}
    print(f"Loaded annotation cache: {len(cache)} entries from {args.annotation_cache_path}")

    compact_rows: list[dict[str, Any]] = []
    aligned_raw_rows: list[dict[str, Any]] = []
    cache_hits = 0
    seed_hits = 0
    missing_seen = 0

    for row in raw_rows:
        key = row_key(row)
        record = None
        if key in cache:
            record = copy.deepcopy(cache[key])
            cache_hits += 1
        elif key in seed:
            record = copy.deepcopy(seed[key])
            seed_hits += 1
        else:
            missing_seen += 1
            continue

        compact_rows.append(record)
        aligned_raw_rows.append(row)
        if args.max_rows > 0 and len(compact_rows) >= args.max_rows:
            break

    write_jsonl(compact_rows, args.output_path)
    write_jsonl(aligned_raw_rows, args.raw_output_path)
    print(
        "Preview export summary: "
        f"output={len(compact_rows)} cache_hits={cache_hits} seed_hits={seed_hits} missing_seen={missing_seen}"
    )


if __name__ == "__main__":
    main()
