#!/usr/bin/env python3
"""Select one annotated training sample and one target token for saliency overfit debugging.

The selector mirrors the current training loss convention: each annotation edge
is converted to an effective causal edge min(src, dst) -> max(src, dst).  It then
looks for completion target tokens with a moderate number of effective incoming
annotation sources.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/interim/benchmark_legacy_fim/sft_data/ours_graphsignal_train_first500.json"
DEFAULT_OUTPUT = ROOT / "outputs/saliency_exp/selected_single_target.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=str(DEFAULT_DATA))
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--max_rows", type=int, default=500)
    parser.add_argument("--min_total_edges", type=int, default=30)
    parser.add_argument("--max_total_edges", type=int, default=180)
    parser.add_argument("--min_in_edges", type=int, default=5)
    parser.add_argument("--max_in_edges", type=int, default=18)
    parser.add_argument("--target_in_edges", type=int, default=8)
    parser.add_argument("--target_total_edges", type=int, default=90)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--allow_whitespace_target", action="store_true")
    parser.add_argument("--local_files_only", type=int, default=1)
    return parser.parse_args()


def read_jsonl(path: Path, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def token_display(text: str) -> str:
    if text == "\n":
        return "\\n"
    if text == "\t":
        return "\\t"
    return text.replace("\n", "\\n").replace("\t", "\\t")


def is_special(text: str) -> bool:
    return text.startswith("<|") and text.endswith("|>")


def is_punctuationish(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and re.fullmatch(r"[\W_]+", stripped) is not None


def token_quality(text: str, *, allow_whitespace: bool) -> float:
    stripped = text.strip()
    if is_special(text):
        return -3.0
    if not stripped:
        return -2.0 if not allow_whitespace else -0.3
    if is_punctuationish(text):
        return -0.7
    if re.match(r"[A-Za-z_]", stripped):
        return 1.0
    if re.search(r"[A-Za-z_]", stripped):
        return 0.05
    return 0.2


def first_label_index(labels: list[int]) -> int | None:
    return next((i for i, value in enumerate(labels) if int(value) != IGNORE_INDEX), None)


def normalize_edge(edge: dict[str, Any]) -> tuple[int, int, str]:
    src = int(edge.get("src", edge.get("source", -1)))
    dst = int(edge.get("dst", edge.get("target", -1)))
    subtype = str(edge.get("subtype", edge.get("type", "")))
    return src, dst, subtype


def decode_tokens(tokenizer: Any, input_ids: list[int]) -> list[dict[str, Any]]:
    out = []
    for idx, tid in enumerate(input_ids):
        text = tokenizer.decode([int(tid)], skip_special_tokens=False)
        out.append({
            "idx": idx,
            "id": int(tid),
            "text": text,
            "display": token_display(text),
            "is_special": is_special(text),
            "is_whitespace": text.strip() == "",
        })
    return out


def source_payload(src_idx: int, rows: list[dict[str, Any]], tokens: list[dict[str, Any]]) -> dict[str, Any]:
    tok = tokens[src_idx] if 0 <= src_idx < len(tokens) else {}
    subtypes = Counter(row["subtype"] for row in rows)
    return {
        "idx": src_idx,
        "token": tok.get("text", ""),
        "display": tok.get("display", str(src_idx)),
        "num_edges": len(rows),
        "subtypes": dict(subtypes),
        "raw_edges": [
            {
                "src": row["raw_src"],
                "dst": row["raw_dst"],
                "effective_src": row["effective_src"],
                "effective_tgt": row["effective_tgt"],
                "subtype": row["subtype"],
            }
            for row in rows[:8]
        ],
    }


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    data_path = Path(args.data_path)
    rows = read_jsonl(data_path, args.max_rows)
    if not rows:
        raise SystemExit(f"No rows read from {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )

    candidates: list[dict[str, Any]] = []
    skipped = Counter()
    for sample_index, row in enumerate(rows):
        input_ids = [int(x) for x in row.get("input_ids", [])]
        labels = [int(x) for x in row.get("label", row.get("labels", []))]
        edges = list(row.get("attention_edges", []))
        if not input_ids or not labels or not edges:
            skipped["missing_fields"] += 1
            continue
        if len(input_ids) != len(labels):
            skipped["length_mismatch"] += 1
            continue
        total_edges = len(edges)
        if not (args.min_total_edges <= total_edges <= args.max_total_edges):
            skipped["total_edges_out_of_range"] += 1
            continue

        completion_start = first_label_index(labels)
        if completion_start is None:
            skipped["no_completion"] += 1
            continue

        tokens = decode_tokens(tokenizer, input_ids)
        by_tgt_src: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        raw_incoming: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            raw_src, raw_dst, subtype = normalize_edge(edge)
            if not (0 <= raw_src < len(input_ids) and 0 <= raw_dst < len(input_ids)):
                continue
            effective_src = min(raw_src, raw_dst)
            effective_tgt = max(raw_src, raw_dst)
            payload = {
                "raw_src": raw_src,
                "raw_dst": raw_dst,
                "effective_src": effective_src,
                "effective_tgt": effective_tgt,
                "subtype": subtype,
            }
            by_tgt_src[effective_tgt][effective_src].append(payload)
            if raw_dst == effective_tgt:
                raw_incoming[effective_tgt].append(payload)

        for target_idx, src_map in by_tgt_src.items():
            if target_idx < completion_start:
                continue
            if target_idx >= len(tokens):
                continue
            src_count = len(src_map)
            if not (args.min_in_edges <= src_count <= args.max_in_edges):
                continue
            target_tok = tokens[target_idx]
            quality = token_quality(target_tok["text"], allow_whitespace=args.allow_whitespace_target)
            if quality < -1.0:
                continue
            raw_in = len(raw_incoming.get(target_idx, []))
            direction_flip_count = sum(
                1 for rows_for_src in src_map.values() for item in rows_for_src
                if item["raw_dst"] != item["effective_tgt"]
            )
            subtype_counts = Counter(
                item["subtype"]
                for rows_for_src in src_map.values()
                for item in rows_for_src
            )
            source_rows = [
                source_payload(src_idx, edge_rows, tokens)
                for src_idx, edge_rows in sorted(src_map.items(), key=lambda kv: kv[0])
            ]
            score = (
                quality
                - abs(src_count - args.target_in_edges) * 0.12
                - abs(total_edges - args.target_total_edges) * 0.006
                + min(raw_in, src_count) * 0.03
                - direction_flip_count * 0.015
            )
            candidates.append({
                "score": score,
                "sample_index": sample_index,
                "target_idx": target_idx,
                "target_token": target_tok,
                "completion_start": completion_start,
                "num_tokens": len(input_ids),
                "total_edges": total_edges,
                "effective_incoming_sources": src_count,
                "effective_incoming_edges": sum(len(v) for v in src_map.values()),
                "raw_incoming_edges": raw_in,
                "direction_flip_edges": direction_flip_count,
                "subtypes": dict(subtype_counts),
                "source_tokens": source_rows,
                "row_meta": {
                    key: row.get(key)
                    for key in ("uid", "raw_id", "language", "source_dataset", "task_id")
                    if key in row
                },
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected = candidates[0] if candidates else None
    payload = {
        "version": 1,
        "data_path": str(data_path),
        "selection_rule": {
            "effective_edge": "min(src,dst) -> max(src,dst), matching current compute_saliency_loss",
            "max_rows": args.max_rows,
            "total_edges": [args.min_total_edges, args.max_total_edges],
            "incoming_sources": [args.min_in_edges, args.max_in_edges],
        },
        "num_rows_scanned": len(rows),
        "num_candidates": len(candidates),
        "skipped": dict(skipped),
        "selected": selected,
        "candidates": candidates[: args.top_k],
    }
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rows scanned: {len(rows)}")
    print(f"candidates: {len(candidates)}")
    print(f"skipped: {dict(skipped)}")
    print(f"wrote {out}")
    if selected is None:
        print("No candidate found. Relax --min/--max edges constraints.")
        return

    print("\nSELECTED")
    print(
        f"sample={selected['sample_index']} target=#{selected['target_idx']} "
        f"token={selected['target_token']['display']!r} "
        f"score={selected['score']:.3f}"
    )
    print(
        f"completion_start={selected['completion_start']} total_edges={selected['total_edges']} "
        f"incoming_sources={selected['effective_incoming_sources']} "
        f"incoming_edges={selected['effective_incoming_edges']} "
        f"raw_incoming_edges={selected['raw_incoming_edges']} "
        f"flipped_edges={selected['direction_flip_edges']}"
    )
    print(f"subtypes={selected['subtypes']}")
    print("annotation sources:")
    for src in selected["source_tokens"]:
        print(f"  #{src['idx']:>4} {src['display']!r} edges={src['num_edges']} subtypes={src['subtypes']}")

    print("\nTOP CANDIDATES")
    for item in candidates[: min(args.top_k, 12)]:
        print(
            f"score={item['score']:.3f} sample={item['sample_index']} "
            f"target=#{item['target_idx']} token={item['target_token']['display']!r} "
            f"sources={item['effective_incoming_sources']} total_edges={item['total_edges']} "
            f"flipped={item['direction_flip_edges']}"
        )


if __name__ == "__main__":
    main()
