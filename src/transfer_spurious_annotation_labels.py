from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


KEY_FIELDS = (
    "test_sample_index",
    "target_token_index",
    "target_token",
    "source_token_index",
    "source_token",
)
LABEL_FIELDS = (
    "human_label",
    "human_confidence",
    "human_rationale",
)
TRANSFER_FIELD = "transferred_from_item_id"


def _read_tsv(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple((row.get(field) or "").strip() for field in KEY_FIELDS)


def _has_label(row: dict[str, str]) -> bool:
    return bool((row.get("human_label") or "").strip())


def transfer_labels(
    old_annotation: str,
    new_annotation: str,
    output: str,
) -> dict[str, int | str]:
    old_rows = _read_tsv(old_annotation)
    new_rows = _read_tsv(new_annotation)

    old_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    duplicate_labeled_keys = 0
    old_labeled_rows = 0
    for row in old_rows:
        if not _has_label(row):
            continue
        old_labeled_rows += 1
        key = _key(row)
        if key in old_by_key:
            duplicate_labeled_keys += 1
            continue
        old_by_key[key] = row

    transferred = 0
    for row in new_rows:
        old_row = old_by_key.get(_key(row))
        if old_row is None:
            row[TRANSFER_FIELD] = ""
            continue
        for field in LABEL_FIELDS:
            row[field] = old_row.get(field, "")
        row[TRANSFER_FIELD] = old_row.get("item_id", "")
        transferred += 1

    fieldnames = list(new_rows[0].keys()) if new_rows else []
    if TRANSFER_FIELD not in fieldnames:
        fieldnames.append(TRANSFER_FIELD)
    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(new_rows)

    return {
        "old_annotation": str(Path(old_annotation).resolve()),
        "new_annotation": str(Path(new_annotation).resolve()),
        "output": str(Path(output).resolve()),
        "old_rows": len(old_rows),
        "old_labeled_rows": old_labeled_rows,
        "new_rows": len(new_rows),
        "transferred_rows": transferred,
        "duplicate_labeled_keys": duplicate_labeled_keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer reusable human labels from an old spurious-correlation TSV to a new TSV."
    )
    parser.add_argument("--old-annotation", required=True)
    parser.add_argument(
        "--new-annotation",
        default="spurious_correlation_results/human_annotation/human_annotation_top10.tsv",
    )
    parser.add_argument(
        "--output",
        default="spurious_correlation_results/human_annotation/human_annotation_top10_transferred.tsv",
    )
    args = parser.parse_args()

    stats = transfer_labels(args.old_annotation, args.new_annotation, args.output)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
