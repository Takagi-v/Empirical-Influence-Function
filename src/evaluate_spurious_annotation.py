from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_tsv(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _normalise_human_label(raw: str) -> str:
    value = (raw or "").strip().upper()
    if value in {"S", "SPURIOUS", "虚假", "虚假相关性"}:
        return "spurious"
    if value in {"N", "SUPPORTED", "NON-SPURIOUS", "合理", "非虚假", "非虚假相关性"}:
        return "supported"
    if value in {"U", "UNCERTAIN", "不确定"}:
        return "uncertain"
    return "missing"


def _safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _metrics(rows: list[tuple[str, str]]) -> dict[str, object]:
    tp = sum(1 for human, auto in rows if human == "spurious" and auto == "spurious")
    fp = sum(1 for human, auto in rows if human == "supported" and auto == "spurious")
    fn = sum(1 for human, auto in rows if human == "spurious" and auto == "supported")
    tn = sum(1 for human, auto in rows if human == "supported" and auto == "supported")

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    jaccard = _safe_div(tp, tp + fp + fn)
    accuracy = _safe_div(tp + tn, tp + fp + fn + tn)

    return {
        "evaluated_items": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "accuracy": accuracy,
    }


def evaluate(annotation_path: str, key_path: str) -> dict[str, object]:
    annotation_rows = _read_tsv(annotation_path)
    key_rows = {row["item_id"]: row for row in _read_tsv(key_path)}

    strict_pairs: list[tuple[str, str]] = []
    loose_pairs: list[tuple[str, str]] = []
    skipped_uncertain = 0
    skipped_missing = 0
    missing_key = 0

    for row in annotation_rows:
        item_id = row.get("item_id", "")
        key = key_rows.get(item_id)
        if key is None:
            missing_key += 1
            continue

        human = _normalise_human_label(row.get("human_label", ""))
        if human == "uncertain":
            skipped_uncertain += 1
            continue
        if human == "missing":
            skipped_missing += 1
            continue

        strict_pairs.append((human, key["auto_strict_label"]))
        loose_pairs.append((human, key["auto_loose_label"]))

    return {
        "annotation_path": str(Path(annotation_path).resolve()),
        "key_path": str(Path(key_path).resolve()),
        "total_annotation_rows": len(annotation_rows),
        "skipped_uncertain": skipped_uncertain,
        "skipped_missing": skipped_missing,
        "missing_key": missing_key,
        "strict": _metrics(strict_pairs),
        "loose": _metrics(loose_pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate automatic spurious-correlation labels against human annotation."
    )
    parser.add_argument(
        "--annotation",
        default="spurious_correlation_results/human_annotation/human_annotation_top10.tsv",
    )
    parser.add_argument(
        "--key",
        default="spurious_correlation_results/human_annotation/human_annotation_top10_auto_key.tsv",
    )
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    result = evaluate(args.annotation, args.key)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
