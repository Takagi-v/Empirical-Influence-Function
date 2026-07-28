from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


LANGUAGE_FILES = {
    "go": "data/go/go_single_train_v2_chatml.jsonl",
    "java": "data/java/java_single_train_codesearchnet_20000_chatml.jsonl",
    "javascript": "data/javascript/javascript_single_train_codesearchnet_20000_chatml.jsonl",
    "python": "data/python/python_single_train_codesearchnet_20000_chatml.jsonl",
}


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _with_task_ids(rows: list[dict], language: str, split: str) -> list[dict]:
    result = []
    for idx, row in enumerate(rows):
        copied = dict(row)
        copied["task_id"] = copied.get("task_id") or f"{language}_{split}_{idx}"
        copied["eval_language"] = language
        result.append(copied)
    return result


def stratified_split(rows: list[dict], test_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    by_kind: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_kind[str(row.get("target_kind") or "unknown")].append(idx)

    rng = random.Random(seed)
    test_indices: set[int] = set()
    total = len(rows)
    remaining_slots = min(test_size, total)

    kind_items = sorted(by_kind.items(), key=lambda item: item[0])
    fractional: list[tuple[float, str, int]] = []
    for kind, indices in kind_items:
        exact = len(indices) * min(test_size, total) / total
        take = int(exact)
        fractional.append((exact - take, kind, len(indices)))
        shuffled = list(indices)
        rng.shuffle(shuffled)
        selected = shuffled[: min(take, len(shuffled))]
        test_indices.update(selected)
        remaining_slots -= len(selected)

    if remaining_slots > 0:
        for _, kind, _ in sorted(fractional, reverse=True):
            candidates = [idx for idx in by_kind[kind] if idx not in test_indices]
            rng.shuffle(candidates)
            for idx in candidates[:remaining_slots]:
                test_indices.add(idx)
                remaining_slots -= 1
                if remaining_slots <= 0:
                    break
            if remaining_slots <= 0:
                break

    train_rows = [row for idx, row in enumerate(rows) if idx not in test_indices]
    test_rows = [row for idx, row in enumerate(rows) if idx in test_indices]
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/test splits for the multilingual code-completion data."
    )
    parser.add_argument("--output-dir", default="data/multilang_splits")
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary = {}
    for language, raw_path in LANGUAGE_FILES.items():
        rows = _read_jsonl(Path(raw_path))
        train_rows, test_rows = stratified_split(rows, args.test_size, args.seed)
        train_rows = _with_task_ids(train_rows, language, "train")
        test_rows = _with_task_ids(test_rows, language, "test")
        _write_jsonl(output_dir / f"{language}_train.jsonl", train_rows)
        _write_jsonl(output_dir / f"{language}_test.jsonl", test_rows)
        summary[language] = {
            "raw": len(rows),
            "train": len(train_rows),
            "test": len(test_rows),
        }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
