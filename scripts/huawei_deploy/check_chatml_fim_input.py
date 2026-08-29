#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def validate_row(row: dict[str, Any], index: int) -> list[str]:
    errors: list[str] = []
    if not row.get("uid"):
        errors.append("missing uid")
    if str(row.get("language", "")).lower() not in {"go", "golang"}:
        errors.append("language is not go/golang")
    target = row.get("target", row.get("fim_completion"))
    if not isinstance(target, str) or not target:
        errors.append("missing non-empty target")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        errors.append("messages must contain system/user/assistant turns")
    else:
        roles = [m.get("role") for m in messages if isinstance(m, dict)]
        if "user" not in roles:
            errors.append("missing user message")
        if "assistant" not in roles:
            errors.append("missing assistant message")
        user_text = next((str(m.get("content", "")) for m in messages if isinstance(m, dict) and m.get("role") == "user"), "")
        assistant_text = next((str(m.get("content", "")) for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "assistant"), "")
        if "[MASK]" not in user_text:
            errors.append("user message does not contain [MASK]")
        if isinstance(target, str) and assistant_text != target:
            errors.append("assistant content does not equal target")
    return [f"row {index}: {err}" for err in errors]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ChatML/FIM jsonl rows before annotation.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    rows = load_rows(Path(args.input_path), args.limit)
    all_errors: list[str] = []
    for idx, row in enumerate(rows):
        all_errors.extend(validate_row(row, idx))

    print(f"checked={len(rows)}")
    if all_errors:
        print(f"errors={len(all_errors)}")
        for err in all_errors[:50]:
            print(err)
        raise SystemExit(1)
    print("ok")


if __name__ == "__main__":
    main()
