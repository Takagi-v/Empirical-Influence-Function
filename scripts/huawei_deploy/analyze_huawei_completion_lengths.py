#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|:=|&&|\|\||[-+*/%&|^!~<>=.:;,{}()\[\]]|\S")
LINE_THRESHOLDS = [1, 2, 3, 4, 5, 8, 10, 20]
TOKEN_THRESHOLDS = [32, 64, 96, 128, 192, 256, 512]
CHAR_THRESHOLDS = [128, 256, 512, 1024, 2048]
PACKAGE_RE = re.compile(r"Below is the package path:(.*?)\n\n", re.DOTALL)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield index, json.loads(line)


def normalize_marker_payload(text: str) -> str:
    text = text.strip("\n")
    if text.startswith(" "):
        text = text[1:]
    return text


def extract_fim_parts(prompt: str) -> tuple[str, str] | None:
    pre_pos = prompt.rfind("<PRE>")
    if pre_pos < 0:
        return None
    suf_pos = prompt.find("<SUF>", pre_pos + len("<PRE>"))
    if suf_pos < 0:
        return None
    mid_pos = prompt.find("<MID>", suf_pos + len("<SUF>"))
    if mid_pos < 0:
        return None
    prefix = normalize_marker_payload(prompt[pre_pos + len("<PRE>"):suf_pos])
    suffix = normalize_marker_payload(prompt[suf_pos + len("<SUF>"):mid_pos])
    return prefix, suffix


def package_path(prompt: str) -> str:
    match = PACKAGE_RE.search(prompt)
    return match.group(1).strip() if match else ""


def nonempty_line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def rough_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def has_cjk_comment(text: str) -> bool:
    in_block = False
    for raw_line in text.splitlines():
        line = raw_line
        while line:
            if in_block:
                end = line.find("*/")
                comment = line if end < 0 else line[:end]
                if CJK_RE.search(comment):
                    return True
                if end < 0:
                    line = ""
                else:
                    line = line[end + 2:]
                    in_block = False
                continue
            slash = line.find("//")
            block = line.find("/*")
            if slash < 0 and block < 0:
                break
            if slash >= 0 and (block < 0 or slash < block):
                if CJK_RE.search(line[slash + 2:]):
                    return True
                break
            if block >= 0:
                end = line.find("*/", block + 2)
                comment = line[block + 2:] if end < 0 else line[block + 2:end]
                if CJK_RE.search(comment):
                    return True
                if end < 0:
                    in_block = True
                    break
                line = line[end + 2:]
    return False


def percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    values = sorted(values)
    idx = int(round((pct / 100) * (len(values) - 1)))
    return values[idx]


def pct(n: int, total: int) -> float:
    return round(n * 100.0 / total, 4) if total else 0.0


def add_threshold_counts(report: dict[str, Any], name: str, values: list[int], thresholds: list[int], total: int) -> None:
    report[name] = {
        f">{thr}": {"count": sum(1 for x in values if x > thr), "pct": pct(sum(1 for x in values if x > thr), total)}
        for thr in thresholds
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Huawei raw Go FIM completion length and Chinese-comment frequency.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=Path("outputs/huawei_deploy/huawei_completion_length_report.json"))
    parser.add_argument("--long-sample-output", type=Path, default=Path("outputs/huawei_deploy/huawei_long_completion_samples.jsonl"))
    parser.add_argument("--max-nonempty-lines", type=int, default=3)
    parser.add_argument("--max-rough-tokens", type=int, default=96)
    parser.add_argument("--max-chars", type=int, default=512)
    parser.add_argument("--sample-limit", type=int, default=100)
    args = parser.parse_args()

    stats: Counter[str] = Counter()
    line_counts: list[int] = []
    token_counts: list[int] = []
    char_counts: list[int] = []
    long_examples: list[dict[str, Any]] = []

    for index, row in iter_jsonl(args.input_path):
        stats["seen"] += 1
        prompt = str(row.get("prompt") or "")
        target = str(row.get("response") or "")
        task_id = str(row.get("task_id") or f"row_{index}")
        parts = extract_fim_parts(prompt)
        if parts is None or not target:
            stats["missing_markers_or_target"] += 1
            continue
        prefix, suffix = parts
        stats["valid"] += 1
        lines = nonempty_line_count(target)
        rough_tokens = rough_token_count(target)
        chars = len(target)
        line_counts.append(lines)
        token_counts.append(rough_tokens)
        char_counts.append(chars)

        if CJK_RE.search(prompt):
            stats["prompt_has_cjk"] += 1
        if has_cjk_comment(prefix):
            stats["prefix_has_cjk_comment"] += 1
        if has_cjk_comment(suffix):
            stats["suffix_has_cjk_comment"] += 1
        if has_cjk_comment(target):
            stats["target_has_cjk_comment"] += 1

        is_long = lines > args.max_nonempty_lines or rough_tokens > args.max_rough_tokens or chars > args.max_chars
        if is_long:
            stats["long_by_default_rule"] += 1
            if len(long_examples) < args.sample_limit:
                long_examples.append({
                    "index": index,
                    "task_id": task_id,
                    "package_path": package_path(prompt),
                    "nonempty_lines": lines,
                    "rough_tokens": rough_tokens,
                    "chars": chars,
                    "target_has_cjk_comment": has_cjk_comment(target),
                    "prefix_has_cjk_comment": has_cjk_comment(prefix),
                    "suffix_has_cjk_comment": has_cjk_comment(suffix),
                    "target": target,
                })

    total = stats["valid"]
    report: dict[str, Any] = {
        "input_path": str(args.input_path),
        "default_long_rule": {
            "max_nonempty_lines": args.max_nonempty_lines,
            "max_rough_tokens": args.max_rough_tokens,
            "max_chars": args.max_chars,
            "long_count": stats["long_by_default_rule"],
            "long_pct": pct(stats["long_by_default_rule"], total),
            "kept_count": total - stats["long_by_default_rule"],
            "kept_pct": pct(total - stats["long_by_default_rule"], total),
        },
        "stats": dict(stats),
        "completion_nonempty_lines": {
            "mean": round(mean(line_counts), 4) if line_counts else None,
            "p50": percentile(line_counts, 50),
            "p75": percentile(line_counts, 75),
            "p90": percentile(line_counts, 90),
            "p95": percentile(line_counts, 95),
            "p99": percentile(line_counts, 99),
            "max": max(line_counts) if line_counts else None,
        },
        "completion_rough_tokens": {
            "mean": round(mean(token_counts), 4) if token_counts else None,
            "p50": percentile(token_counts, 50),
            "p75": percentile(token_counts, 75),
            "p90": percentile(token_counts, 90),
            "p95": percentile(token_counts, 95),
            "p99": percentile(token_counts, 99),
            "max": max(token_counts) if token_counts else None,
        },
        "completion_chars": {
            "mean": round(mean(char_counts), 4) if char_counts else None,
            "p50": percentile(char_counts, 50),
            "p75": percentile(char_counts, 75),
            "p90": percentile(char_counts, 90),
            "p95": percentile(char_counts, 95),
            "p99": percentile(char_counts, 99),
            "max": max(char_counts) if char_counts else None,
        },
        "chinese_comment_pct": {
            "prompt_has_cjk": pct(stats["prompt_has_cjk"], total),
            "prefix_has_cjk_comment": pct(stats["prefix_has_cjk_comment"], total),
            "suffix_has_cjk_comment": pct(stats["suffix_has_cjk_comment"], total),
            "target_has_cjk_comment": pct(stats["target_has_cjk_comment"], total),
        },
    }
    add_threshold_counts(report, "line_thresholds", line_counts, LINE_THRESHOLDS, total)
    add_threshold_counts(report, "rough_token_thresholds", token_counts, TOKEN_THRESHOLDS, total)
    add_threshold_counts(report, "char_thresholds", char_counts, CHAR_THRESHOLDS, total)

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.long_sample_output.parent.mkdir(parents=True, exist_ok=True)
    with args.long_sample_output.open("w", encoding="utf-8") as f:
        for row in long_examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"long_examples={args.long_sample_output}")


if __name__ == "__main__":
    main()
