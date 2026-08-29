#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = "You are a Go code completion assistant."
USER_TEMPLATE = (
    "Fill the missing part in the Go function. Return only the missing Go code, "
    "without Markdown fences or explanation.\n\n"
    "* Incomplete Code:\n"
    "{prefix}[MASK]{suffix}"
)
PACKAGE_RE = re.compile(r"Below is the package path:(.*?)\n\n", re.DOTALL)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|:=|&&|\|\||[-+*/%&|^!~<>=.:;,{}()\[\]]|\S")


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


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


def extract_package_path(prompt: str) -> str:
    m = PACKAGE_RE.search(prompt)
    return m.group(1).strip() if m else ""


def rough_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def nonempty_line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def gofmt_check_full_code(full_code: str, gofmt_bin: str) -> tuple[bool, str]:
    stripped = full_code.lstrip()
    wrapped = full_code if stripped.startswith("package ") else "package main\n\n" + full_code
    proc = subprocess.run(
        [gofmt_bin],
        input=wrapped,
        text=True,
        capture_output=True,
    )
    if proc.returncode == 0:
        return True, ""
    stderr = (proc.stderr or "").strip()
    return False, stderr.splitlines()[0] if stderr else f"{gofmt_bin} failed with exit code {proc.returncode}"


def strip_cjk_comments(text: str) -> tuple[str, int]:
    """Remove Go line/block comments containing CJK characters; keep non-CJK comments."""
    out: list[str] = []
    i = 0
    removed = 0
    n = len(text)
    string_quote: str | None = None
    escaped = False

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if string_quote:
            out.append(ch)
            if string_quote != "`" and escaped:
                escaped = False
            elif string_quote != "`" and ch == "\\":
                escaped = True
            elif ch == string_quote:
                string_quote = None
            i += 1
            continue

        if ch in {'"', "'", "`"}:
            string_quote = ch
            out.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            end = text.find("\n", i)
            if end < 0:
                comment = text[i:]
                newline = ""
                i = n
            else:
                comment = text[i:end]
                newline = text[end]
                i = end + 1
            if CJK_RE.search(comment):
                removed += 1
                out.append(newline)
            else:
                out.append(comment)
                out.append(newline)
            continue

        if ch == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            if end < 0:
                comment = text[i:]
                i = n
            else:
                comment = text[i:end + 2]
                i = end + 2
            if CJK_RE.search(comment):
                removed += 1
                out.append("\n" * comment.count("\n"))
            else:
                out.append(comment)
            continue

        out.append(ch)
        i += 1

    return "".join(out), removed


def make_sample(
    row: dict[str, Any],
    index: int,
    *,
    include_raw_prompt: bool,
    strip_comments: bool,
    max_target_nonempty_lines: int,
    max_target_rough_tokens: int,
    max_target_chars: int,
    filter_gofmt_valid: bool,
    gofmt_bin: str,
) -> tuple[dict[str, Any] | None, str | None]:
    prompt = str(row.get("prompt") or "")
    target = str(row.get("target", row.get("response", "")) or "")
    task_id = str(row.get("task_id") or row.get("raw_id") or row.get("uid") or f"row_{index}")
    source_format = "prompt_response_task_id_with_PRE_SUF_MID"
    parts = extract_fim_parts(prompt)
    if parts is None:
        if "prefix" in row and "suffix" in row and target:
            prefix = str(row.get("prefix") or "")
            suffix = str(row.get("suffix") or "")
            source_format = "canonical_or_chatml_prefix_target_suffix"
        else:
            return None, "missing_markers_or_target"
    elif target:
        prefix, suffix = parts
    else:
        return None, "missing_markers_or_target"

    removed_comments = {"prefix": 0, "target": 0, "suffix": 0}
    if strip_comments:
        prefix, removed_comments["prefix"] = strip_cjk_comments(prefix)
        target, removed_comments["target"] = strip_cjk_comments(target)
        suffix, removed_comments["suffix"] = strip_cjk_comments(suffix)
        if not target.strip():
            return None, "empty_target_after_comment_strip"

    target_nonempty_lines = nonempty_line_count(target)
    target_rough_tokens = rough_token_count(target)
    target_chars = len(target)
    if max_target_nonempty_lines > 0 and target_nonempty_lines > max_target_nonempty_lines:
        return None, "target_too_many_nonempty_lines"
    if max_target_rough_tokens > 0 and target_rough_tokens > max_target_rough_tokens:
        return None, "target_too_many_rough_tokens"
    if max_target_chars > 0 and target_chars > max_target_chars:
        return None, "target_too_many_chars"

    full_code = prefix + target + suffix
    if filter_gofmt_valid:
        ok, _ = gofmt_check_full_code(full_code, gofmt_bin)
        if not ok:
            return None, "gofmt_invalid_full_code"

    uid_seed = "\n".join([task_id, prefix, target, suffix])
    uid = f"huawei_go_{stable_hash(uid_seed)}"
    user = USER_TEMPLATE.format(prefix=prefix, suffix=suffix)
    sample: dict[str, Any] = {
        "uid": uid,
        "raw_id": str(row.get("raw_id") or task_id),
        "source_dataset": str(row.get("source_dataset") or "huawei_cloud_core_go"),
        "split": "train",
        "language": "go",
        "task_type": "go_fim_completion_huawei_derived",
        "prefix": prefix,
        "target": target,
        "suffix": suffix,
        "full_code": full_code,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "only_last_turn_loss": True,
        "metadata": {
            "raw_task_id": task_id,
            "source_benchmark": "Huawei cloud_core Go",
            "source_format": source_format,
            "package_path": extract_package_path(prompt),
            "raw_prompt_sha1": hashlib.sha1(prompt.encode("utf-8", errors="ignore")).hexdigest(),
            "raw_index": index,
            "prefix_chars": len(prefix),
            "target_chars": target_chars,
            "suffix_chars": len(suffix),
            "target_nonempty_lines": target_nonempty_lines,
            "target_rough_tokens": target_rough_tokens,
            "strip_cjk_comments": strip_comments,
            "removed_cjk_comments": removed_comments,
            "filter_gofmt_valid": filter_gofmt_valid,
        },
        "judge_payload": {"kind": "none"},
    }
    if include_raw_prompt:
        sample["metadata"]["raw_prompt"] = prompt
    return sample, None


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield index, json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Huawei <PRE>/<SUF>/<MID> Go data to project ChatML/FIM format.")
    p.add_argument("--input-path", required=True, type=Path)
    p.add_argument("--chatml-output", required=True, type=Path)
    p.add_argument("--canonical-output", type=Path)
    p.add_argument("--report-output", type=Path)
    p.add_argument("--max-rows", type=int, default=0, help="Read at most N raw rows; 0 means all rows")
    p.add_argument("--max-accepted-rows", type=int, default=0, help="Stop after writing N accepted rows; 0 means no accepted-row limit")
    p.add_argument("--include-raw-prompt", action="store_true", help="Store full raw prompt in metadata; increases output size a lot.")
    p.add_argument("--strip-cjk-comments", action="store_true", help="Remove Go comments containing CJK characters from prefix/target/suffix.")
    p.add_argument("--max-target-nonempty-lines", type=int, default=0, help="Reject samples with target non-empty lines above this value; 0 disables.")
    p.add_argument("--max-target-rough-tokens", type=int, default=0, help="Reject samples with target rough token count above this value; 0 disables.")
    p.add_argument("--max-target-chars", type=int, default=0, help="Reject samples with target char count above this value; 0 disables.")
    p.add_argument("--filter-gofmt-valid", action="store_true", help="Reject samples whose prefix+target+suffix is not parseable by gofmt.")
    p.add_argument("--gofmt-bin", default="gofmt", help="gofmt executable used by --filter-gofmt-valid.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.filter_gofmt_valid and shutil.which(args.gofmt_bin) is None:
        raise SystemExit(f"--filter-gofmt-valid requires gofmt executable, but not found: {args.gofmt_bin}")

    rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen: set[str] = set()

    for index, row in iter_jsonl(args.input_path):
        if args.max_rows > 0 and stats["seen"] >= args.max_rows:
            break
        if args.max_accepted_rows > 0 and stats["accepted"] >= args.max_accepted_rows:
            break
        stats["seen"] += 1
        sample, reason = make_sample(
            row,
            index,
            include_raw_prompt=args.include_raw_prompt,
            strip_comments=args.strip_cjk_comments,
            max_target_nonempty_lines=args.max_target_nonempty_lines,
            max_target_rough_tokens=args.max_target_rough_tokens,
            max_target_chars=args.max_target_chars,
            filter_gofmt_valid=args.filter_gofmt_valid,
            gofmt_bin=args.gofmt_bin,
        )
        if sample is None:
            stats["rejected"] += 1
            stats[f"reject_{reason or 'unknown'}"] += 1
            rejects.append({"index": index, "task_id": row.get("task_id"), "reason": reason or "unknown"})
            continue
        if sample["uid"] in seen:
            stats["duplicate_uid"] += 1
            stats["rejected"] += 1
            rejects.append({"index": index, "task_id": row.get("task_id"), "reason": "duplicate_uid", "uid": sample["uid"]})
            continue
        seen.add(sample["uid"])
        rows.append(sample)
        stats["accepted"] += 1
        removed = sample["metadata"].get("removed_cjk_comments", {})
        for part, count in removed.items():
            if count:
                stats[f"removed_cjk_comment_samples_{part}"] += 1
                stats[f"removed_cjk_comments_{part}"] += int(count)

    canonical_output = args.canonical_output or args.chatml_output.with_name(args.chatml_output.name.replace("_chatml.jsonl", "_canonical.jsonl"))
    report_output = args.report_output or args.chatml_output.with_suffix(args.chatml_output.suffix + ".report.json")
    write_jsonl(args.chatml_output, rows)
    write_jsonl(canonical_output, rows)
    report = {
        "input_path": str(args.input_path),
        "chatml_output": str(args.chatml_output),
        "canonical_output": str(canonical_output),
        "cleaning_config": {
            "strip_cjk_comments": args.strip_cjk_comments,
            "max_rows": args.max_rows,
            "max_accepted_rows": args.max_accepted_rows,
            "max_target_nonempty_lines": args.max_target_nonempty_lines,
            "max_target_rough_tokens": args.max_target_rough_tokens,
            "max_target_chars": args.max_target_chars,
            "filter_gofmt_valid": args.filter_gofmt_valid,
            "gofmt_bin": args.gofmt_bin,
        },
        "stats": dict(stats),
        "rejects_preview": rejects[:50],
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if rejects:
        reject_path = report_output.with_suffix(report_output.suffix + ".rejects.json")
        reject_path.write_text(json.dumps(rejects, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
