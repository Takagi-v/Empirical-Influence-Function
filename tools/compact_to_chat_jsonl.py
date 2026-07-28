#!/usr/bin/env python3
"""Convert graphsignal compact JSONL -> EIF chat JSONL.

Compact rows look like:
  {uid, input_ids, label, attention_edges, ...}

EIF / intervention_experiment accepts:
  {"messages":[{"role":"system","content":...},{"role":"user",...},{"role":"assistant",...}],
   "task_id": "..."}
or
  {"system":"...","prompt":"...","response":"...","task_id":"..."}

This script emits Format A (messages) and optionally verifies that re-encoding
with the same ChatML recipe as src.process_data.process_func_chatml reproduces
the original input_ids (token-level roundtrip).

Example:
  python tools/compact_to_chat_jsonl.py \\
    --tokenizer d:/AAAworks/code-corr-annotation/models/Qwen2.5-Coder-7B-Instruct \\
    --input  d:/AAAworks/code-corr-annotation/data/data_5k/eval_data/codesearchnet_go_test_1000_graphsignal_500_compact.json \\
    --output data/converted/csn500_test_chat.jsonl \\
    --verify

  python tools/compact_to_chat_jsonl.py \\
    --tokenizer d:/AAAworks/code-corr-annotation/models/Qwen2.5-Coder-7B-Instruct \\
    --input  d:/AAAworks/code-corr-annotation/data/data_10k/train_data/go_single_train_v2_graphsignal_10k_compact.json.bak \\
    --output data/converted/csn10k_train_chat.jsonl \\
    --verify --verify-limit 200
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

TURN_RE = re.compile(
    r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>",
    flags=re.DOTALL,
)


def load_tokenizer(path: str):
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)


def encode_chatml_like_eif(tokenizer, system: str, user: str, assistant: str) -> list[int]:
    """Mirror src.process_data.process_func_chatml encoding (no truncation)."""
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_tokens = tokenizer.encode("\n", add_special_tokens=False)

    def build_turn(role: str, content: str) -> list[int]:
        role_ids = [im_start_id] + tokenizer.encode(role, add_special_tokens=False) + nl_tokens
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        footer_ids = [im_end_id] + nl_tokens
        return role_ids + content_ids + footer_ids

    return build_turn("system", system) + build_turn("user", user) + build_turn("assistant", assistant)


def split_chatml_roles(decoded: str) -> dict[str, str]:
    parts = TURN_RE.findall(decoded)
    if len(parts) < 3:
        raise ValueError(f"expected >=3 ChatML turns, got {len(parts)}: roles={[p[0] for p in parts]}")
    roles = {role: content for role, content in parts}
    for need in ("system", "user", "assistant"):
        if need not in roles:
            raise ValueError(f"missing role {need!r}; found {sorted(roles)}")
    # If multiple turns of same role appear, keep the last (should not happen for this data).
    return {
        "system": roles["system"],
        "user": roles["user"],
        "assistant": roles["assistant"],
    }


def compact_row_to_messages(tokenizer, row: dict[str, Any]) -> dict[str, Any]:
    ids = row.get("input_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("row missing input_ids")

    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    roles = split_chatml_roles(decoded)

    # Optional consistency check vs label mask (answer region).
    labels = row.get("label") or row.get("labels")
    if isinstance(labels, list) and len(labels) == len(ids):
        ans_ids = [tid for tid, lab in zip(ids, labels) if lab != -100]
        if ans_ids:
            ans_text = tokenizer.decode(ans_ids, skip_special_tokens=False)
            # Label span is assistant content + <|im_end|>\n ; strip footer for compare.
            ans_core = re.sub(r"<\|im_end\|>\n?$", "", ans_text)
            if ans_core != roles["assistant"]:
                # Soft warning only — still emit role-split content.
                pass

    uid = row.get("uid") or row.get("raw_id") or row.get("task_id") or ""
    out = {
        "messages": [
            {"role": "system", "content": roles["system"]},
            {"role": "user", "content": roles["user"]},
            {"role": "assistant", "content": roles["assistant"]},
        ],
        "task_id": str(uid),
        "uid": str(uid),
        "language": row.get("language"),
    }
    # Keep original token ids/labels so free-run matches viz (ids[:first label!=-100]).
    out["input_ids"] = list(ids)
    if isinstance(labels, list) and len(labels) == len(ids):
        out["label"] = list(labels)
    # Keep graphsignal edges so CE+saliency train-bank can use the training objective.
    edges = row.get("attention_edges")
    if edges is not None:
        out["attention_edges"] = edges
    return out


def verify_row(tokenizer, row: dict[str, Any], messages_obj: dict[str, Any]) -> tuple[bool, str]:
    msgs = messages_obj["messages"]
    system = msgs[0]["content"]
    user = msgs[1]["content"]
    assistant = msgs[2]["content"]
    reenc = encode_chatml_like_eif(tokenizer, system, user, assistant)
    orig = row["input_ids"]
    if reenc == orig:
        return True, "ok"
    # Detail first mismatch
    n = min(len(reenc), len(orig))
    for i in range(n):
        if reenc[i] != orig[i]:
            return False, f"mismatch at {i}: orig={orig[i]} reenc={reenc[i]} (len {len(orig)} vs {len(reenc)})"
    return False, f"length mismatch: orig={len(orig)} reenc={len(reenc)}"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert compact JSONL to EIF chat JSONL.")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer / base model dir (Qwen).")
    parser.add_argument("--input", required=True, help="Compact JSONL path (.json / .json.bak).")
    parser.add_argument("--output", required=True, help="Output chat JSONL path.")
    parser.add_argument("--limit", type=int, default=0, help="Convert at most N rows (0=all).")
    parser.add_argument("--verify", action="store_true", help="Roundtrip-check re-encoding vs input_ids.")
    parser.add_argument("--verify-limit", type=int, default=0, help="Only verify first N converted rows (0=all verified rows).")
    parser.add_argument("--fail-on-verify-error", action="store_true", help="Exit non-zero if any verify fails.")
    parser.add_argument("--skip-bad", action="store_true", help="Skip rows that fail parse/verify instead of aborting.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print(f"[error] input not found: {in_path}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[info] loading tokenizer from {args.tokenizer}", flush=True)
    tokenizer = load_tokenizer(args.tokenizer)

    n_ok = n_skip = n_verify_fail = 0
    verify_budget = args.verify_limit if args.verify_limit > 0 else None

    with out_path.open("w", encoding="utf-8") as out_f:
        for line_no, row in iter_jsonl(in_path):
            if args.limit and n_ok + n_skip >= args.limit:
                break
            try:
                obj = compact_row_to_messages(tokenizer, row)
                if args.verify and (verify_budget is None or (n_ok + n_verify_fail) < verify_budget):
                    ok, reason = verify_row(tokenizer, row, obj)
                    if not ok:
                        n_verify_fail += 1
                        msg = f"[verify-fail] line={line_no} uid={obj.get('task_id')}: {reason}"
                        print(msg, file=sys.stderr)
                        if args.fail_on_verify_error and not args.skip_bad:
                            return 2
                        if args.skip_bad:
                            n_skip += 1
                            continue
                # Drop null language for cleaner output
                if obj.get("language") is None:
                    obj.pop("language", None)
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_ok += 1
                if n_ok % 500 == 0:
                    print(f"[info] wrote {n_ok} rows...", flush=True)
            except Exception as e:
                n_skip += 1
                print(f"[skip] line={line_no}: {e}", file=sys.stderr)
                if not args.skip_bad:
                    return 1

    print(
        f"[done] output={out_path} wrote={n_ok} skipped={n_skip} verify_fail={n_verify_fail}",
        flush=True,
    )
    if args.fail_on_verify_error and n_verify_fail:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
