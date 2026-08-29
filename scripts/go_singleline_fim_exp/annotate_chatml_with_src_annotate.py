#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import inspect
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotate.neural_annot import (
    AnnotatorAgent,
    SyntacticCheckerTool,
    _rate_limited_chat_completion,
    get_thread_local_openai_client,
)
from src.annotate.utils import (
    TokenCorrelation,
    get_token_indices_in_span,
    map_simple_to_bpe,
    normalize_fim_annotation_edge_direction,
    tokenize_code_for_annotation,
)


IGNORE_INDEX = -100
MASK_TOKEN = "[MASK]"
INCOMPLETE_CODE_RE = re.compile(r"\* Incomplete Code:\n(.*?)(?:\n+Please fill|$)", re.DOTALL)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows from {path}")
    return rows


def write_jsonl_atomic(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"Saved {len(rows)} rows to {path}")


def to_dict(obj: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return dict(obj)


def row_key(row: dict[str, Any]) -> str:
    uid = row.get("uid")
    if uid:
        return f"uid:{uid}"
    payload = json.dumps(
        {
            "messages": row.get("messages"),
            "target": row.get("target", row.get("fim_completion", "")),
            "full_code": row.get("full_code", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return "sha:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_cache(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
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
            if key:
                cache[str(key)] = obj
    print(f"Loaded src-annotate cache: {len(cache)} entries from {p}")
    return cache


def save_cache(cache: dict[str, dict[str, Any]], path: str | Path | None) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for key in sorted(cache):
            f.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
    tmp.replace(p)
    print(f"Saved src-annotate cache: {len(cache)} entries to {p}")


def load_seed_compact(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = row.get("uid") or row.get("raw_id")
            if uid and row.get("attention_edges"):
                out[f"uid:{uid}"] = row
    print(f"Loaded seed compact annotations: {len(out)} entries from {p}")
    return out


def normalize_language(value: Any) -> str:
    text = str(value or "Go")
    aliases = {
        "go": "Go",
        "golang": "Go",
        "python": "Python",
        "java": "Java",
        "c": "C",
        "cpp": "CPP",
        "c++": "CPP",
        "csharp": "C#",
        "c#": "C#",
    }
    return aliases.get(text.lower(), text)


def get_user_content(row: dict[str, Any]) -> str:
    for msg in row.get("messages") or []:
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return str(row.get("instruction", ""))


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = [
        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
        for m in row.get("messages") or []
    ]
    if not any(m["role"] == "assistant" for m in messages):
        messages.append({"role": "assistant", "content": str(row.get("target", row.get("fim_completion", "")))})
    return messages


def chatml_text(messages: list[dict[str, str]]) -> str:
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)


def binarize_messages(messages: list[dict[str, str]], tokenizer: Any, max_len: int) -> dict[str, Any] | None:
    text = chatml_text(messages)
    enc = tokenizer(text, add_special_tokens=False)
    input_ids = list(enc.input_ids)
    if not input_ids or len(input_ids) > max_len:
        return None

    labels = [IGNORE_INDEX] * len(input_ids)
    cursor = 0
    assistant_indices = [i for i, msg in enumerate(messages) if msg["role"] == "assistant"]
    if not assistant_indices:
        return None
    target_idx = assistant_indices[-1]
    for idx, msg in enumerate(messages):
        prefix = f"<|im_start|>{msg['role']}\n"
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        content_ids = tokenizer(msg["content"], add_special_tokens=False).input_ids
        content_start = cursor + len(prefix_ids)
        content_end = content_start + len(content_ids)
        if idx == target_idx:
            labels[content_start:content_end] = input_ids[content_start:content_end]
        cursor += len(tokenizer(f"{prefix}{msg['content']}<|im_end|>\n", add_special_tokens=False).input_ids)

    if not any(v != IGNORE_INDEX for v in labels):
        return None
    return {"input_ids": input_ids, "label": labels, "length": len(input_ids), "chatml_text": text}


def get_chatml_offsets(messages: list[dict[str, str]]) -> tuple[int, int, str, str]:
    text_parts: list[str] = []
    user_content_start = -1
    assistant_content_start = -1
    user_content = ""
    assistant_content = ""
    for msg in messages:
        prefix = f"<|im_start|>{msg['role']}\n"
        if msg["role"] == "user" and user_content_start < 0:
            user_content_start = sum(len(p) for p in text_parts) + len(prefix)
            user_content = msg["content"]
        if msg["role"] == "assistant":
            assistant_content_start = sum(len(p) for p in text_parts) + len(prefix)
            assistant_content = msg["content"]
        text_parts.append(f"{prefix}{msg['content']}<|im_end|>\n")
    if user_content_start < 0 or assistant_content_start < 0:
        raise ValueError("messages must contain user and assistant content")
    return user_content_start, assistant_content_start, user_content, assistant_content


def find_code_mask_pos(user_content: str) -> int:
    marker = "* Incomplete Code:\n"
    marker_pos = user_content.find(marker)
    if marker_pos >= 0:
        code_start = marker_pos + len(marker)
        mask_pos = user_content.find(MASK_TOKEN, code_start)
        if mask_pos >= 0:
            return mask_pos
    return user_content.rfind(MASK_TOKEN)


def build_filled_instruction(user_content: str, target: str) -> tuple[str, int, int]:
    mask_pos = find_code_mask_pos(user_content)
    if mask_pos < 0:
        raise ValueError("user content does not contain [MASK]")
    filled = user_content[:mask_pos] + target + user_content[mask_pos + len(MASK_TOKEN):]
    return filled, mask_pos, mask_pos + len(target)


def remap_filled_offset_to_chatml(
    offset: int,
    *,
    user_content_start: int,
    assistant_content_start: int,
    mask_pos: int,
    target_len: int,
) -> int:
    target_end = mask_pos + target_len
    if offset < mask_pos:
        return user_content_start + offset
    if offset < target_end:
        return assistant_content_start + (offset - mask_pos)
    shrink = target_len - len(MASK_TOKEN)
    return user_content_start + offset - shrink


def make_ns_token(tok: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        surface=tok.get("surface", ""),
        token_id=tok.get("token_id", -1),
        char_start=int(tok.get("char_start", 0)),
        char_end=int(tok.get("char_end", 0)),
    )


def map_to_qwen_annotations(
    *,
    tokenizer: Any,
    messages: list[dict[str, str]],
    filled_tokens: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    mask_pos: int,
    target_len: int,
    completion_indices: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text = chatml_text(messages)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    qwen_tokens = [
        {
            "surface": tokenizer.decode([token_id]),
            "token_id": int(token_id),
            "char_start": int(start),
            "char_end": int(end),
        }
        for token_id, (start, end) in zip(enc["input_ids"], enc["offset_mapping"])
    ]
    user_start, assistant_start, _, _ = get_chatml_offsets(messages)

    remapped_simple: list[SimpleNamespace] = []
    for tok in filled_tokens:
        start = int(tok["char_start"])
        end = int(tok["char_end"])
        new_start = remap_filled_offset_to_chatml(
            start,
            user_content_start=user_start,
            assistant_content_start=assistant_start,
            mask_pos=mask_pos,
            target_len=target_len,
        )
        new_end = new_start + max(0, end - start)
        remapped_simple.append(
            SimpleNamespace(
                surface=tok.get("surface", ""),
                token_id=tok.get("token_id", -1),
                char_start=new_start,
                char_end=new_end,
            )
        )

    qwen_ns = [SimpleNamespace(char_start=t["char_start"], char_end=t["char_end"]) for t in qwen_tokens]
    sorted_idx = sorted(range(len(remapped_simple)), key=lambda i: remapped_simple[i].char_start)
    sorted_simple = [remapped_simple[i] for i in sorted_idx]
    sorted_map = map_simple_to_bpe(sorted_simple, qwen_ns)
    s2b: dict[int, list[int]] = {}
    for pos, orig_idx in enumerate(sorted_idx):
        s2b[orig_idx] = sorted_map.get(pos, [])

    qwen_annotations: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for ann in annotations:
        try:
            si = int(ann["token_i_idx"])
            sj = int(ann["token_j_idx"])
        except Exception:
            continue
        subtype = str(ann.get("subtype", "") or "")
        si_is_completion = si in completion_indices
        sj_is_completion = sj in completion_indices
        for qi in s2b.get(si, []):
            for qj in s2b.get(sj, []):
                normalized = normalize_fim_annotation_edge_direction(
                    si_is_completion=si_is_completion,
                    sj_is_completion=sj_is_completion,
                    qi=qi,
                    qj=qj,
                    edge_scope="context_response_prompt",
                )
                if normalized is None:
                    continue
                src, dst = normalized
                key = (src, dst, subtype)
                if key in seen:
                    continue
                seen.add(key)
                qwen_annotations.append(
                    {
                        "token_i": qwen_tokens[src]["surface"],
                        "token_j": qwen_tokens[dst]["surface"],
                        "source": ann.get("source", "Neural"),
                        "subtype": subtype,
                        "token_i_idx": int(src),
                        "token_j_idx": int(dst),
                    }
                )
    return qwen_tokens, qwen_annotations


def mask_qwen_token_indices(tokenizer: Any, messages: list[dict[str, str]]) -> set[int]:
    text = chatml_text(messages)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    user_start, _, user_content, _ = get_chatml_offsets(messages)
    out: set[int] = set()
    search_from = 0
    while True:
        mask_pos = user_content.find(MASK_TOKEN, search_from)
        if mask_pos < 0:
            break
        mask_start = user_start + mask_pos
        mask_end = mask_start + len(MASK_TOKEN)
        for idx, (start, end) in enumerate(enc["offset_mapping"]):
            if int(start) < mask_end and mask_start < int(end):
                out.add(idx)
        search_from = mask_pos + len(MASK_TOKEN)
    return out


def drop_mask_token_annotations(
    qwen_annotations: list[dict[str, Any]],
    *,
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> list[dict[str, Any]]:
    mask_indices = mask_qwen_token_indices(tokenizer, messages)
    if not mask_indices:
        return qwen_annotations
    out: list[dict[str, Any]] = []
    for ann in qwen_annotations:
        try:
            src = int(ann["token_i_idx"])
            dst = int(ann["token_j_idx"])
        except Exception:
            continue
        if src in mask_indices or dst in mask_indices:
            continue
        out.append(ann)
    return out


def filter_qwen_annotations_for_fim(
    qwen_annotations: list[dict[str, Any]],
    labels: list[int],
) -> list[dict[str, Any]]:
    """Keep edges consistent with FIM completion semantics.

    The observed context is the user prompt around [MASK], i.e. prefix + suffix;
    the answer is the assistant completion.  Valid directed edges are:

      * context -> context, in prompt order
      * context -> completion, including suffix -> completion after ChatML remap
      * completion -> completion, in completion order

    The only invalid class is completion -> context because completion is never
    available when reasoning about the prompt context.
    """
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    seq_len = len(labels)
    for ann in qwen_annotations:
        try:
            src = int(ann["token_i_idx"])
            dst = int(ann["token_j_idx"])
        except Exception:
            continue
        subtype = str(ann.get("subtype", "") or "")
        if not (0 <= src < dst < seq_len):
            continue
        src_is_completion = labels[src] != IGNORE_INDEX
        dst_is_completion = labels[dst] != IGNORE_INDEX
        if src_is_completion and not dst_is_completion:
            continue
        key = (src, dst, subtype)
        if key in seen:
            continue
        seen.add(key)
        out.append({**ann, "token_i_idx": src, "token_j_idx": dst})
    return out


def qwen_to_attention_edges(qwen_annotations: list[dict[str, Any]], seq_len: int) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for ann in qwen_annotations:
        try:
            src = int(ann["token_i_idx"])
            dst = int(ann["token_j_idx"])
        except Exception:
            continue
        subtype = str(ann.get("subtype", "") or "")
        if not (0 <= src < dst < seq_len):
            continue
        key = (src, dst, subtype)
        if key in seen:
            continue
        seen.add(key)
        edges.append({"src": src, "dst": dst, "subtype": subtype})
    return edges


VALID_REASONS = {"bracket", "defuse", "call", "return", "type", "dataflow", "semantic", "api"}




def extract_chat_message_text(response: Any) -> str:
    try:
        msg = response.choices[0].message
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
    except Exception:
        pass
    return str(response)


def parse_json_payload(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


PAIR_KEYS = ("pairs", "edges", "annotations", "correlations", "dependencies", "links")
SRC_KEYS = ("i", "src", "source", "from", "token_i_idx", "token_i_index", "source_idx")
DST_KEYS = ("j", "dst", "target", "to", "token_j_idx", "token_j_index", "target_idx")
REASON_KEYS = ("reason", "subtype", "type", "label", "relation")
IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")


def extract_pair_items(parsed: Any) -> list[Any]:
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict):
        return []
    for key in PAIR_KEYS:
        value = parsed.get(key)
        if isinstance(value, list):
            return value
    return []


def first_present(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def normalize_token_surface(surface: str) -> str:
    text = str(surface).strip()
    if text.startswith(".") and len(text) > 1:
        text = text[1:]
    return text


def is_meaningful_bridge_token(surface: str) -> bool:
    text = normalize_token_surface(surface)
    if not text:
        return False
    if IDENT_RE.match(text):
        return True
    if re.match(r"^\d+(?:\.\d+)?$", text):
        return True
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return len(text) > 2
    return False


def local_completion_bridge_edges(
    *,
    simple_tokens: list[Any],
    correlations: list[TokenCorrelation],
    completion_indices: list[int],
    code_indices: set[int] | None,
) -> list[TokenCorrelation]:
    completion_set = set(completion_indices)
    if not completion_set:
        return []
    allowed_context = code_indices if code_indices is not None else set(range(len(simple_tokens)))
    context_indices = [i for i in sorted(allowed_context) if i not in completion_set]

    existing = {(int(c.token_i_idx), int(c.token_j_idx), str(c.subtype)) for c in correlations}
    context_by_surface: dict[str, list[int]] = {}
    for idx in context_indices:
        norm = normalize_token_surface(simple_tokens[idx].surface)
        if is_meaningful_bridge_token(norm):
            context_by_surface.setdefault(norm, []).append(idx)

    new_edges: list[TokenCorrelation] = []
    for dst in completion_indices:
        norm = normalize_token_surface(simple_tokens[dst].surface)
        if not is_meaningful_bridge_token(norm):
            continue
        matches = context_by_surface.get(norm, [])
        if not matches:
            continue
        # Prefer nearby code mentions but keep both prefix and suffix usable.
        matches = sorted(matches, key=lambda src: (abs(src - dst), src))[:4]
        for src in matches:
            subtype = "api" if norm and norm[0].isupper() else "dataflow"
            key = (src, dst, subtype)
            if key in existing:
                continue
            existing.add(key)
            new_edges.append(
                TokenCorrelation(
                    token_i=simple_tokens[src].surface,
                    token_j=simple_tokens[dst].surface,
                    source="LocalCompletionBridge",
                    subtype=subtype,
                    token_i_idx=src,
                    token_j_idx=dst,
                )
            )

    for src, dst in zip(completion_indices, completion_indices[1:]):
        key = (src, dst, "semantic")
        if key in existing:
            continue
        existing.add(key)
        new_edges.append(
            TokenCorrelation(
                token_i=simple_tokens[src].surface,
                token_j=simple_tokens[dst].surface,
                source="LocalCompletionBridge",
                subtype="semantic",
                token_i_idx=src,
                token_j_idx=dst,
            )
        )
    return new_edges


def dedupe_correlations(correlations: list[TokenCorrelation]) -> list[TokenCorrelation]:
    out: list[TokenCorrelation] = []
    seen: set[tuple[int, int, str]] = set()
    for corr in correlations:
        subtype = corr.subtype if corr.subtype in VALID_REASONS else "semantic"
        if subtype == "dataflow":
            src_norm = normalize_token_surface(str(corr.token_i))
            dst_norm = normalize_token_surface(str(corr.token_j))
            if src_norm and src_norm == dst_norm and IDENT_RE.match(src_norm):
                subtype = "defuse"
        key = (int(corr.token_i_idx), int(corr.token_j_idx), subtype)
        if key in seen or key[0] == key[1]:
            continue
        seen.add(key)
        out.append(
            TokenCorrelation(
                token_i=corr.token_i,
                token_j=corr.token_j,
                source=corr.source,
                subtype=subtype,
                token_i_idx=int(corr.token_i_idx),
                token_j_idx=int(corr.token_j_idx),
            )
        )
    return out


def annotate_oneshot_fim(
    *,
    filled: str,
    simple_tokens: list[Any],
    language: str,
    ts_code: str | None,
    ts_char_offset: int,
    completion_indices: list[int],
    code_indices: set[int] | None,
    max_edges: int,
) -> list[TokenCorrelation]:
    """Fast GraphSignal annotation: local structural pass + one LLM JSON call."""
    indexed = {i: tok.surface for i, tok in enumerate(simple_tokens)}
    completion_set = set(completion_indices)
    code_index_set = set(code_indices) if code_indices is not None else set(indexed)
    context_indices = [i for i in indexed if i not in completion_set and i in code_index_set]

    syntactic_tool = SyntacticCheckerTool()
    parse_text = ts_code if ts_code is not None else filled
    structural_raw = syntactic_tool.get_edges(
        parse_text,
        simple_tokens,
        language,
        char_offset=ts_char_offset,
    )
    correlations: list[TokenCorrelation] = [
        TokenCorrelation(
            token_i=simple_tokens[e.token_i_idx].surface,
            token_j=simple_tokens[e.token_j_idx].surface,
            source="Structural",
            subtype=e.reason,
            token_i_idx=e.token_i_idx,
            token_j_idx=e.token_j_idx,
        )
        for e in structural_raw
        if 0 <= e.token_i_idx < len(simple_tokens)
        and 0 <= e.token_j_idx < len(simple_tokens)
        and e.token_i_idx != e.token_j_idx
    ]

    structural_seed = [
        [int(c.token_i_idx), int(c.token_j_idx), str(c.subtype)]
        for c in correlations
    ]
    context_preview = [[int(i), indexed[i]] for i in context_indices]
    completion_preview = [[int(i), indexed[i]] for i in completion_indices]

    system = (
        f"You are a {language} token-level dependency annotator for fill-in-the-middle code completion. "
        "Return JSON only, no markdown. Schema: "
        "{\"pairs\":[{\"i\":0,\"j\":1,\"reason\":\"dataflow\"}]}. "
        "Allowed reasons: bracket, defuse, call, return, type, dataflow, semantic, api. "
        "An edge i->j means token i helps predict token j. "
        "The most important edges are context->completion. Also include context->context and completion->completion. "
        "Do not emit completion->context. Do not repeat seeded_structural_edges."
    )
    payload = {
        "language": language,
        "code_with_completion_filled_for_reference": filled,
        "context_tokens": context_preview,
        "completion_tokens": completion_preview,
        "seeded_structural_edges": structural_seed,
        "task": (
            "Find high-confidence missing token dependency edges. Prioritize context->completion edges: "
            "prefix or suffix context tokens that determine each hidden completion token. Then add useful "
            "context->context and completion->completion edges. Return at most " + str(max_edges) + " pairs."
        ),
    }
    user = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    client = get_thread_local_openai_client()
    model = os.environ.get("ANNOTATE_MODEL", "gpt-4o-mini")

    response = None
    try:
        response = _rate_limited_chat_completion(
            client,
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=int(os.environ.get("ANNOTATE_MAX_TOKENS", "2048")),
            response_format={"type": "json_object"},
        )
    except Exception:
        response = _rate_limited_chat_completion(
            client,
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=int(os.environ.get("ANNOTATE_MAX_TOKENS", "2048")),
        )

    raw = extract_chat_message_text(response)
    parsed = parse_json_payload(raw)
    pairs = extract_pair_items(parsed)
    teacher_added = 0

    for pair in pairs:
        try:
            if isinstance(pair, dict):
                i = int(first_present(pair, SRC_KEYS))
                j = int(first_present(pair, DST_KEYS))
                reason_value = first_present(pair, REASON_KEYS)
                reason = str(reason_value or "semantic").lower()
            else:
                i = int(pair[0])
                j = int(pair[1])
                reason = str(pair[2] if len(pair) > 2 else "semantic").lower()
        except Exception:
            continue
        if not (0 <= i < len(simple_tokens) and 0 <= j < len(simple_tokens)) or i == j:
            continue
        if reason not in VALID_REASONS:
            reason = "semantic"
        correlations.append(
            TokenCorrelation(
                token_i=simple_tokens[i].surface,
                token_j=simple_tokens[j].surface,
                source="OneShotTeacher",
                subtype=reason,
                token_i_idx=i,
                token_j_idx=j,
            )
        )
        teacher_added += 1
        if teacher_added >= max_edges:
            break

    deduped = dedupe_correlations(correlations)
    has_completion_edge = any(
        int(c.token_j_idx) in completion_set and int(c.token_i_idx) != int(c.token_j_idx)
        for c in deduped
    )
    if not has_completion_edge:
        deduped = dedupe_correlations(
            deduped
            + local_completion_bridge_edges(
                simple_tokens=simple_tokens,
                correlations=deduped,
                completion_indices=completion_indices,
                code_indices=code_index_set,
            )
        )
    return deduped


def annotate_row(row: dict[str, Any], tokenizer: Any, max_len: int, max_rounds: int, annotation_mode: str, max_teacher_edges: int) -> dict[str, Any] | None:
    target = str(row.get("target", row.get("fim_completion", "")))
    user_content = get_user_content(row)
    if not target or MASK_TOKEN not in user_content:
        return None
    messages = build_messages(row)
    packed = binarize_messages(messages, tokenizer, max_len)
    if packed is None:
        return None

    filled, mask_pos, target_end = build_filled_instruction(user_content, target)
    simple_tokens = tokenize_code_for_annotation(filled)
    m = INCOMPLETE_CODE_RE.search(filled)
    if m:
        target_indices = get_token_indices_in_span(simple_tokens, m.start(1), m.end(1))
        ts_code = m.group(1)
        ts_char_offset = m.start(1)
    else:
        target_indices = None
        ts_code = None
        ts_char_offset = 0

    completion_indices = sorted(get_token_indices_in_span(simple_tokens, mask_pos, target_end))
    if not completion_indices:
        return None
    completion_preview = ", ".join(f"{i}:{simple_tokens[i].surface}" for i in completion_indices[:80])
    extra_task_instruction = (
        "This is a fill-in-the-middle code completion annotation task. "
        "The original prompt contained a [MASK]. The following token indices are the hidden completion tokens: "
        f"{completion_preview}. "
        "The observed context consists of all non-completion tokens before and after [MASK]. "
        "The most important edges are context -> completion: prefix or suffix context tokens that help predict each completion token. "
        "Also emit context -> context edges for normal left-to-right dependencies inside the prompt context, "
        "and completion -> completion edges for normal left-to-right dependencies inside the hidden completion. "
        "Do not emit completion -> context edges. Be especially exhaustive for context -> completion dataflow, call, api, type, return, and semantic edges."
    )
    language = normalize_language(row.get("language", "Go"))
    if annotation_mode == "oneshot":
        correlations = annotate_oneshot_fim(
            filled=filled,
            simple_tokens=simple_tokens,
            language=language,
            ts_code=ts_code,
            ts_char_offset=ts_char_offset,
            completion_indices=completion_indices,
            code_indices=set(target_indices) if target_indices is not None else None,
            max_edges=max_teacher_edges,
        )
    else:
        agent = AnnotatorAgent(language=language, max_rounds=max_rounds)
        annotate_kwargs = {
            "target_indices": target_indices,
            "ts_code": ts_code,
            "ts_char_offset": ts_char_offset,
        }
        if "extra_task_instruction" in inspect.signature(agent.annotate).parameters:
            annotate_kwargs["extra_task_instruction"] = extra_task_instruction
        correlations = agent.annotate(filled, simple_tokens, **annotate_kwargs)

    completion_set = set(completion_indices)
    has_completion_edge = any(
        int(c.token_j_idx) in completion_set and int(c.token_i_idx) != int(c.token_j_idx)
        for c in correlations
    )
    if not has_completion_edge:
        correlations = dedupe_correlations(
            list(correlations)
            + local_completion_bridge_edges(
                simple_tokens=simple_tokens,
                correlations=list(correlations),
                completion_indices=completion_indices,
                code_indices=set(target_indices) if target_indices is not None else None,
            )
        )

    tokens = [to_dict(t) for t in simple_tokens]
    annotations = [to_dict(c) for c in correlations]
    qwen_tokens, raw_qwen_annotations = map_to_qwen_annotations(
        tokenizer=tokenizer,
        messages=messages,
        filled_tokens=tokens,
        annotations=annotations,
        mask_pos=mask_pos,
        target_len=target_end - mask_pos,
        completion_indices=set(completion_indices),
    )
    qwen_annotations = filter_qwen_annotations_for_fim(raw_qwen_annotations, packed["label"])
    qwen_annotations = drop_mask_token_annotations(qwen_annotations, tokenizer=tokenizer, messages=messages)
    attention_edges = qwen_to_attention_edges(qwen_annotations, len(packed["input_ids"]))
    out = {
        "input_ids": packed["input_ids"],
        "label": packed["label"],
        "length": packed["length"],
        "attention_edges": attention_edges,
        "uid": row.get("uid"),
        "language": row.get("language", "Go"),
        "raw_id": row.get("raw_id", row.get("uid")),
        "annotation_meta": {
            "annotator": "src.annotate.oneshot" if annotation_mode == "oneshot" else "src.annotate.AnnotatorAgent",
            "annotation_scope": "fim_context_context_completion",
            "valid_edge_classes": ["context->context", "context->completion", "completion->completion"],
            "dropped_edge_class": "completion->context",
            "num_raw_qwen_annotations": len(raw_qwen_annotations),
            "num_fim_qwen_annotations": len(qwen_annotations),
            "num_causal_attention_edges": len(attention_edges),
        },
        "tokens": tokens,
        "annotations": annotations,
        "qwen_tokens": qwen_tokens,
        "qwen_annotations": qwen_annotations,
        "raw_qwen_annotations": raw_qwen_annotations,
    }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate ChatML Go-single data with src.annotate.AnnotatorAgent.")
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--model_name_or_path", default="/mnt/nvme0n1/wenhao/models/Empirical-Influence-Function/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--model_max_length", type=int, default=4096)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--max_rounds", type=int, default=6)
    p.add_argument("--annotation_mode", choices=["oneshot", "agent"], default="oneshot")
    p.add_argument("--max_teacher_edges", type=int, default=128)
    p.add_argument("--annotation_cache_path", default="")
    p.add_argument("--seed_compact_path", default="")
    p.add_argument("--flush_every", type=int, default=20)
    p.add_argument("--overwrite_cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_path)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    cache = load_cache(args.annotation_cache_path)
    seed = load_seed_compact(args.seed_compact_path)
    cache_lock = threading.Lock()

    output_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row_key(row)
        if not args.overwrite_cache and key in seed:
            output_by_key[key] = copy.deepcopy(seed[key])
        elif not args.overwrite_cache and key in cache and cache[key].get("record"):
            output_by_key[key] = copy.deepcopy(cache[key]["record"])

    todo = [row for row in rows if row_key(row) not in output_by_key]
    print(f"Applied cached/seed annotations: {len(output_by_key)}/{len(rows)}")
    print(f"Annotation plan: {len(todo)} rows, workers={args.num_workers}, mode={args.annotation_mode}, max_rounds={args.max_rounds}")

    done_since_flush = 0

    def work(row: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
        key = row_key(row)
        try:
            return key, annotate_row(row, tokenizer, args.model_max_length, args.max_rounds, args.annotation_mode, args.max_teacher_edges), None
        except Exception as exc:
            return key, None, repr(exc)

    if args.num_workers <= 1:
        iterator: Any = map(work, todo)
    else:
        pool = ThreadPoolExecutor(max_workers=args.num_workers)
        futures = [pool.submit(work, row) for row in todo]
        iterator = (future.result() for future in as_completed(futures))

    failures: dict[str, str] = {}
    try:
        progress_iter = tqdm.tqdm(iterator, total=len(todo), desc="src.annotate")
        for key, record, err in progress_iter:
            if record is None:
                failures[key] = err or "unknown"
            else:
                output_by_key[key] = record
                with cache_lock:
                    cache[key] = {"key": key, "record": record}
                done_since_flush += 1
            if done_since_flush >= max(1, args.flush_every):
                save_cache(cache, args.annotation_cache_path)
                done_since_flush = 0
    finally:
        if args.num_workers > 1:
            pool.shutdown(wait=False, cancel_futures=True)

    save_cache(cache, args.annotation_cache_path)

    output = [output_by_key[row_key(row)] for row in rows if row_key(row) in output_by_key]
    write_jsonl_atomic(output, args.output_path)
    if failures:
        fail_path = Path(args.output_path).with_suffix(Path(args.output_path).suffix + ".failures.json")
        fail_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Failures: {len(failures)} -> {fail_path}")
    print(f"Done: {len(output)}/{len(rows)} annotated")
    return


if __name__ == "__main__":
    main()
