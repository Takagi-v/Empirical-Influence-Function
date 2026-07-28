from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterable

from tree_sitter import Language, Parser
import tree_sitter_go as tsgo


LEX_RE = re.compile(
    r"`[^`]*`"
    r"|\"(?:\\.|[^\"\\])*\""
    r"|'(?:\\.|[^'\\])*'"
    r"|[A-Za-z_]\w*"
    r"|\d+(?:\.\d+)?"
    r"|==|!=|<=|>=|:=|&&|\|\||\+\+|--|<<|>>"
    r"|[-+*/%&|^!<>.=,:;(){}\[\]]"
)

IDENTIFIER_NODE_TYPES = {
    "identifier",
    "field_identifier",
    "type_identifier",
    "package_identifier",
    "label_name",
}
STRING_NODE_TYPES = {
    "interpreted_string_literal",
    "raw_string_literal",
    "rune_literal",
}
NUMERIC_NODE_TYPES = {
    "int_literal",
    "float_literal",
    "imaginary_literal",
}
STATEMENT_NODE_TYPES = {
    "assignment_statement",
    "short_var_declaration",
    "return_statement",
    "expression_statement",
    "if_statement",
    "for_statement",
    "range_clause",
    "switch_statement",
    "type_switch_statement",
    "case_clause",
    "communication_case",
    "go_statement",
    "defer_statement",
    "send_statement",
    "inc_statement",
    "dec_statement",
}
CONTROL_NODE_TYPES = {
    "if_statement",
    "for_statement",
    "switch_statement",
    "type_switch_statement",
    "select_statement",
    "case_clause",
    "communication_case",
}
CHAIN_NODE_TYPES = {
    "selector_expression",
    "call_expression",
    "index_expression",
    "slice_expression",
    "qualified_type",
}


@dataclass(frozen=True)
class TokenSpan:
    seq_idx: int
    token: str
    start: int
    end: int


@dataclass(frozen=True)
class LexToken:
    text: str
    seq_idx: int | None
    start: int
    end: int


@dataclass
class CodeContext:
    code: str
    prompt_segments: list[tuple[int, int, int]]
    completion_start: int


@dataclass
class PairJudgment:
    mapped: bool
    strict_supported: bool
    loose_supported: bool
    reason: str
    source_node_type: str | None = None
    target_node_type: str | None = None
    source_node_text: str | None = None
    target_node_text: str | None = None


def _new_go_parser() -> Parser:
    language = Language(tsgo.language())
    try:
        return Parser(language)
    except TypeError:
        parser = Parser()
        parser.language = language
        return parser


def _char_to_byte(text: str, char_idx: int) -> int:
    return len(text[:char_idx].encode("utf-8"))


def _byte_text(source: bytes, node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _token_spans(tokens: list[str], start_idx: int, *, stop_at_im_end: bool) -> tuple[str, list[TokenSpan]]:
    spans: list[TokenSpan] = []
    parts: list[str] = []
    pos = 0
    for seq_idx, token in enumerate(tokens[start_idx:], start=start_idx):
        if stop_at_im_end and token == "<|im_end|>":
            break
        start = pos
        pos += len(token)
        spans.append(TokenSpan(seq_idx=seq_idx, token=token, start=start, end=pos))
        parts.append(token)
    return "".join(parts), spans


def _valid_offsets(offsets) -> bool:
    return (
        isinstance(offsets, list)
        and all(
            isinstance(item, list | tuple)
            and len(item) == 2
            and isinstance(item[0], int)
            and isinstance(item[1], int)
            for item in offsets
        )
    )


def _align_tokens_ignoring_whitespace(full_text: str, tokens: list[str]) -> list[list[int]] | None:
    raw_positions = [
        (idx, ch)
        for idx, ch in enumerate(full_text)
        if not ch.isspace()
    ]
    offsets: list[list[int]] = []
    cursor = 0
    previous_end = 0
    for token in tokens:
        token_chars = [ch for ch in token if not ch.isspace()]
        if not token_chars:
            offsets.append([previous_end, previous_end])
            continue

        start = None
        end = None
        for ch in token_chars:
            while cursor < len(raw_positions) and raw_positions[cursor][1] != ch:
                cursor += 1
            if cursor >= len(raw_positions):
                return None
            raw_idx = raw_positions[cursor][0]
            if start is None:
                start = raw_idx
            end = raw_idx + 1
            cursor += 1
        if start is None or end is None:
            offsets.append([previous_end, previous_end])
            continue
        offsets.append([start, end])
        previous_end = end
    return offsets


def _prefix_end_from_offsets(offsets: list[list[int]], prompt_len: int) -> int | None:
    if prompt_len <= 0:
        return 0
    if prompt_len > len(offsets):
        return None
    return int(offsets[prompt_len - 1][1])


def _stop_end_from_offsets(
    full_text: str,
    tokens: list[str],
    offsets: list[list[int]],
    start_idx: int,
    *,
    stop_at_im_end: bool,
) -> int:
    end = len(full_text)
    if not stop_at_im_end:
        return end
    for seq_idx in range(start_idx, min(len(tokens), len(offsets))):
        if tokens[seq_idx] == "<|im_end|>":
            return int(offsets[seq_idx][0])
    return end


def _spans_from_offsets(
    full_text: str,
    tokens: list[str],
    offsets: list[list[int]],
    start_idx: int,
    end_idx: int,
    text_start: int,
) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    for seq_idx in range(start_idx, min(end_idx, len(tokens), len(offsets))):
        abs_start, abs_end = int(offsets[seq_idx][0]), int(offsets[seq_idx][1])
        if abs_end <= abs_start:
            continue
        rel_start = abs_start - text_start
        rel_end = abs_end - text_start
        if rel_end <= 0:
            continue
        token_text = full_text[abs_start:abs_end]
        spans.append(
            TokenSpan(
                seq_idx=seq_idx,
                token=token_text,
                start=max(0, rel_start),
                end=max(0, rel_end),
            )
        )
    return spans


def _build_text_and_spans(
    tokens: list[str],
    prompt_len: int,
    *,
    full_text: str | None,
    offsets: list[list[int]] | None,
    stop_at_im_end: bool,
) -> tuple[str, list[TokenSpan], str, list[TokenSpan]] | None:
    if not full_text or not offsets or not _valid_offsets(offsets):
        return None
    prompt_end = _prefix_end_from_offsets(offsets, prompt_len)
    if prompt_end is None:
        return None
    completion_end = _stop_end_from_offsets(
        full_text,
        tokens,
        offsets,
        prompt_len,
        stop_at_im_end=stop_at_im_end,
    )
    prompt_text = full_text[:prompt_end]
    completion_text = full_text[prompt_end:completion_end]
    prompt_spans = _spans_from_offsets(
        full_text,
        tokens,
        offsets,
        0,
        prompt_len,
        0,
    )
    completion_spans = _spans_from_offsets(
        full_text,
        tokens,
        offsets,
        prompt_len,
        len(tokens),
        prompt_end,
    )
    return prompt_text, prompt_spans, completion_text, completion_spans


def _lex_tokens(text: str, spans: list[TokenSpan] | None = None) -> list[LexToken]:
    lexed: list[LexToken] = []
    for match in LEX_RE.finditer(text):
        seq_idx = None
        if spans is not None:
            pos = match.start()
            for span in spans:
                if span.start <= pos < span.end:
                    seq_idx = span.seq_idx
                    break
        lexed.append(LexToken(match.group(0), seq_idx, match.start(), match.end()))
    return lexed


def _first_lexical_mismatch(
    generated_text: str,
    generated_spans: list[TokenSpan],
    reference_text: str,
) -> tuple[int, LexToken | None, str, str] | None:
    generated_lex = _lex_tokens(generated_text, generated_spans)
    reference_lex = _lex_tokens(reference_text, None)
    for ordinal in range(max(len(generated_lex), len(reference_lex))):
        generated_value = generated_lex[ordinal].text if ordinal < len(generated_lex) else "<EOS_GEN>"
        reference_value = reference_lex[ordinal].text if ordinal < len(reference_lex) else "<EOS_REF>"
        if generated_value != reference_value:
            generated_token = generated_lex[ordinal] if ordinal < len(generated_lex) else None
            return ordinal, generated_token, generated_value, reference_value
    return None


def _valid_code_fragment(fragment: str) -> bool:
    stripped = fragment.strip()
    if not stripped:
        return False
    return stripped.lower() not in {"not exist", "not exists", "none", "null"}


def _find_prompt_sections(prompt_text: str) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    before_match = re.search(
        r"The code snippets before the function is ([\s\S]*?) And here is the function you are asked to complete ",
        prompt_text,
    )
    before_range = None
    if before_match and _valid_code_fragment(before_match.group(1)):
        before_range = before_match.span(1)

    task_match = re.search(
        r"And here is the function you are asked to complete ([\s\S]*?) Ensure that only missing codes marked as <MID> are returned",
        prompt_text,
    )
    skeleton_range = None
    if task_match:
        task = task_match.group(1)
        mid = task.find("<MID>")
        if mid >= 0:
            rel_start = task.rfind("\nfunc", 0, mid)
            if rel_start < 0:
                rel_start = task.rfind("func", 0, mid)
            if rel_start >= 0:
                # Do not strip here; prompt offsets must stay exact.
                skeleton_range = (
                    task_match.start(1) + rel_start,
                    task_match.end(1),
                )
    return before_range, skeleton_range


def _build_code_context(prompt_text: str, completion_text: str) -> CodeContext | None:
    before_range, skeleton_range = _find_prompt_sections(prompt_text)
    if skeleton_range is None:
        return None

    prefix = "package main\n\n"
    parts: list[str] = [prefix]
    prompt_segments: list[tuple[int, int, int]] = []

    def append_prompt_segment(prompt_start: int, prompt_end: int) -> None:
        if prompt_end <= prompt_start:
            return
        assembled_start = sum(len(part) for part in parts)
        segment = prompt_text[prompt_start:prompt_end]
        parts.append(segment)
        prompt_segments.append((prompt_start, prompt_end, assembled_start))

    if before_range is not None:
        append_prompt_segment(*before_range)
        parts.append("\n\n")

    skeleton = prompt_text[skeleton_range[0]:skeleton_range[1]]
    mid = skeleton.find("<MID>")
    if mid < 0:
        return None

    append_prompt_segment(skeleton_range[0], skeleton_range[0] + mid)
    completion_start = sum(len(part) for part in parts)
    parts.append(completion_text)
    append_prompt_segment(
        skeleton_range[0] + mid + len("<MID>"),
        skeleton_range[1],
    )

    return CodeContext(
        code="".join(parts),
        prompt_segments=prompt_segments,
        completion_start=completion_start,
    )


def _build_prompt_spans(tokens: list[str], prompt_len: int) -> tuple[str, list[TokenSpan]]:
    text_parts: list[str] = []
    spans: list[TokenSpan] = []
    pos = 0
    for seq_idx, token in enumerate(tokens[:prompt_len]):
        start = pos
        pos += len(token)
        text_parts.append(token)
        spans.append(TokenSpan(seq_idx=seq_idx, token=token, start=start, end=pos))
    return "".join(text_parts), spans


def _trim_token_to_lexical(span: TokenSpan) -> tuple[int, int] | None:
    for match in LEX_RE.finditer(span.token):
        return span.start + match.start(), span.start + match.end()
    return None


def _map_prompt_char_to_code(context: CodeContext, start: int, end: int) -> tuple[int, int] | None:
    for prompt_start, prompt_end, code_start in context.prompt_segments:
        if prompt_start <= start and end <= prompt_end:
            rel_start = start - prompt_start
            rel_end = end - prompt_start
            return code_start + rel_start, code_start + rel_end
    return None


def _map_seq_token_to_code_chars(
    seq_idx: int,
    *,
    prompt_len: int,
    prompt_spans: dict[int, TokenSpan],
    completion_spans: dict[int, TokenSpan],
    context: CodeContext,
) -> tuple[int, int] | None:
    if seq_idx >= prompt_len:
        span = completion_spans.get(seq_idx)
        if span is None:
            return None
        trimmed = _trim_token_to_lexical(span)
        if trimmed is None:
            return None
        return (
            context.completion_start + trimmed[0],
            context.completion_start + trimmed[1],
        )

    span = prompt_spans.get(seq_idx)
    if span is None:
        return None
    trimmed = _trim_token_to_lexical(span)
    if trimmed is None:
        return None
    return _map_prompt_char_to_code(context, trimmed[0], trimmed[1])


def _node_at(root, start_byte: int, end_byte: int):
    if end_byte <= start_byte:
        end_byte = start_byte + 1
    try:
        return root.descendant_for_byte_range(start_byte, max(start_byte, end_byte - 1))
    except Exception:
        return None


def _meaningful_node(node):
    cur = node
    while cur is not None and cur.parent is not None:
        if cur.type in IDENTIFIER_NODE_TYPES | STRING_NODE_TYPES | NUMERIC_NODE_TYPES:
            return cur
        if cur.type in {".", ",", ":", ";", "(", ")", "{", "}", "[", "]"}:
            cur = cur.parent
            continue
        if cur.is_named:
            return cur
        cur = cur.parent
    return cur


def _ancestor_of_type(node, types: set[str]):
    cur = node
    while cur is not None:
        if cur.type in types:
            return cur
        cur = cur.parent
    return None


def _ancestor_chain_node(node):
    return _ancestor_of_type(node, CHAIN_NODE_TYPES)


def _enclosing_statement(node):
    return _ancestor_of_type(node, STATEMENT_NODE_TYPES)


def _node_identifier_text(source: bytes, node) -> str | None:
    if node is None:
        return None
    cur = node
    while cur is not None:
        if cur.type in IDENTIFIER_NODE_TYPES:
            return _byte_text(source, cur)
        if cur.type in STRING_NODE_TYPES | NUMERIC_NODE_TYPES:
            return _byte_text(source, cur)
        cur = cur.parent
    return None


def _identifier_set(source: bytes, node) -> set[str]:
    result: set[str] = set()

    def visit(cur) -> None:
        if cur.type in IDENTIFIER_NODE_TYPES:
            result.add(_byte_text(source, cur))
        for child in cur.children:
            visit(child)

    if node is not None:
        visit(node)
    return result


def _source_is_control_dependency(source_node, target_node) -> bool:
    target_statement = _enclosing_statement(target_node)
    cur = target_node.parent
    while cur is not None:
        if cur.type in CONTROL_NODE_TYPES:
            if (
                source_node.start_byte >= cur.start_byte
                and source_node.end_byte <= cur.end_byte
                and target_statement is not None
                and not (
                    source_node.start_byte >= target_statement.start_byte
                    and source_node.end_byte <= target_statement.end_byte
                )
            ):
                return True
        cur = cur.parent
    return False


def _judge_pair(code: str, parser: Parser, source_chars: tuple[int, int] | None, target_chars: tuple[int, int] | None) -> PairJudgment:
    if source_chars is None or target_chars is None:
        return PairJudgment(False, False, False, "unmapped")

    source_bytes = (
        _char_to_byte(code, source_chars[0]),
        _char_to_byte(code, source_chars[1]),
    )
    target_bytes = (
        _char_to_byte(code, target_chars[0]),
        _char_to_byte(code, target_chars[1]),
    )
    tree = parser.parse(code.encode("utf-8"))
    source_blob = code.encode("utf-8")
    source_node = _meaningful_node(_node_at(tree.root_node, *source_bytes))
    target_node = _meaningful_node(_node_at(tree.root_node, *target_bytes))
    if source_node is None or target_node is None:
        return PairJudgment(False, False, False, "node_missing")

    source_text = _node_identifier_text(source_blob, source_node)
    target_text = _node_identifier_text(source_blob, target_node)
    source_chain = _ancestor_chain_node(source_node)
    target_chain = _ancestor_chain_node(target_node)
    source_chain_ids = _identifier_set(source_blob, source_chain)
    target_chain_ids = _identifier_set(source_blob, target_chain)
    source_stmt = _enclosing_statement(source_node)
    target_stmt = _enclosing_statement(target_node)

    strict = False
    loose = False
    reason = "unsupported"
    if source_text and target_text and source_text == target_text:
        strict = loose = True
        reason = "same_identifier"
    elif source_text and source_text in target_chain_ids:
        strict = loose = True
        reason = "source_in_target_selector_chain"
    elif target_text and target_text in source_chain_ids:
        strict = loose = True
        reason = "target_in_source_selector_chain"
    elif source_chain_ids and target_chain_ids and source_chain_ids == target_chain_ids:
        strict = loose = True
        reason = "same_selector_chain"
    elif (
        source_stmt is not None
        and target_stmt is not None
        and source_stmt.id == target_stmt.id
    ):
        loose = True
        reason = "same_statement"
    elif _source_is_control_dependency(source_node, target_node):
        loose = True
        reason = "control_dependency"

    return PairJudgment(
        mapped=True,
        strict_supported=strict,
        loose_supported=loose,
        reason=reason,
        source_node_type=source_node.type,
        target_node_type=target_node.type,
        source_node_text=source_text,
        target_node_text=target_text,
    )


def _safe_float(value) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def _feature_files(feature_dir: str) -> list[str]:
    def key(path: str) -> tuple[int, str]:
        match = re.search(r"test(\d+)_feature\.json$", path)
        return (int(match.group(1)) if match else 10**9, path)

    return sorted(glob(os.path.join(feature_dir, "*_feature.json")), key=key)


def _rank_source_items(items: list[dict], source_ranking: str) -> list[dict]:
    if source_ranking == "stored":
        return list(items)
    if source_ranking == "rank_score":
        return sorted(
            items,
            key=lambda item: (
                _safe_float(item.get("rank_score")),
                _safe_float(item.get("alti_saliency")),
                -int(item.get("source_token_index", 0)),
            ),
            reverse=True,
        )
    if source_ranking == "alti_saliency":
        return sorted(
            items,
            key=lambda item: (
                _safe_float(item.get("alti_saliency")),
                _safe_float(item.get("rank_score")),
                -int(item.get("source_token_index", 0)),
            ),
            reverse=True,
        )
    raise ValueError(f"Unsupported source ranking: {source_ranking}")


def analyze_file(
    path: str,
    parser: Parser,
    k_values: Iterable[int],
    source_ranking: str,
) -> dict | None:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    meta = payload["experiment_meta"]
    baseline = payload["test_sample_baseline"]
    if payload.get("status") == "skipped" or baseline.get("status") == "skipped":
        return {
            "status": baseline.get("skip_reason") or payload.get("skip_reason") or "skipped",
            "path": path,
            "test_sample_index": int(meta["test_sample_index"]),
            "task_id": meta.get("task_id"),
        }
    required_baseline_keys = (
        "generated_full_tokens",
        "ground_truth_full_tokens",
    )
    missing_baseline_keys = [
        key for key in required_baseline_keys
        if key not in baseline
    ]
    if missing_baseline_keys or meta.get("prompt_len") is None:
        return {
            "status": "missing_baseline_context",
            "path": path,
            "test_sample_index": int(meta["test_sample_index"]),
            "task_id": meta.get("task_id"),
            "missing_keys": missing_baseline_keys,
        }
    generated_tokens = baseline["generated_full_tokens"]
    reference_tokens = baseline["ground_truth_full_tokens"]
    prompt_len = int(meta["prompt_len"])

    generated_full_text = baseline.get("raw_generated_full_text") or baseline.get("generated_full_text")
    generated_offsets = (
        _align_tokens_ignoring_whitespace(generated_full_text, generated_tokens)
        if baseline.get("raw_generated_full_text") and generated_full_text
        else None
    )
    if generated_offsets is None:
        generated_offsets = baseline.get("generated_full_token_offsets")

    generated_text_spans = _build_text_and_spans(
        generated_tokens,
        prompt_len,
        full_text=generated_full_text,
        offsets=generated_offsets,
        stop_at_im_end=True,
    )
    if generated_text_spans is None:
        prompt_text, prompt_span_list = _build_prompt_spans(generated_tokens, prompt_len)
        completion_text, completion_span_list = _token_spans(
            generated_tokens,
            prompt_len,
            stop_at_im_end=True,
        )
    else:
        prompt_text, prompt_span_list, completion_text, completion_span_list = generated_text_spans

    reference_full_text = baseline.get("raw_ground_truth_full_text") or baseline.get("ground_truth_full_text")
    reference_offsets = (
        _align_tokens_ignoring_whitespace(reference_full_text, reference_tokens)
        if baseline.get("raw_ground_truth_full_text") and reference_full_text
        else None
    )
    if reference_offsets is None:
        reference_offsets = baseline.get("ground_truth_full_token_offsets")

    reference_text_spans = _build_text_and_spans(
        reference_tokens,
        prompt_len,
        full_text=reference_full_text,
        offsets=reference_offsets,
        stop_at_im_end=True,
    )
    if reference_text_spans is None:
        reference_text, _ = _token_spans(reference_tokens, prompt_len, stop_at_im_end=True)
    else:
        _, _, reference_text, _ = reference_text_spans

    mismatch = _first_lexical_mismatch(completion_text, completion_span_list, reference_text)
    if mismatch is None:
        return {
            "status": "no_lexical_mismatch",
            "path": path,
            "test_sample_index": int(meta["test_sample_index"]),
            "task_id": meta.get("task_id"),
        }

    mismatch_ordinal, generated_lex, generated_value, reference_value = mismatch
    if generated_lex is None or generated_lex.seq_idx is None:
        return {
            "status": "uncovered_eos",
            "test_sample_index": int(meta["test_sample_index"]),
            "task_id": meta.get("task_id"),
            "first_mismatch_ordinal": int(mismatch_ordinal),
            "generated_lex": generated_value,
            "reference_lex": reference_value,
        }

    target_idx = int(generated_lex.seq_idx)
    row_by_target = {
        int(row["target_token_index"]): row
        for row in payload.get("feature_attribution", [])
    }
    feature_row = row_by_target.get(target_idx)
    if feature_row is None:
        return {
            "status": "uncovered_target",
            "test_sample_index": int(meta["test_sample_index"]),
            "task_id": meta.get("task_id"),
            "target_token_index": target_idx,
            "first_mismatch_ordinal": int(mismatch_ordinal),
            "generated_lex": generated_value,
            "reference_lex": reference_value,
        }

    context = _build_code_context(prompt_text, completion_text)
    if context is None:
        return {
            "status": "context_unavailable",
            "test_sample_index": int(meta["test_sample_index"]),
            "task_id": meta.get("task_id"),
            "target_token_index": target_idx,
            "generated_lex": generated_value,
            "reference_lex": reference_value,
        }

    prompt_spans = {span.seq_idx: span for span in prompt_span_list}
    completion_spans = {span.seq_idx: span for span in completion_span_list}
    target_chars = (
        context.completion_start + generated_lex.start,
        context.completion_start + generated_lex.end,
    )

    method_top = list(feature_row.get("method_top", []))
    ranked_items = _rank_source_items(method_top, source_ranking)
    source_rows = []
    for selected_rank, item in enumerate(ranked_items, start=1):
        source_idx = int(item["source_token_index"])
        source_chars = _map_seq_token_to_code_chars(
            source_idx,
            prompt_len=prompt_len,
            prompt_spans=prompt_spans,
            completion_spans=completion_spans,
            context=context,
        )
        judgment = _judge_pair(context.code, parser, source_chars, target_chars)
        source_rows.append({
            "rank": selected_rank,
            "method_rank": int(item.get("rank", len(source_rows) + 1)),
            "source_token_index": source_idx,
            "source_token": item.get("source_token"),
            "alti_saliency": _safe_float(item.get("alti_saliency")),
            "rank_score": _safe_float(item.get("rank_score")),
            "oracle_effect": _safe_float(item.get("oracle_effect")),
            "mapped": judgment.mapped,
            "strict_supported": judgment.strict_supported,
            "loose_supported": judgment.loose_supported,
            "strict_spurious": judgment.mapped and not judgment.strict_supported,
            "loose_spurious": judgment.mapped and not judgment.loose_supported,
            "reason": judgment.reason,
            "source_node_type": judgment.source_node_type,
            "target_node_type": judgment.target_node_type,
            "source_node_text": judgment.source_node_text,
            "target_node_text": judgment.target_node_text,
        })

    metrics: dict[str, float] = {}
    for k in k_values:
        top = source_rows[: int(k)]
        mapped = [row for row in top if row["mapped"]]
        total_saliency_all = sum(max(0.0, row["alti_saliency"]) for row in top)
        total_saliency = sum(max(0.0, row["alti_saliency"]) for row in mapped)
        for mode in ("strict", "loose"):
            spurious = [
                row for row in mapped
                if row[f"{mode}_spurious"]
            ]
            spurious_all = [
                row for row in top
                if not row["mapped"] or row[f"{mode}_spurious"]
            ]
            spurious_saliency = sum(max(0.0, row["alti_saliency"]) for row in spurious)
            spurious_saliency_all = sum(max(0.0, row["alti_saliency"]) for row in spurious_all)
            metrics[f"{mode}_spurious@{k}"] = (
                len(spurious_all) / len(top) if top else float("nan")
            )
            metrics[f"{mode}_spurious_mass@{k}"] = (
                spurious_saliency_all / total_saliency_all
                if total_saliency_all > 0 else float("nan")
            )
            metrics[f"{mode}_mapped@{k}"] = len(mapped)
            metrics[f"{mode}_mapped_spurious@{k}"] = (
                len(spurious) / len(mapped) if mapped else float("nan")
            )
            metrics[f"{mode}_mapped_spurious_mass@{k}"] = (
                spurious_saliency / total_saliency if total_saliency > 0 else float("nan")
            )

    return {
        "status": "ok",
        "path": path,
        "test_sample_index": int(meta["test_sample_index"]),
        "task_id": meta.get("task_id"),
        "prompt_len": prompt_len,
        "target_token_index": target_idx,
        "target_token": feature_row.get("target_token"),
        "generated_lex": generated_value,
        "reference_lex": reference_value,
        "first_mismatch_ordinal": int(mismatch_ordinal),
        "source_ranking": source_ranking,
        "source_pool": "method_top",
        "source_pool_size": len(method_top),
        "metrics": metrics,
        "top_sources": source_rows,
        "generated_completion_preview": completion_text[:500],
        "reference_completion_preview": reference_text[:500],
    }


def _mean(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def build_summary(records: list[dict], k_values: list[int]) -> dict:
    ok_records = [record for record in records if record.get("status") == "ok"]
    summary = {
        "input_count": len(records),
        "ok_count": len(ok_records),
        "status_counts": {},
        "metrics": {},
    }
    for record in records:
        status = record.get("status", "unknown")
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1

    for k in k_values:
        for mode in ("strict", "loose"):
            for metric in (
                "spurious",
                "spurious_mass",
                "mapped",
                "mapped_spurious",
                "mapped_spurious_mass",
            ):
                key = f"{mode}_{metric}@{k}"
                values = [
                    float(record["metrics"][key])
                    for record in ok_records
                    if key in record.get("metrics", {})
                ]
                summary["metrics"][key] = _mean(values)
    return summary


def write_outputs(records: list[dict], summary: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "per_sample.jsonl"), "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    candidate_path = os.path.join(output_dir, "case_candidates.tsv")
    ok_records = [record for record in records if record.get("status") == "ok"]
    metric_keys = sorted({key for record in ok_records for key in record.get("metrics", {})})
    with open(candidate_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "test_sample_index",
            "task_id",
            "target_token_index",
            "target_token",
            "generated_lex",
            "reference_lex",
            *metric_keys,
            "top5_sources",
        ])
        for record in sorted(
            ok_records,
            key=lambda r: (
                r["metrics"].get("strict_spurious@10", -1),
                r["metrics"].get("strict_spurious_mass@10", -1),
            ),
            reverse=True,
        ):
            top5 = "; ".join(
                f"{row['rank']}:{row['source_token']}[{row['reason']}]"
                for row in record["top_sources"][:5]
            )
            writer.writerow([
                record["test_sample_index"],
                record.get("task_id"),
                record["target_token_index"],
                record.get("target_token"),
                record["generated_lex"],
                record["reference_lex"],
                *[record["metrics"].get(key) for key in metric_keys],
                top5,
            ])


def parse_k_values(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value > 0 and value not in values:
            values.append(value)
    return values


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute structural spurious-correlation metrics from feature attribution JSON files."
    )
    parser.add_argument(
        "--feature-dir",
        default="attribution_results_feature_alti_saliency_full100/feature",
    )
    parser.add_argument("--output-dir", default="spurious_correlation_results")
    parser.add_argument("--k-values", default="5,10,20,50")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--source-ranking",
        choices=["alti_saliency", "rank_score", "stored"],
        default="alti_saliency",
        help=(
            "How to rank source correlations before computing top-k metrics. "
            "Use alti_saliency for ALTI top-k; rank_score reproduces the old signed_clip method order."
        ),
    )
    args = parser.parse_args(argv)

    k_values = parse_k_values(args.k_values)
    parser_go = _new_go_parser()
    files = _feature_files(args.feature_dir)
    if args.limit is not None:
        files = files[: int(args.limit)]

    records = []
    for path in files:
        records.append(analyze_file(path, parser_go, k_values, args.source_ranking))
    summary = build_summary(records, k_values)
    write_outputs(records, summary, args.output_dir)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
