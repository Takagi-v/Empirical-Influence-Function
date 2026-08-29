from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.annotate.utils import tokenize_code_for_annotation, get_token_indices_in_span
from src.annotate.neural_annot import AnnotatorAgent
from src.annotate.viz_utils import *
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from collections import Counter as _Counter
from typing import Tuple
os.environ.setdefault("http_proxy", "http://127.0.0.1:7890")
os.environ.setdefault("https_proxy", "http://127.0.0.1:7890")

TASK_SUFFIXES = {"multi", "single", "span", "light-span"}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _to_dict(obj):
    """Serialize dataclass or plain object to dict."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def _flush(entries: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def get_base_id(task_id: str) -> str:
    # Step 1: strip suffix (e.g. "-multi", "-light-span")
    for suffix in TASK_SUFFIXES:
        if task_id.endswith("-" + suffix):
            task_id = task_id[: -(len(suffix) + 1)]
            break
    return task_id


def match_lang(lang: str) -> str | None:
    if "Python" in lang:
        return "Python"
    if "C#" in lang:
        return "C#"
    if "CPP" in lang:
        return "CPP"
    if "Go" in lang:
        return "Go"
    if "Java" in lang and "JavaScript" not in lang:
        return "Java"
    if lang == "C":
        return "C"
    return None

def _parse_incomplete_code(instruction: str) -> str | None:
    """Extract incomplete code block from instruction."""
    m = re.search(r'\* Incomplete Code:\n(.*?)(?:\n+Please fill)', instruction, re.DOTALL)
    return m.group(1).strip() if m else None


def _detect_lang(instruction: str) -> str:
    """Extract language name from the preamble line."""
    m = re.search(r"Below is a explanation of (.+?) code", instruction)
    return m.group(1) if m else "the"


def _normalize_seg(seg: str, solution: str, strip_sig: bool = False) -> str:
    """
    Normalize a segment's indentation to match canonical_solution.
    strip_sig=True for the first segment (may include function signature).
    """
    lines = seg.strip("\n").split("\n")
    TRIVIAL = {"{", "}", "(", ")", "};", ");"}

    first_body_idx = 0
    if strip_sig:
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if not stripped or stripped in TRIVIAL:
                continue
            indent = len(line) - len(stripped)
            for d in range(0, indent + 1, 2):
                if " " * (indent - d) + stripped in solution:
                    first_body_idx = i
                    break
            if first_body_idx:
                break

    deltas = []
    for line in lines[first_body_idx:]:
        stripped = line.lstrip()
        if not stripped or stripped in TRIVIAL:
            continue
        indent = len(line) - len(stripped)
        for d in range(0, indent + 1, 2):
            if " " * (indent - d) + stripped in solution:
                deltas.append(d)
                break

    if not deltas:
        result = []
        for line in lines[first_body_idx:]:
            stripped = line.lstrip()
            result.append(stripped if (stripped in solution or not stripped) else line)
        return "\n".join(result)

    best_delta = _Counter(deltas).most_common(1)[0][0]
    result = []
    for line in lines[first_body_idx:]:
        if len(line) >= best_delta and line[:best_delta] == " " * best_delta:
            result.append(line[best_delta:])
        else:
            result.append(line)
    return "\n".join(result)


def extract_masked_spans(entry: dict) -> str | None:
    canonical_solution = entry["canonical_solution"].strip()
    prefix_code = entry.get("prefix_code", "").strip()
    suffix_code = entry.get("suffix_code", "").strip()

    incomplete_code = entry.get("mask_code") or _parse_incomplete_code(entry["instruction"])
    if not incomplete_code:
        return None

    segments = incomplete_code.strip().split("[MASK]")
    if len(segments) < 2:
        return None

    is_light_span = (prefix_code or suffix_code) and not prefix_code.endswith("\n") and "\n" not in prefix_code.split("\n")[-1].strip()

    if is_light_span:
        start = 0
        if prefix_code:
            for trim in range(len(prefix_code) + 1):
                tail = prefix_code[trim:]
                if not tail.strip():
                    break
                idx = canonical_solution.find(tail)
                if idx != -1:
                    start = idx + len(tail)
                    break
        end = len(canonical_solution)
        if suffix_code:
            for trim in range(len(suffix_code) + 1):
                head = suffix_code[:len(suffix_code) - trim]
                if not head.strip():
                    break
                idx = canonical_solution.find(head, start)
                if idx != -1:
                    end = idx
                    break
        span = canonical_solution[start:end].strip("\n")
        return span if span else None

    masked_parts = []
    cursor = 0

    for i in range(len(segments) - 1):
        raw_seg = segments[i]
        norm_seg = _normalize_seg(raw_seg, canonical_solution, strip_sig=(i == 0))
        search_seg = norm_seg.lstrip("\n")

        print(f"[debug] seg[{i}] raw={repr(raw_seg[:60])}")
        print(f"[debug] seg[{i}] search={repr(search_seg[:60])}")

        if not search_seg.strip():
            seg_end = cursor
        else:
            seg_end = canonical_solution.find(search_seg, cursor)
            print(f"[debug] seg[{i}] find result: {seg_end}")
            if seg_end == -1:
                return None
            seg_end += len(search_seg)

        next_seg = segments[i + 1]
        if next_seg.strip():
            search_next = _normalize_seg(next_seg, canonical_solution).lstrip("\n")
            print(f"[debug] next_seg search={repr(search_next[:60])}")
            next_start = canonical_solution.find(search_next, seg_end)
            print(f"[debug] next_seg find result: {next_start}")
            if next_start == -1:
                return None
        else:
            next_start = len(canonical_solution)

        masked_parts.append(canonical_solution[seg_end:next_start])
        cursor = next_start

    return "\n[MASK]\n".join(masked_parts)

def build_annotation_code(entry: dict) -> Tuple[str | None, str | None]:
    masked_spans = extract_masked_spans(entry)
    if masked_spans is None:
        return masked_spans, None

    incomplete_code = entry.get("mask_code") or _parse_incomplete_code(entry["instruction"])
    if not incomplete_code:
        return masked_spans, None

    spans = masked_spans.split("\n[MASK]\n")
    parts = incomplete_code.strip().split("[MASK]")
    if len(parts) != len(spans) + 1:
        return masked_spans, None

    # Fill the mask_code
    filled_code = parts[0]
    for span, tail in zip(spans, parts[1:]):
        filled_code += span + tail

    # Replace the mask_code block inside instruction with the filled version
    instruction = entry["instruction"]
    incomplete_code_stripped = incomplete_code.strip()
    if incomplete_code_stripped in instruction:
        return masked_spans, instruction.replace(incomplete_code_stripped, filled_code, 1)

    # Fallback: replace [MASK] occurrences directly in instruction
    result = instruction
    for span in spans:
        result = result.replace("[MASK]", span, 1)
    return masked_spans, result

def annotate_one(base_id: str, code: str, matched: str, max_rounds: int = 6) -> tuple[str, list, list]:
    """Annotate a single base_id. Returns (base_id, tokens_out, annots_out)."""
    subwords = tokenize_code_for_annotation(code)

    _incomplete_code_re = re.compile(
        r'\* Incomplete Code:\n(.*?)(?:\n+Please fill)', re.DOTALL
    )
    m = _incomplete_code_re.search(code)
    if m:
        target_indices = get_token_indices_in_span(subwords, m.start(1), m.end(1))
        # Slice text that tree-sitter will parse — valid source only, no wrapper
        ts_code = m.group(1)
        ts_char_offset = m.start(1)
    else:
        # Fallback: annotate everything
        target_indices = None
        ts_code = None
        ts_char_offset = 0

    ann = AnnotatorAgent(language=matched, max_rounds=max_rounds)
    neu_sw = ann.annotate(code,
                          subwords,
                          target_indices=target_indices,
                          ts_code=ts_code,
                          ts_char_offset=ts_char_offset)

    html_path = f"./debug/{base_id.replace('/', '_')}.html"
    visualize_correlations(
        neu_sw,
        title="Token Correlation · Attention View",
        code=code,
        subwords=subwords,
        output_path=html_path,
        open_browser=False,
    )

    tokens_out = [_to_dict(sw) for sw in subwords]
    annots_out = [_to_dict(c) for c in neu_sw]
    return base_id, tokens_out, annots_out


if __name__ == "__main__":

    JSONL_PATH = "./data/mceval/mceval-completion.jsonl"
    MAX_WORKERS = 16  # tune to your API rate limit

    data = load_jsonl(JSONL_PATH)

    # ── Pass 1: collect unique base_ids that need annotation ──────────────────
    # base_id → (code, matched_lang)
    todo: dict[str, tuple[str, str]] = {}
    # base_id → (filled_instruction, masked_spans)
    base_id_meta: dict[str, tuple[str, str]] = {}

    for entry in data:
        task_id = entry["task_id"]
        lang = task_id.split("/")[0]
        matched = match_lang(lang)
        if matched is None:
            continue
        if "-single" not in task_id:
            continue
        base_id = get_base_id(task_id)

        if base_id not in todo:
            masked_spans, code = build_annotation_code(entry)
            if code is None:
                continue
            todo[base_id] = (code, matched)
            base_id_meta[base_id] = (code, masked_spans)

    # ── Pass 2: parallel annotation ───────────────────────────────────────────
    flush_lock = threading.Lock()
    ct = 0


    def _annotate_and_cache(item):
        base_id, (code, matched) = item
        try:
            base_id, tokens_out, annots_out = annotate_one(base_id, code, matched)
            return base_id, tokens_out, annots_out, None
        except Exception as e:
            return base_id, None, None, e


    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_annotate_and_cache, item): item[0] for item in todo.items()}

        for future in as_completed(futures):
            base_id, tokens_out, annots_out, err = future.result()
            if err:
                print(f"[error] {base_id}: {err}")
                continue

            ct += 1
            print(f"[done] {base_id} ({ct}/{len(todo)})")

            # Write tokens/annotations back to -single entries with this base_id
            for entry in data:
                if "-single" not in entry["task_id"]:
                    continue
                if get_base_id(entry["task_id"]) == base_id:
                    entry["tokens"]             = tokens_out
                    entry["annotations"]        = annots_out
                    filled, spans = base_id_meta.get(base_id, (None, None))
                    entry["filled_instruction"] = filled
                    entry["masked_spans"]       = spans

            if ct % 10 == 0:
                with flush_lock:
                    _flush(data, JSONL_PATH)
                    print(f"[flush] saved checkpoint at {ct} annotations")

    # ── Pass 3: apply cache to all entries and final flush ────────────────────
    lang_ct: dict[str, int] = {}
    for entry in data:
        task_id = entry["task_id"]
        lang = task_id.split("/")[0]
        matched = match_lang(lang)
        if matched is None:
            continue
        if "-single" not in task_id:
            continue

        base_id = get_base_id(task_id)
        if "tokens" in entry:
            lang_ct[matched] = lang_ct.get(matched, 0) + 1

    _flush(data, JSONL_PATH)
    print(f"Done. {ct} base_ids annotated → {JSONL_PATH}")
    print(f"Per-lang entry counts: {lang_ct}")