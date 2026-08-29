import os
import json
from types import SimpleNamespace
from src.annotate.main import match_lang, build_annotation_code, load_jsonl, extract_masked_spans
from src.annotate.utils import get_qwen3_tokenizer, map_simple_to_bpe
from src.annotate.viz_utils import visualize_correlations

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

JSONL_PATH  = "./data/mceval/safim-train.jsonl"
OUTPUT_PATH = "./data/mceval/safim-train-sft.jsonl"
VIZ_DIR     = "./debug_safim/sft_viz"

SYSTEM_PROMPT = "You are a helpful assistant."
CHAT_PREFIX   = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
CHAT_MIDDLE   = "<|im_end|>\n<|im_start|>assistant\n"
CHAT_SUFFIX   = "<|im_end|>\n"
MASK_TOKEN    = "[MASK]"


def _find_mask_positions_in_sft_input(sft_input: str) -> list[int]:

    sentinel = "\nPlease fill"
    code_part_end = sft_input.find(sentinel)
    search_in = sft_input if code_part_end == -1 else sft_input[:code_part_end]
    positions, cursor = [], 0
    while True:
        idx = search_in.find(MASK_TOKEN, cursor)
        if idx == -1:
            break
        positions.append(idx)
        cursor = idx + len(MASK_TOKEN)
    return positions


def _find_mask_spans_in_filled(
    filled_instruction: str,
    sft_input: str,
    sft_output: str,
) -> list[tuple[int, int]]:

    sft_mask_positions = _find_mask_positions_in_sft_input(sft_input)
    sft_parts = sft_output.split("\n[MASK]\n")

    if len(sft_mask_positions) != len(sft_parts):
        return []

    ANCHOR_LEN = 60
    spans = []

    for i, sft_pos in enumerate(sft_mask_positions):
        span_start = sft_pos
        sft_part   = sft_parts[i]

        candidate_end = span_start + len(sft_part)
        if filled_instruction[span_start:candidate_end] == sft_part:
            spans.append((span_start, candidate_end))
            continue

        after_span_in_sft = sft_input[sft_pos + len(MASK_TOKEN) + len(sft_part):]
        suffix_anchor = after_span_in_sft[:ANCHOR_LEN] if after_span_in_sft.strip() else None

        if suffix_anchor:
            idx = filled_instruction.find(suffix_anchor, span_start)
            if idx == -1:
                # Last resort: use the immediate after-[MASK] content
                after_mask = sft_input[sft_pos + len(MASK_TOKEN):]
                suffix_anchor2 = after_mask[:ANCHOR_LEN]
                idx = filled_instruction.find(suffix_anchor2, span_start)
                if idx == -1:
                    return []
            span_end = idx
        else:
            sentinel_idx = filled_instruction.find("\nPlease fill", span_start)
            span_end = sentinel_idx if sentinel_idx != -1 else len(filled_instruction)

        spans.append((span_start, span_end))

    return spans


def remap_char_offset(
    orig_offset: int,
    mask_spans: list[tuple[int, int]],
    masked_spans_str: str,
    input_section_len: int,
    output_section_start: int,
) -> int | None:

    sft_parts = masked_spans_str.split("\n[MASK]\n")
    if len(sft_parts) != len(mask_spans):
        mask_spans = []
        sft_parts  = []

    sft_part_starts: list[int] = []
    pos = 0
    for p in sft_parts:
        sft_part_starts.append(pos)
        pos += len(p) + len("\n[MASK]\n")

    shrinkage = 0
    for i, (ms, me) in enumerate(mask_spans):
        span_len = me - ms
        mask_len = len(MASK_TOKEN)

        if orig_offset < ms:
            return len(CHAT_PREFIX) + orig_offset - shrinkage

        if orig_offset < me:
            local = orig_offset - ms
            return output_section_start + sft_part_starts[i] + local

        shrinkage += span_len - mask_len

    return len(CHAT_PREFIX) + orig_offset - shrinkage


def remap_tokens_and_annotations(
    entry: dict,
    filled_instruction: str,
    sft_input: str,
    sft_output: str,
) -> tuple[list[dict], list[dict]]:

    mask_spans = _find_mask_spans_in_filled(filled_instruction, sft_input, sft_output)
    sft_parts  = sft_output.split("\n[MASK]\n")

    if len(mask_spans) != len(sft_parts):
        print(f"[warn] {entry.get('task_id','?')}: mask_spans={len(mask_spans)} sft_parts={len(sft_parts)}")

    input_section_len    = len(CHAT_PREFIX) + len(sft_input)
    output_section_start = input_section_len + len(CHAT_MIDDLE)

    def remap(offset):
        return remap_char_offset(offset, mask_spans, sft_output,
                                 input_section_len, output_section_start)

    new_tokens = []
    for tok in entry.get("tokens", []):
        ns = remap(tok["char_start"])
        # char_end must use the same region as char_start.
        # If char_start is inside a mask hole, char_end = char_start + len(surface)
        # to avoid crossing the span boundary and taking a different path.
        if ns is not None:
            ne = ns + (tok["char_end"] - tok["char_start"])
        else:
            ns = tok["char_start"]
            ne = tok["char_end"]
        new_tokens.append({**tok, "char_start": ns, "char_end": ne})

    new_annotations = list(entry.get("annotations", []))
    return new_tokens, new_annotations


# ── Viz helpers ───────────────────────────────────────────────────────────────

def _make_subword(tok):
    ns = SimpleNamespace()
    ns.surface    = tok["surface"]
    ns.token_id   = tok.get("token_id", -1)
    ns.char_start = tok["char_start"]
    ns.char_end   = tok["char_end"]
    ns.clean      = tok["surface"].lstrip("\u0120 \t\n")
    return ns

def _make_correlation(ann):
    ns = SimpleNamespace()
    ns.token_i     = ann["token_i"]
    ns.token_j     = ann["token_j"]
    ns.source      = ann.get("source", "Neural")
    ns.subtype     = ann.get("subtype", "")
    ns.token_i_idx = ann.get("token_i_idx", -1)
    ns.token_j_idx = ann.get("token_j_idx", -1)
    return ns

def _to_subword(tok: dict):
    """Convert token dict to SubwordToken-like namespace for map_simple_to_bpe."""
    ns = SimpleNamespace()
    ns.surface    = tok["surface"]
    ns.token_id   = tok.get("token_id", -1)
    ns.char_start = tok["char_start"]
    ns.char_end   = tok["char_end"]
    return ns


def convert_to_qwen(
    sft_sequence: str,
    new_tokens: list[dict],
    new_annotations: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Re-tokenize sft_sequence with the Qwen BPE tokenizer and expand
    new_annotations so that each simple-token edge becomes all
    qwen-subtoken × qwen-subtoken cross-product edges.

    Returns (qwen_tokens, qwen_annotations).
    """
    tok = get_qwen3_tokenizer()
    enc = tok(sft_sequence, add_special_tokens=False, return_offsets_mapping=True)
    ids: list[int]              = enc["input_ids"]
    offsets: list[tuple[int,int]] = enc["offset_mapping"]

    # Build qwen_tokens as dicts
    qwen_tokens = [
        {
            "surface":    tok.decode([token_id]),
            "token_id":   token_id,
            "char_start": start,
            "char_end":   end,
        }
        for token_id, (start, end) in zip(ids, offsets)
    ]

    # Build simple → qwen index mapping using char overlap.
    # map_simple_to_bpe assumes simple_tokens are sorted by char_start.
    # new_tokens may be out of order (inside-mask tokens have larger char_start
    # than after-mask tokens but appear earlier in the list), so we sort first.
    simple_ns  = [_to_subword(t) for t in new_tokens]
    qwen_ns    = [SimpleNamespace(char_start=q["char_start"],
                                  char_end=q["char_end"]) for q in qwen_tokens]

    # Sort simple tokens by char_start, keeping track of original indices
    sorted_indices = sorted(range(len(simple_ns)), key=lambda i: simple_ns[i].char_start)
    sorted_simple  = [simple_ns[i] for i in sorted_indices]

    # map_simple_to_bpe returns mapping keyed by position in sorted_simple
    sorted_s2b = map_simple_to_bpe(sorted_simple, qwen_ns)

    # Remap back to original simple token indices
    s2b: dict[int, list[int]] = {}
    for sorted_pos, orig_idx in enumerate(sorted_indices):
        s2b[orig_idx] = sorted_s2b.get(sorted_pos, [])

    # Expand annotations: one simple edge → |qwen_i| × |qwen_j| edges
    seen: set[tuple[int,int]] = set()
    qwen_annotations = []
    missing = 0
    for ann in new_annotations:
        si = ann.get("token_i_idx", -1)
        sj = ann.get("token_j_idx", -1)
        qi_list = s2b.get(si, [])
        qj_list = s2b.get(sj, [])
        if not qi_list or not qj_list:
            missing += 1
            continue
        for qi in qi_list:
            for qj in qj_list:
                if qi == qj:
                    continue
                key = (qi, qj)
                if key in seen:
                    continue
                seen.add(key)
                qwen_annotations.append({
                    "token_i":     qwen_tokens[qi]["surface"],
                    "token_j":     qwen_tokens[qj]["surface"],
                    "source":      ann.get("source", "Neural"),
                    "subtype":     ann.get("subtype", ""),
                    "token_i_idx": qi,
                    "token_j_idx": qj,
                })

    if missing:
        print(f"[warn] {missing}/{len(new_annotations)} edges dropped (no qwen mapping)")

    return qwen_tokens, qwen_annotations


def _build_sft_sequence(sft_input: str, sft_output: str) -> str:
    return CHAT_PREFIX + sft_input + CHAT_MIDDLE + sft_output + CHAT_SUFFIX

def visualize_sft_entry(task_id, new_tokens, new_annotations, sft_input, sft_output):
    subwords     = [_make_subword(t)     for t in new_tokens]
    correlations = [_make_correlation(a) for a in new_annotations]
    code         = _build_sft_sequence(sft_input, sft_output)
    safe_id      = task_id.replace("/", "_")
    visualize_correlations(
        correlations=correlations,
        title=f"SFT Annotation · {task_id}",
        code=code,
        subwords=subwords,
        output_path=f"{VIZ_DIR}/{safe_id}.html",
        open_browser=False,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

data = load_jsonl(JSONL_PATH)
ok = skipped = 0
os.makedirs(VIZ_DIR, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for entry in data:
        task_id = entry["task_id"]
        lang    = task_id.split("/")[0]
        if match_lang(lang) is None:
            continue
        if "-single" not in task_id:
            continue
        if "tokens" not in entry or "annotations" not in entry:
            continue
        # if task_id != "Python/50-0-single":
        #     continue

        masked_spans = entry.get("masked_spans") or extract_masked_spans(entry)
        if masked_spans is None:
            print(f"[skip] {task_id} — extract_masked_spans returned None")
            skipped += 1
            continue

        filled_instruction = entry.get("filled_instruction") or build_annotation_code(entry)[1]
        if filled_instruction is None:
            print(f"[skip] {task_id} — build_annotation_code returned None")
            skipped += 1
            continue

        sft_input  = entry["instruction"]
        sft_output = masked_spans

        new_tokens, new_annotations = remap_tokens_and_annotations(
            entry, filled_instruction, sft_input, sft_output
        )

        sft_seq = _build_sft_sequence(sft_input, sft_output)
        qwen_tokens, qwen_annotations = convert_to_qwen(sft_seq, new_tokens, new_annotations)

        if len(qwen_tokens) == 0:
            raise ValueError(task_id)

        visualize_sft_entry(task_id, qwen_tokens, qwen_annotations, sft_input, sft_output)

        fout.write(json.dumps({
            **entry,
            "sft_input":        sft_input,
            "sft_output":       sft_output,
            "tokens":           new_tokens,
            "annotations":      new_annotations,
            "qwen_tokens":      qwen_tokens,
            "qwen_annotations": qwen_annotations,
        }, ensure_ascii=False) + "\n")
        ok += 1

print(f"Done. {ok} entries written → {OUTPUT_PATH}")
print(f"Skipped: {skipped}")