from __future__ import annotations
import os
import json
from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.annotate.utils import get_qwen3_tokenizer, map_simple_to_bpe
from src.annotate.viz_utils import visualize_correlations

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ── Config ────────────────────────────────────────────────────────────────────

JSONL_PATH  = "./data/safim/safim-train.jsonl"
OUTPUT_PATH = "./data/safim/safim-train-sft.jsonl"
VIZ_DIR     = "./debug_safim/sft_viz"

SYSTEM_PROMPT = "You are a helpful assistant."
CHAT_PREFIX   = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
CHAT_MIDDLE   = "<|im_end|>\n<|im_start|>assistant\n"
CHAT_SUFFIX   = "<|im_end|>\n"

TODO_STR = "/* TODO: Your code here */"   # SAFIM placeholder, kept in sft_input

# SAFIM lang field → our canonical lang
_SAFIM_LANG_MAP = {
    "java": "Java", "cpp": "CPP", "python": "Python",
    "csharp": "C#", "c#": "C#", "go": "Go", "c": "C",
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def safim_match_lang(lang: str) -> str | None:
    return _SAFIM_LANG_MAP.get(lang.lower())


# ── Code reconstruction (must match main_safim.safim_reconstruct_code) ────────

def safim_reconstruct_code(entry: dict) -> tuple[str, int, int]:
    """
    Return (full_code, gt_start_in_full, gt_end_in_full).

    Mirrors main_safim.safim_reconstruct_code but also tells us WHERE in the
    full code the ground_truth region lives — we need this for the char remap.
    """
    import re
    prompt: str = entry["prompt"]
    ground_truth: str = entry["ground_truth"]

    todo_pattern = re.compile(r'/\*\s*TODO[^*]*\*/', re.DOTALL)
    m = todo_pattern.search(prompt)
    if m:
        # Full code = prompt[:m.start] + ground_truth + prompt[m.end:]
        full = prompt[:m.start()] + ground_truth + prompt[m.end():]
        return full, m.start(), m.start() + len(ground_truth)
    else:
        # Fallback: prompt + "\n" + ground_truth
        full = prompt + "\n" + ground_truth
        gt_start = len(prompt) + 1
        return full, gt_start, gt_start + len(ground_truth)


# ── Char offset remapping ─────────────────────────────────────────────────────

def remap_char_offset(
    orig_offset: int,
    gt_start: int,
    gt_end: int,
    todo_start_in_sft_input: int,
    sft_input_len: int,
) -> int:

    P        = len(CHAT_PREFIX)
    mid_off  = P + sft_input_len + len(CHAT_MIDDLE)   # start of sft_output
    gt_len   = gt_end - gt_start

    if orig_offset < gt_start:
        return P + orig_offset

    if orig_offset < gt_end:
        local = orig_offset - gt_start
        return mid_off + 1 + local       # +1 for the leading "\n" in sft_output

    after_local = orig_offset - gt_end
    return P + todo_start_in_sft_input + len(TODO_STR) + after_local


def remap_tokens(entry: dict, full_code: str, gt_start: int, gt_end: int,
                 sft_input: str) -> list[dict]:
    """Remap every token's char_start/char_end onto the full SFT sequence."""
    todo_start_in_sft_input = sft_input.find(TODO_STR)
    if todo_start_in_sft_input == -1:
        # No placeholder in sft_input — this happens only in the fallback
        # branch where we appended ground_truth. Treat gt as sitting at the end.
        todo_start_in_sft_input = len(sft_input)   # caller must handle fallback

    sft_input_len = len(sft_input)

    new_tokens = []
    for tok in entry.get("tokens", []):
        ns = remap_char_offset(tok["char_start"], gt_start, gt_end, todo_start_in_sft_input, sft_input_len)
        ne = remap_char_offset(tok["char_end"], gt_start, gt_end, todo_start_in_sft_input, sft_input_len)
        new_tokens.append({**tok, "char_start": ns, "char_end": ne})
    return new_tokens


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
    ns = SimpleNamespace()
    ns.surface    = tok["surface"]
    ns.token_id   = tok.get("token_id", -1)
    ns.char_start = tok["char_start"]
    ns.char_end   = tok["char_end"]
    return ns


# ── Qwen BPE re-tokenization ──────────────────────────────────────────────────

def convert_to_qwen(
    sft_sequence: str,
    new_tokens: list[dict],
    new_annotations: list[dict],
) -> tuple[list[dict], list[dict]]:
    tok = get_qwen3_tokenizer()
    enc = tok(sft_sequence, add_special_tokens=False, return_offsets_mapping=True)
    ids     = enc["input_ids"]
    offsets = enc["offset_mapping"]

    qwen_tokens = [
        {
            "surface":    tok.decode([token_id]),
            "token_id":   token_id,
            "char_start": start,
            "char_end":   end,
        }
        for token_id, (start, end) in zip(ids, offsets)
    ]

    simple_ns = [_to_subword(t) for t in new_tokens]
    qwen_ns   = [SimpleNamespace(char_start=q["char_start"], char_end=q["char_end"]) for q in qwen_tokens]

    # map_simple_to_bpe requires simple tokens sorted by char_start
    sorted_indices = sorted(range(len(simple_ns)), key=lambda i: simple_ns[i].char_start)
    sorted_simple  = [simple_ns[i] for i in sorted_indices]
    sorted_s2b     = map_simple_to_bpe(sorted_simple, qwen_ns)

    s2b: dict[int, list[int]] = {}
    for sorted_pos, orig_idx in enumerate(sorted_indices):
        s2b[orig_idx] = sorted_s2b.get(sorted_pos, [])

    seen: set[tuple[int, int]] = set()
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


def visualize_sft_entry(task_id, qwen_tokens, qwen_annotations, sft_input, sft_output):
    subwords     = [_make_subword(t)     for t in qwen_tokens]
    correlations = [_make_correlation(a) for a in qwen_annotations]
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

if __name__ == "__main__":
    data = load_jsonl(JSONL_PATH)
    ok = skipped = 0
    os.makedirs(VIZ_DIR, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for entry in data:
            task_id  = entry["task_id"]
            lang_raw = entry.get("lang", "")
            matched  = safim_match_lang(lang_raw)
            if matched is None:
                continue
            if "tokens" not in entry or "annotations" not in entry:
                continue

            # Reconstruct the full code exactly as main_safim did, and locate
            # the ground_truth region inside it.
            full_code, gt_start, gt_end = safim_reconstruct_code(entry)

            # SFT format: keep TODO placeholder in input, put ground_truth in output
            sft_input  = entry["prompt"]
            sft_output = "\n" + entry["ground_truth"] + "\n"

            # Sanity: char offsets in tokens should reference full_code
            if entry["tokens"]:
                max_end = max(t["char_end"] for t in entry["tokens"])
                if max_end > len(full_code):
                    print(f"[skip] {task_id}: token char_end {max_end} > len(full_code) {len(full_code)}")
                    skipped += 1
                    continue

            new_tokens      = remap_tokens(entry, full_code, gt_start, gt_end, sft_input)
            new_annotations = list(entry["annotations"])  # edges are index-based, no remap needed

            sft_seq = _build_sft_sequence(sft_input, sft_output)
            qwen_tokens, qwen_annotations = convert_to_qwen(sft_seq, new_tokens, new_annotations)

            if len(qwen_tokens) == 0:
                print(f"[skip] {task_id}: empty qwen tokenization")
                skipped += 1
                continue

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