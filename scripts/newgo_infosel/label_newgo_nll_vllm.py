#!/usr/bin/env python3
"""Teacher-gated informative-token labeling for the new-go compact set — vLLM.

Reads the compact rows produced by build_newgo_compact.py (which carry
prompt_text / comp_text / comp_spans / comp_q) and adds `comp_teacher_nll`:
the per-target-token NLL under the teacher, produced by ONE teacher-forcing
forward per row (feed prompt_text + comp_text, read vLLM `prompt_logprobs`).

The teacher tokenizes independently, so its per-token NLL is char-span-aligned
back onto the TARGET response tokens (comp_spans / comp_q) — identical alignment
to scripts/data_process/label_token_informativeness_vllm.py.

Output row (lean, ready for training via src/train/dataset.py token_select):
  {task_id, input_ids, label, comp_teacher_nll: [[q, nll], ...]}

Run in the `vllm` conda env:
  python scripts/newgo_infosel/label_newgo_nll_vllm.py \
    --in  data/new_go_data/train_data/newgo_infosel_5k_compact.jsonl \
    --out data/new_go_data/train_data/newgo_infosel_5k_tlabeled_compact.jsonl \
    --teacher models/Qwen3-Coder-30B-A3B-Instruct \
    --tp 1 --max_len 4096 --chunk 128 --gpu_mem_util 0.92
"""
import argparse
import json
import os
import sys
import time
from bisect import bisect_right
from collections import defaultdict


def span_index(starts, a):
    """Largest k with starts[k] <= a (which target token owns teacher char a)."""
    k = bisect_right(starts, a) - 1
    return k if k >= 0 else None


def build_seq(row, ttok, max_len):
    """Return (full_ids, H, comp_ids, comp_off) or None. Teacher head = prompt_text."""
    prompt = row.get("prompt_text", "")
    comp = row.get("comp_text", "")
    if not prompt or not comp:
        return None
    head_ids = ttok(prompt, add_special_tokens=False).input_ids
    enc = ttok(comp, add_special_tokens=False, return_offsets_mapping=True)
    comp_ids, comp_off = enc["input_ids"], enc["offset_mapping"]
    if not comp_ids:
        return None
    full = head_ids + comp_ids
    if len(full) > max_len:
        over = len(full) - max_len
        if over >= len(head_ids):
            return None                       # completion alone too long
        return full[over:], len(head_ids) - over, comp_ids, comp_off
    return full, len(head_ids), comp_ids, comp_off


def align(prompt_logprobs, H, comp_ids, comp_off, comp_spans, comp_q):
    """Sum teacher token NLLs into TARGET response tokens via char spans.

    Returns [[q, nll], ...] over target response tokens that received any mass.
    """
    if not prompt_logprobs:
        return None
    starts = [s[0] for s in comp_spans]        # target-token char starts in comp_text
    acc = defaultdict(float)
    for k, (a, b) in enumerate(comp_off):
        if a == b:
            continue
        pos = H + k
        if pos >= len(prompt_logprobs):
            continue
        lp = prompt_logprobs[pos]
        if not lp:
            continue
        entry = lp.get(comp_ids[k])
        if entry is None:
            continue
        idx = span_index(starts, a)
        if idx is not None and idx < len(comp_q):
            acc[idx] += -float(entry.logprob)
    if not acc:
        return None
    return [[int(comp_q[idx]), round(v, 4)] for idx, v in sorted(acc.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="models/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--max_num_batched_tokens", type=int, default=2048,
                    help="cap prefill positions/step -> bounds the prompt_logprobs "
                         "transient over the 151k vocab (prevents CUDA OOM).")
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    try:
        from vllm import TokensPrompt
    except ImportError:
        from vllm.inputs import TokensPrompt

    ttok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    print(f"[label] teacher={args.teacher} tp={args.tp} max_len={args.max_len} "
          f"chunk={args.chunk} gpu_mem_util={args.gpu_mem_util} "
          f"max_num_batched_tokens={args.max_num_batched_tokens}", flush=True)
    # skip_tokenizer_init: we feed token ids + read prompt_logprobs by id, so vLLM
    # never needs its own tokenizer (also dodges any tokenizer version-skew crash).
    llm = LLM(model=args.teacher, tensor_parallel_size=args.tp, dtype="bfloat16",
              max_model_len=args.max_len, gpu_memory_utilization=args.gpu_mem_util,
              max_num_batched_tokens=args.max_num_batched_tokens,
              enforce_eager=True, trust_remote_code=True, skip_tokenizer_init=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)

    rows = []
    with open(args.inp, encoding="utf-8") as f:
        for li, line in enumerate(f):
            if args.max_rows and li >= args.max_rows:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[label] {len(rows)} rows loaded", flush=True)

    built = [None] * len(rows)
    prompts, idxmap = [], []
    for i, row in enumerate(rows):
        b = build_seq(row, ttok, args.max_len)
        if b is None:
            continue
        full, H, comp_ids, comp_off = b
        built[i] = (H, comp_ids, comp_off)
        prompts.append(TokensPrompt(prompt_token_ids=full))
        idxmap.append(i)
    print(f"[label] {len(prompts)}/{len(rows)} rows buildable", flush=True)

    results = [None] * len(rows)
    t0 = time.time()
    for c0 in range(0, len(prompts), args.chunk):
        outs = llm.generate(prompts[c0:c0 + args.chunk], sp, use_tqdm=False)
        for j, o in enumerate(outs):
            i = idxmap[c0 + j]
            H, comp_ids, comp_off = built[i]
            results[i] = align(o.prompt_logprobs, H, comp_ids, comp_off,
                               rows[i]["comp_spans"], rows[i]["comp_q"])
        done = min(c0 + args.chunk, len(prompts))
        print(f"  {done}/{len(prompts)}  ({done/max(time.time()-t0,1e-6):.1f} rows/s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    scored = empties = 0
    with open(args.out, "w", encoding="utf-8") as g:
        for i, row in enumerate(rows):
            nlls = results[i]
            out_row = {
                "task_id": row.get("task_id", ""),
                "input_ids": row["input_ids"],
                "label": row["label"],
                "comp_teacher_nll": nlls if nlls else [],
            }
            if nlls:
                scored += 1
            else:
                empties += 1
            g.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    dt = time.time() - t0
    print(f"[label] DONE {args.inp} -> {args.out}: {len(rows)} rows, {scored} scored, "
          f"{empties} empty ({dt:.0f}s, {len(rows)/max(dt,1e-6):.1f} rows/s)", flush=True)
    if scored < max(1, len(rows) // 5):
        print(f"[FATAL] only {scored}/{len(rows)} scored -> aborting (no empty label file).", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
