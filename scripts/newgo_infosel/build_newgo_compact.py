#!/usr/bin/env python3
"""Convert the new enterprise-Go FIM set ({prompt, response, task_id}) into the
COMPACT training schema the infosel pipeline consumes.

Each output row:
  {
    "task_id":   str,
    "input_ids": [int]            target-tokenized  prompt + response + EOS
    "label":     [int]            IGNORE_INDEX over the prompt, real ids over
                                  response + EOS  (teacher-forcing target)
    "prompt_text": str            the ORIGINAL full prompt (teacher scores with
                                  full context; it truncates on its own side)
    "comp_text": str             the response text (what we score / supervise)
    "comp_spans": [[a,b], ...]    char span of each response token in comp_text
    "comp_q":     [int, ...]      position of each response token in input_ids
  }

`comp_spans` + `comp_q` let the teacher labeler char-span-align its own
tokenization back onto the target token positions (comp_teacher_nll[q]).

The prompt already carries the full FIM context (<PRE>/<SUF>/<MID> markers +
"### Response:"), so NO chat template is added — training on prompt->response is
byte-for-byte what the eval feeds the model.

Usage:
  python scripts/newgo_infosel/build_newgo_compact.py \
    --in  data/new_go_data/train/cloud_core_test_25.JunJunly_GoOnly_length_filter.jsonl \
    --out data/new_go_data/train_data/newgo_infosel_5k_compact.jsonl \
    --target_model models/Qwen2.5-Coder-7B-Instruct \
    --subsample 5000 --seed 42 --max_len 2048
"""
import argparse
import json
import os
import random

IGNORE_INDEX = -100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_model", default="models/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--subsample", type=int, default=0, help="0 = keep all valid rows")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise SystemExit("target tokenizer has no eos_token_id")
    print(f"[build] target={args.target_model} eos_id={eos_id} max_len={args.max_len}", flush=True)

    rng = random.Random(args.seed)
    N = args.subsample
    reservoir = []            # reservoir sample of size N over VALID rows
    seen_valid = 0
    n_read = n_empty = n_skip_long = 0

    def build_row(rec):
        """Return a compact row dict or None (unusable)."""
        prompt = rec.get("prompt", "")
        resp = rec.get("response", "")
        task_id = rec.get("task_id", "")
        if not prompt or not resp.strip():
            return None
        prompt_ids = tok(prompt, add_special_tokens=False).input_ids
        enc = tok(resp, add_special_tokens=False, return_offsets_mapping=True)
        resp_ids, resp_off = enc["input_ids"], enc["offset_mapping"]
        if not resp_ids:
            return None
        # left-truncate the PROMPT if the concatenation is too long (keep the
        # whole response + EOS — we must never drop supervised/scored tokens).
        budget = args.max_len - len(resp_ids) - 1  # -1 for EOS
        if budget <= 0:
            return None                            # response alone too long -> skip
        if len(prompt_ids) > budget:
            prompt_ids = prompt_ids[-budget:]
        n_prompt = len(prompt_ids)
        input_ids = prompt_ids + resp_ids + [eos_id]
        label = [IGNORE_INDEX] * n_prompt + resp_ids + [eos_id]
        comp_spans = [[int(a), int(b)] for (a, b) in resp_off]
        comp_q = [n_prompt + k for k in range(len(resp_ids))]
        return {
            "task_id": task_id,
            "input_ids": input_ids,
            "label": label,
            "prompt_text": prompt,
            "comp_text": resp,
            "comp_spans": comp_spans,
            "comp_q": comp_q,
        }

    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_read += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            row = build_row(rec)
            if row is None:
                if not rec.get("response", "").strip():
                    n_empty += 1
                else:
                    n_skip_long += 1
                continue
            seen_valid += 1
            if N <= 0:
                reservoir.append(row)
            elif len(reservoir) < N:
                reservoir.append(row)
            else:                                   # reservoir replacement
                j = rng.randint(0, seen_valid - 1)
                if j < N:
                    reservoir[j] = row
            if n_read % 20000 == 0:
                print(f"  read {n_read}  valid {seen_valid}  kept {len(reservoir)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lens = []
    with open(args.out, "w", encoding="utf-8") as g:
        for row in reservoir:
            lens.append(len(row["input_ids"]))
            g.write(json.dumps(row, ensure_ascii=False) + "\n")

    lens.sort()
    def pct(p):
        return lens[min(len(lens) - 1, int(len(lens) * p))] if lens else 0
    print(f"[build] read={n_read} valid={seen_valid} empty_resp={n_empty} "
          f"skip_resp_too_long={n_skip_long}", flush=True)
    print(f"[build] wrote {len(reservoir)} rows -> {args.out}", flush=True)
    print(f"[build] input_ids len: p50={pct(.5)} p90={pct(.9)} p99={pct(.99)} max={lens[-1] if lens else 0}", flush=True)


if __name__ == "__main__":
    main()
