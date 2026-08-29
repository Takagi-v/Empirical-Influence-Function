#!/usr/bin/env python3
"""Exact-match evaluation on compact (or chatml-with-input_ids+label) JSONL.

For each row:
  prompt = input_ids[:answer_start]   # answer_start = first label != -100
  gold   = [label[i] for i in range(answer_start, L) if label[i] != -100]
  generate greedily from prompt; a sample is correct iff the generated
  completion token-ids equal gold (after truncating at <|im_end|>).

This matches "全部相同算正确" on the supervised completion span.

Usage:
  python scripts/eval_exact_match_compact.py \\
    --model /path/to/Qwen2.5-Coder-7B-Instruct \\
    --adapter none \\
    --test data/.../test_compact.jsonl \\
    --out_dir outputs/em_eval/base
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


IGNORE_INDEX = -100


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def answer_start(labels: list[int]) -> int | None:
    for i, lab in enumerate(labels):
        if int(lab) != IGNORE_INDEX:
            return i
    return None


def gold_completion_ids(labels: list[int], start: int) -> list[int]:
    return [int(x) for x in labels[start:] if int(x) != IGNORE_INDEX]


def load_model(base: str, adapter: str, dtype_name: str, device_map: str):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if adapter and adapter.lower() not in ("none", "", "-", "null"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        print(f"[em] loaded adapter: {adapter}", flush=True)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact-match eval on compact JSONL")
    ap.add_argument("--model", required=True, help="base model path")
    ap.add_argument("--adapter", default="none", help="LoRA adapter dir, or 'none'")
    ap.add_argument("--test", required=True, help="compact or chatml JSONL with input_ids+label")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=0,
                    help="0 = gold_len + 8 (per sample)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="keep 1 for variable-length prompts (safest)")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    test_path = Path(args.test)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = iter_jsonl(test_path)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    print(f"[em] test={test_path} n={len(rows)} model={args.model} adapter={args.adapter}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # left pad not needed for bs=1; keep right for id slicing consistency
    tok.padding_side = "left"

    model = load_model(args.model, args.adapter, args.dtype, args.device_map)
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [i for i in {tok.eos_token_id, im_end_id} if i is not None and i >= 0]

    pred_path = out_dir / "predictions.jsonl"
    n_ok = 0
    n_total = 0
    n_skip = 0

    with pred_path.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            ids = [int(x) for x in (row.get("input_ids") or [])]
            labels = [int(x) for x in (row.get("label") or row.get("labels") or [])]
            uid = str(row.get("uid") or row.get("task_id") or f"row_{i}")
            if not ids or len(ids) != len(labels):
                n_skip += 1
                fout.write(json.dumps({"uid": uid, "status": "bad_row"}, ensure_ascii=False) + "\n")
                continue
            start = answer_start(labels)
            if start is None or start <= 0:
                n_skip += 1
                fout.write(json.dumps({"uid": uid, "status": "no_answer"}, ensure_ascii=False) + "\n")
                continue

            gold = gold_completion_ids(labels, start)
            prompt = ids[:start]
            max_new = args.max_new_tokens if args.max_new_tokens > 0 else (len(gold) + 8)

            device = next(model.parameters()).device
            inp = torch.tensor([prompt], dtype=torch.long, device=device)
            attn = torch.ones_like(inp)
            with torch.no_grad():
                out = model.generate(
                    input_ids=inp,
                    attention_mask=attn,
                    max_new_tokens=max_new,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=eos_ids if eos_ids else tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,
                )
            gen = out[0, inp.shape[1]:].tolist()

            def truncate_at_eos(seq: list[int]) -> list[int]:
                for j, tid in enumerate(seq):
                    if tid in eos_ids:
                        return seq[: j + 1]
                return seq

            pred_ids = truncate_at_eos(gen)
            gold_cmp = truncate_at_eos(gold)
            exact = pred_ids == gold_cmp

            gold_text = tok.decode(gold_cmp, skip_special_tokens=True)
            pred_text = tok.decode(pred_ids, skip_special_tokens=True)
            text_exact = gold_text.strip() == pred_text.strip()

            n_total += 1
            n_ok += int(exact)
            rec = {
                "uid": uid,
                "status": "ok",
                "exact_match": exact,
                "text_exact_match": text_exact,
                "gold_len": len(gold),
                "pred_len": len(pred_ids),
                "gold_text": gold_text,
                "pred_text": pred_text,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0 or (i + 1) == len(rows):
                acc = n_ok / max(n_total, 1)
                print(f"  [{i+1}/{len(rows)}] running EM={acc:.4f} ({n_ok}/{n_total})", flush=True)

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "test": str(test_path),
        "n_rows": len(rows),
        "n_eval": n_total,
        "n_skip": n_skip,
        "n_exact": n_ok,
        "exact_match_acc": n_ok / max(n_total, 1),
        "definition": "pred completion token-ids == gold labels (label!=-100 span); truncate at eos/im_end",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[em] ===== SUMMARY =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[em] wrote {pred_path}", flush=True)


if __name__ == "__main__":
    main()
