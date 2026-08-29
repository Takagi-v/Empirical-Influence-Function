#!/usr/bin/env python3
"""Evaluate a trained (LoRA) model on the new enterprise-Go FIM test set using the
AI4Go metric (es / bleu / rougeL / line-hit / acc).

Steps:
  1. load base model (+ LoRA adapter if given), greedy-generate a completion for
     each test `prompt` (raw prompt -> completion, exactly as trained/served);
  2. write {task_id, label(=response), predict} JSONL;
  3. score it with the AI4Go MetricsEvaluator (eval_main) and save the summary.

Run in the `code-attribute` env (transformers + peft):
  python scripts/newgo_infosel/eval_newgo.py \
    --base_model models/Qwen2.5-Coder-7B-Instruct \
    --adapter outputs/lora_infosel_newgo5k_7b_s42/checkpoints/checkpoint-XXX \
    --test data/new_go_data/test/processed_part1.jsonl \
    --out_predictions outputs/newgo_eval/infosel_predictions.jsonl \
    --out_metrics     outputs/newgo_eval/infosel_metrics.json
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="models/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--adapter", default="none", help="LoRA checkpoint dir, or 'none' for the base model")
    ap.add_argument("--test", required=True)
    ap.add_argument("--out_predictions", required=True)
    ap.add_argument("--out_metrics", required=True)
    ap.add_argument("--ai4go_dir",
                    default=os.environ.get("AI4GO_DIR", ""),
                    help="path to the extracted AI4Go eval framework (or set $AI4GO_DIR)")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--prompt_max_len", type=int, default=1792)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all test rows")
    ap.add_argument("--device_map", default="auto",
                    help="HF device_map (default 'auto' = spread a large model across all "
                         "visible GPUs; on 1 GPU it just uses cuda:0)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = []
    with open(args.test, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[eval] {len(rows)} test rows | base={args.base_model} adapter={args.adapter}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"        # decoder-only batched generation
    tok.truncation_side = "left"     # keep the code nearest the <MID> hole

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map=args.device_map, trust_remote_code=True)
    if args.adapter and args.adapter.lower() != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[eval] loaded adapter {args.adapter}", flush=True)
    model.eval()

    preds = []
    bs = args.batch_size
    for b0 in range(0, len(rows), bs):
        batch = rows[b0:b0 + bs]
        prompts = [r.get("prompt", "") for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.prompt_max_len).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                  do_sample=False, num_beams=1,
                                  eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for r, g in zip(batch, gen):
            predict = tok.decode(g, skip_special_tokens=True)
            preds.append({"task_id": r.get("task_id", ""),
                          "label": r.get("response", ""), "predict": predict})
        print(f"  generated {min(b0 + bs, len(rows))}/{len(rows)}", flush=True)

    os.makedirs(os.path.dirname(args.out_predictions) or ".", exist_ok=True)
    with open(args.out_predictions, "w", encoding="utf-8") as g:
        for p in preds:
            g.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[eval] wrote {len(preds)} predictions -> {args.out_predictions}", flush=True)

    # ── score with the AI4Go metric (es/bleu/rougeL/line-hit/acc) ──────────────
    sys.path.insert(0, args.ai4go_dir)
    from eval.eval import eval_main          # noqa: E402  (AI4Go framework)
    df = eval_main(args.out_predictions)
    summary = {k: (v[0] if hasattr(v, "__len__") and not isinstance(v, str) else v)
               for k, v in df.to_dict(orient="list").items()}
    os.makedirs(os.path.dirname(args.out_metrics) or ".", exist_ok=True)
    with open(args.out_metrics, "w", encoding="utf-8") as g:
        json.dump(summary, g, indent=2, default=str)
    print("[eval] ===== METRICS =====", flush=True)
    print(df.to_string(), flush=True)
    print(f"[eval] wrote metrics -> {args.out_metrics}", flush=True)


if __name__ == "__main__":
    main()
