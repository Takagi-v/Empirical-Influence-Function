"""Counterfactual robustness on held-out CSN test_500.

Direct behaviour test of the cfmask hypothesis. For each target token (teacher-
forced on gold), compare the model's gold-token prediction under:
  (a) CLEAN input
  (b) SHORTCUT-MASKED input (hide non-annotated context tokens as attn keys —
      the SAME build_shortcut_mask used in training)

A model that genuinely routes information through the annotated tokens should
degrade LESS when the putative shortcuts are removed. We use an IDENTICAL mask
across all models (fixed seed + same sample order) so the comparison is fair.

Reported per model: gold-token accuracy (argmax==gold) and mean gold NLL under
clean vs masked, and the degradation (Δacc, ΔNLL). cfmask-A/B trained for this;
the question is whether they generalize better than CE-only / contrastive on
HELD-OUT data, and how much robustness they buy vs the saliency baseline.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "train"))
sys.path.insert(0, str(ROOT / "src"))
from dataset import AnnotatedSFTDataset, DataCollatorForAnnotatedSFT, IGNORE_INDEX
from loss import build_shortcut_mask
from data.registry import get_dataset, default_model

ap = argparse.ArgumentParser(
    description="Counterfactual robustness: mask shortcut tokens at eval, measure target-token Δacc/ΔNLL.")
ap.add_argument("--dataset", default="csn_test", help="registry dataset name (uses its compact representation)")
ap.add_argument("--models", required=True,
                help="comma-separated label=path pairs; use 'base' for the base model, e.g. "
                     "'base=base,CE=outputs/ce_only_10k/checkpoints,cfmaskA=outputs/cfmask_A_rate30_10k/checkpoints'")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--rate", type=float, default=0.3, help="match training mask rate")
ap.add_argument("--recency_window", type=int, default=8)
ap.add_argument("--protect_prefix", type=int, default=16)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--out", default=str(ROOT / "outputs/diagnostics/cf_robustness.json"))
args = ap.parse_args()

MODEL = default_model()
if not Path(MODEL).is_absolute():
    MODEL = str(ROOT / MODEL)
TEST = get_dataset(args.dataset).resolve("compact")

def _parse_models(spec: str):
    out = []
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        label, _, path = pair.partition("=")
        path = path.strip()
        out.append((label.strip(), None if path in ("", "base") else path))
    return out

CKPTS = _parse_models(args.models)

device = torch.device("cuda:0")
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
collator = DataCollatorForAnnotatedSFT(tok)
SPECIAL = torch.tensor(sorted({int(i) for i in (tok.all_special_ids or [])}), dtype=torch.long)

ds = AnnotatedSFTDataset(str(TEST), tok)
if args.n > 0 and len(ds) > args.n:
    ds.items = ds.items[:args.n]
print(f"[dataset] CSN test_500: {len(ds)} samples | rate={args.rate} recency={args.recency_window} protect_prefix={args.protect_prefix}")


def to_device(b):
    return {k: (v.to(device) if isinstance(v, torch.Tensor)
                else [t.to(device) if t is not None else t for t in v])
            for k, v in b.items()}


@torch.no_grad()
def gold_stats(logits, input_ids, labels):
    """Per-target-token gold accuracy + NLL with the standard next-token shift."""
    lg = logits[:, :-1, :].float()
    gold = labels[:, 1:]
    valid = gold.ne(IGNORE_INDEX)
    if valid.sum() == 0:
        return 0, 0.0, 0.0
    lv = lg[valid]
    gv = gold[valid]
    logp = F.log_softmax(lv, dim=-1)
    nll = -logp.gather(-1, gv.unsqueeze(-1)).squeeze(-1)
    acc = (lv.argmax(-1) == gv).float()
    return int(valid.sum()), float(acc.sum()), float(nll.sum())


def eval_model(model):
    model.eval()
    n_tok = 0
    clean_acc = clean_nll = mask_acc = mask_nll = 0.0
    # Deterministic mask: identical positions across all models.
    g = torch.Generator(device=device); g.manual_seed(args.seed)
    with torch.no_grad():
        for start in range(0, len(ds), args.batch_size):
            stop = min(start + args.batch_size, len(ds))
            samples = [ds[i] for i in range(start, stop)]
            batch = to_device(collator(samples))
            ids, labels, attn = batch["input_ids"], batch["labels"], batch["attention_mask"]
            pairs = batch["annot_pairs"]
            stats = build_shortcut_mask(
                ids, labels, attn, pairs,
                rate=args.rate, recency_window=args.recency_window,
                protect_prefix=args.protect_prefix, special_ids=SPECIAL.to(device),
                ignore_index=IGNORE_INDEX, generator=g,
            )
            try:
                out_c = model(input_ids=ids, attention_mask=attn)
                out_m = model(input_ids=ids, attention_mask=stats.masked_attention_mask)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            nt, ca, cn = gold_stats(out_c.logits, ids, labels)
            _,  ma, mn = gold_stats(out_m.logits, ids, labels)
            n_tok += nt; clean_acc += ca; clean_nll += cn; mask_acc += ma; mask_nll += mn
            del out_c, out_m
    if n_tok == 0:
        return {}
    return {
        "n_target_tokens": n_tok,
        "clean_acc": clean_acc / n_tok,
        "masked_acc": mask_acc / n_tok,
        "delta_acc": (mask_acc - clean_acc) / n_tok,
        "clean_nll": clean_nll / n_tok,
        "masked_nll": mask_nll / n_tok,
        "delta_nll": (mask_nll - clean_nll) / n_tok,
    }


results = {}
for name, ckpt in CKPTS:
    if ckpt is not None and not (Path(ckpt) / "adapter_config.json").exists():
        print(f"\n=== {name} ckpt={ckpt} -> SKIP (adapter not found, still training?) ===", flush=True)
        continue
    print(f"\n=== {name} ckpt={ckpt} ===", flush=True)
    bm = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True).to(device)
    m = PeftModel.from_pretrained(bm, str(ckpt)) if ckpt else bm
    r = eval_model(m)
    results[name] = r
    print(f"  clean_acc={r['clean_acc']:.4f} masked_acc={r['masked_acc']:.4f} Δacc={r['delta_acc']:+.4f}", flush=True)
    print(f"  clean_nll={r['clean_nll']:.4f} masked_nll={r['masked_nll']:.4f} ΔNLL={r['delta_nll']:+.4f}  (n={r['n_target_tokens']})", flush=True)
    del m, bm; torch.cuda.empty_cache()

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(results, indent=2))
print(f"\nsaved: {args.out}")

print("\n=== Counterfactual robustness (shortcut tokens removed at eval) ===")
print(f"{'ckpt':34s} | clean_acc masked_acc  Δacc  | clean_nll masked_nll  ΔNLL")
for name, r in results.items():
    if not r: continue
    print(f"{name:34s} |   {r['clean_acc']:.3f}     {r['masked_acc']:.3f}   {r['delta_acc']:+.3f} |   {r['clean_nll']:.3f}     {r['masked_nll']:.3f}   {r['delta_nll']:+.3f}")
print("\nLower |Δacc| / ΔNLL = more robust to shortcut removal = genuinely routes through context.")
