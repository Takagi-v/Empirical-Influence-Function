"""Honest-mAP diagnostic: how much of the saliency-align gap is the attention sink?

For base / CE-only / contrastive (10k checkpoints), compute saliency mAP@k +
recall@k over CSN test_full under a 2x2 grid:

    layer        in {last (27), middle (default 14)}
    candidate    in {raw (all causal sources),
                     honest (drop sink-prefix positions 0..P-1 + special tokens)}

A single forward pass per batch is reused for all four settings (every layer's
attentions/hidden_states are available), so the numbers are directly comparable.

Interpretation:
  * last_raw   = the original headline number (0.13 base -> 0.92 contrastive).
  * last_honest= alignment among *content* tokens only. If base/CE-only jump a
    lot here, the original low score was a sink artifact (Goodhart).
  * mid_*      = is value-carrying routing alignment different mid-stack?
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path("/home/v-murongma/code/code-corr-annotation")
sys.path.insert(0, str(ROOT / "src" / "train"))
from dataset import AnnotatedSFTDataset, DataCollatorForAnnotatedSFT
from loss import build_contribution_rows, _annotation_rows_from_pairs

MODEL = str(ROOT / "models" / "Qwen2.5-Coder-7B-Instruct")
TEST = ROOT / "data/data_10k/go_single_fim/eval_data/codesearchnet_go_test_full_srcannotate_compact.json"
KS = [1, 5, 10]
SETTINGS = ["last_raw", "last_honest", "mid_raw", "mid_honest"]

CKPTS = [
    ("base", None),
    ("CE only", ROOT / "outputs/go_single/models/ce_only_10k_r32_e3_run1"),
    ("CE+Contrastive lam1.5", ROOT / "outputs/go_single/models/ce_contrastive_lam1p5_negK64_10k_r32_e3_run1"),
    ("Tier1 nosink ep3", ROOT / "outputs/go_single/models/ce_contrastive_lam1p5_negK64_nosink_10k_r32_e3_run1/checkpoint-3750"),
    ("Tier1 mid14+nosink ep3", ROOT / "outputs/go_single/models/ce_contrastive_lam1p5_negK64_mid14_nosink_10k_r32_e3_run1/checkpoint-3750"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500, help="first-N test samples (0 = all)")
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--mid_layer", type=int, default=14)
ap.add_argument("--sink_prefix", type=int, default=3)
ap.add_argument("--out", default=str(ROOT / "outputs/visual_saliency/honest_map_diag.json"))
args = ap.parse_args()

device = torch.device("cuda:0")
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
collator = DataCollatorForAnnotatedSFT(tok)
SPECIAL = torch.tensor(sorted({int(i) for i in (tok.all_special_ids or [])}),
                       dtype=torch.long, device=device)

ds = AnnotatedSFTDataset(str(TEST), tok)
if args.n > 0 and len(ds) > args.n:
    ds.items = ds.items[:args.n]
print(f"[dataset] test_full: {len(ds)} samples | sink_prefix={args.sink_prefix} mid_layer={args.mid_layer}")


def to_device(b):
    return {k: (v.to(device) if isinstance(v, torch.Tensor)
                else [t.to(device) if t is not None else t for t in v])
            for k, v in b.items()}


def ap_for(C_rows, M_eff, A_bin):
    """Per-row AP + recall@k over candidate set M_eff (rows already filtered valid)."""
    C = C_rows.float().masked_fill(~M_eff, float("-inf"))
    order = torch.argsort(C, dim=-1, descending=True).cpu().numpy()
    ncaus = M_eff.sum(-1).cpu().numpy()
    am = A_bin.cpu().numpy()
    aps = []
    perk = {k: {"rec": [], "prec": []} for k in KS}
    for qi in range(order.shape[0]):
        nc = int(ncaus[qi])
        if nc <= 0:
            continue
        ranked = order[qi, :nc]
        hits = am[qi, ranked].astype(np.float32)
        nrel = int(hits.sum())
        if nrel == 0:
            continue
        cum = np.cumsum(hits)
        ranks = np.arange(1, len(hits) + 1)
        aps.append(float(((cum / ranks) * hits).sum() / nrel))
        for k in KS:
            kk = min(k, nc)
            h = float(hits[:kk].sum())
            perk[k]["rec"].append(h / nrel)
            perk[k]["prec"].append(h / k)
    return aps, perk


def eval_model(model):
    model.eval()
    acc = {s: {"ap": [], "perk": {k: {"rec": [], "prec": []} for k in KS}} for s in SETTINGS}
    with torch.no_grad():
        for start in range(0, len(ds), args.batch_size):
            stop = min(start + args.batch_size, len(ds))
            samples = [ds[i] for i in range(start, stop)]
            batch = to_device(collator(samples))
            try:
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            output_attentions=True, output_hidden_states=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            B, T = batch["input_ids"].shape
            rb, rq, src_all, inv = _annotation_rows_from_pairs(batch["annot_pairs"], B=B, T=T, device=device)
            if rq.numel() == 0:
                del out; continue
            Q = rq.numel()
            A_adj = torch.zeros(Q, T, device=device)
            A_adj.index_put_((inv.long(), src_all.long()),
                             torch.ones_like(src_all, dtype=A_adj.dtype), accumulate=True)
            A_bin = (A_adj > 0)
            src_idx = torch.arange(T, device=device).unsqueeze(0)
            Mc = (src_idx <= rq.unsqueeze(1)) & (src_idx != rq.unsqueeze(1))
            ex = torch.zeros((B, T), dtype=torch.bool, device=device)
            if args.sink_prefix > 0:
                ex[:, :args.sink_prefix] = True
            if SPECIAL.numel() > 0:
                ex |= torch.isin(batch["input_ids"], SPECIAL)
            M_honest = Mc & (~ex[rb])

            nL = len(out.attentions)
            last_li = nL - 1
            mid_li = max(0, min(args.mid_layer, nL - 1))
            C_last = build_contribution_rows(model, out.hidden_states[last_li], out.attentions[last_li],
                                             rb, rq, layer_index=last_li)
            C_mid = build_contribution_rows(model, out.hidden_states[mid_li], out.attentions[mid_li],
                                            rb, rq, layer_index=mid_li)
            grid = {
                "last_raw": (C_last, Mc), "last_honest": (C_last, M_honest),
                "mid_raw": (C_mid, Mc), "mid_honest": (C_mid, M_honest),
            }
            for sname, (C_, M_) in grid.items():
                pos = (M_ & A_bin).sum(-1)
                neg = (M_ & (~A_bin)).sum(-1)
                valid = (pos > 0) & (neg > 0)
                if not valid.any():
                    continue
                aps, perk = ap_for(C_[valid], M_[valid], A_bin[valid])
                acc[sname]["ap"].extend(aps)
                for k in KS:
                    acc[sname]["perk"][k]["rec"].extend(perk[k]["rec"])
                    acc[sname]["perk"][k]["prec"].extend(perk[k]["prec"])
            del out, C_last, C_mid
    res = {}
    for s in SETTINGS:
        r = {"mAP": float(np.mean(acc[s]["ap"])) if acc[s]["ap"] else 0.0,
             "n_queries": len(acc[s]["ap"])}
        for k in KS:
            r[f"recall@{k}"] = float(np.mean(acc[s]["perk"][k]["rec"])) if acc[s]["perk"][k]["rec"] else 0.0
        res[s] = r
    return res


results = {}
for name, ckpt in CKPTS:
    print(f"\n=== {name} ckpt={ckpt} ===", flush=True)
    bm = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True).to(device)
    m = PeftModel.from_pretrained(bm, str(ckpt)) if ckpt else bm
    results[name] = eval_model(m)
    for s in SETTINGS:
        r = results[name][s]
        print(f"  {s:12s} mAP={r['mAP']:.4f}  recall@1={r['recall@1']:.4f}  recall@5={r['recall@5']:.4f}  n={r['n_queries']}", flush=True)
    del m, bm; torch.cuda.empty_cache()

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(results, indent=2))
print(f"\nsaved: {args.out}")

print("\n=== mAP summary (rows=ckpt, cols=setting) ===")
print(f"{'ckpt':24s} | " + " ".join(f"{s:>12s}" for s in SETTINGS))
for name in results:
    print(f"{name:24s} | " + " ".join(f"{results[name][s]['mAP']:>12.4f}" for s in SETTINGS))
