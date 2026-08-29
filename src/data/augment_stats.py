"""Statistics on edge-label augmentation: annotation counts before vs after the
transitive closure.

    python src/data/augment_stats.py --dataset csn_train_5k [--decay 1.0] [--max_hops 0]

Reports, over every sample in the dataset:
  * per-sample edge counts before / after (mean, std, quartiles, min, max),
  * the densification ratio distribution,
  * the per-hop breakdown of the edges added by the closure,
  * a max-hops sensitivity sweep (how many edges each hop budget yields),
  * target connectivity — how many context tokens reach a target token (and at
    what weight) before vs after — which is what weight-aware cfmask consumes.

Edge COUNT is independent of ``--decay`` (decay only weights inherited edges);
``--decay`` only changes the weight / target-connectivity figures.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

THIS = Path(__file__).resolve()
SRC = THIS.parents[1]
sys.path.insert(0, str(SRC))

from data.edge_augment import augment_edges, node_target_weights   # noqa: E402
from data.registry import get_dataset                              # noqa: E402


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))
    return xs[i]


def _row(name, xs):
    return (f"  {name:18s} mean {st.mean(xs):7.2f}  std {(st.pstdev(xs)):6.2f}  "
            f"med {st.median(xs):6.1f}  p25 {_pct(xs,25):5.0f}  p75 {_pct(xs,75):6.0f}  "
            f"p90 {_pct(xs,90):6.0f}  p99 {_pct(xs,99):6.0f}  min {min(xs):4.0f}  max {max(xs):5.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Edge-augmentation count statistics.")
    ap.add_argument("--dataset", default="csn_train_5k")
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--max_hops", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--mode", default="directed", choices=["directed", "undirected"],
                    help="directed closure (a->b,b->c=>a->c) or undirected component closure")
    ap.add_argument("--limit", type=int, default=0, help="cap #samples (0 = all)")
    args = ap.parse_args()

    path = get_dataset(args.dataset).resolve("compact")
    print(f"dataset={args.dataset}  path={path}")
    print(f"mode={args.mode}  decay={args.decay}  max_hops={args.max_hops or 'inf'}\n")

    orig_counts, aug_counts, ratios = [], [], []
    hop_edges = Counter()                 # hop -> total edges across corpus
    sub_orig = Counter()                  # original subtype distribution
    direct_reach, aug_reach = [], []      # #ctx tokens reaching a target
    weight_buckets = Counter()            # rounded node_weight over reaching tokens
    sweep_caps = [1, 2, 3, 4, 0]          # max_hops sensitivity (0 = inf)
    sweep_tot = {c: 0 for c in sweep_caps}
    n = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ids = d.get("input_ids", [])
            lab = d.get("label", d.get("labels", []))
            nt = len(ids)
            if not nt or len(lab) != nt:
                continue
            raw = []
            for e in d.get("attention_edges", []):
                a = int(e.get("src", e.get("token_i_idx", -1)))
                b = int(e.get("dst", e.get("token_j_idx", -1)))
                if 0 <= a < b < nt:
                    raw.append({"src": a, "dst": b, "subtype": e.get("subtype", "edge")})
                    sub_orig[e.get("subtype", "edge")] += 1
            if not raw:
                continue
            n += 1

            aug = augment_edges(raw, decay=args.decay, max_hops=args.max_hops,
                                n_tokens=nt, mode=args.mode)
            o, a = len(raw), len(aug)
            orig_counts.append(o)
            aug_counts.append(a)
            ratios.append(a / o)
            for e in aug:
                hop_edges[e["hops"]] += 1

            # target connectivity
            tgt = [i for i, l in enumerate(lab) if l != -100]
            if tgt:
                direct_reach.append(len(node_target_weights(
                    [{**e, "weight": 1.0} for e in raw], tgt)))
                nw = node_target_weights(aug, tgt)
                aug_reach.append(len(nw))
                for w in nw.values():
                    weight_buckets[round(w, 3)] += 1

            # max-hops sweep (counts only; reuse decay=1.0 since count-independent)
            for c in sweep_caps:
                sweep_tot[c] += len(augment_edges(raw, decay=1.0, max_hops=c,
                                                  n_tokens=nt, mode=args.mode))

            if args.limit and n >= args.limit:
                break

    to, ta = sum(orig_counts), sum(aug_counts)
    print(f"samples with >=1 edge: {n}")
    print(f"TOTAL annotation edges:  before {to:,}   after {ta:,}   "
          f"added {ta-to:,}   (x{ta/to:.3f}, +{100*(ta-to)/to:.1f}%)\n")

    print("PER-SAMPLE edge count:")
    print(_row("before (single-hop)", orig_counts))
    print(_row("after (closure)", aug_counts))
    print(_row("added", [a - o for a, o in zip(aug_counts, orig_counts)]))
    print()

    print("DENSIFICATION ratio (after/before) per sample:")
    print(_row("ratio", ratios))
    print(f"  samples with ratio>1: {100*sum(r>1 for r in ratios)/n:.1f}%   "
          f"ratio==1 (no chains): {100*sum(abs(r-1)<1e-9 for r in ratios)/n:.1f}%\n")

    print("EDGES BY HOP (across corpus):")
    tot = sum(hop_edges.values())
    for h in sorted(hop_edges):
        lbl = "direct (original)" if h == 1 else f"{h-1}-step inherited"
        print(f"  hop {h} ({lbl:18s}): {hop_edges[h]:>9,}  {100*hop_edges[h]/tot:5.1f}%   "
              f"mean/sample {hop_edges[h]/n:6.2f}")
    print()

    print("MAX-HOPS sensitivity (total edges across corpus):")
    base = sweep_tot[1]
    for c in sweep_caps:
        lab_c = "inf" if c == 0 else str(c)
        print(f"  max_hops={lab_c:>3s}: {sweep_tot[c]:>9,}  (x{sweep_tot[c]/base:.3f} vs single-hop)")
    print()

    if aug_reach:
        print("TARGET CONNECTIVITY (context tokens reaching a target token):")
        print(_row("before (direct)", direct_reach))
        print(_row("after (closure)", aug_reach))
        wt = sum(weight_buckets.values())
        first = weight_buckets.get(1.0, 0)
        print(f"  reaching-token weights (decay={args.decay}): "
              f"first-order w=1.0 {100*first/wt:.1f}%, transitive w<1 {100*(wt-first)/wt:.1f}%")
        top = ", ".join(f"{w}:{100*c/wt:.1f}%" for w, c in
                        sorted(weight_buckets.items(), key=lambda x: -x[0])[:6])
        print(f"  top weight buckets: {top}")


if __name__ == "__main__":
    main()
