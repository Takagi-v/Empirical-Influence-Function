#!/usr/bin/env python3
"""Decode ``input_ids`` from an export_token_embeddings .pt file.

Decoding uses the **base tokenizer only**. LoRA adapter is NOT needed
(and is never loaded here). Adapter only changes ``hidden`` vectors, not
token ↔ text mapping.

Examples::

  # Print full text + first/last few token rows
  python scripts/decode_token_embeddings.py \\
    --pt ./outputs/emb_ce_only/embeddings/000000.pt \\
    --model_name_or_path /mnt/md124/jiaxin/models/Qwen3-8B

  # If the .pt already has token_surfaces / text (new exports), tokenizer is optional:
  python scripts/decode_token_embeddings.py --pt ./outputs/emb_ce_only/embeddings/000000.pt

  # Dump index / id / surface / is_completion as tsv
  python scripts/decode_token_embeddings.py \\
    --pt ./outputs/emb_ce_only/embeddings/000000.pt \\
    --model_name_or_path /mnt/md124/jiaxin/models/Qwen3-8B \\
    --tsv ./outputs/emb_ce_only/embeddings/000000.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pt", required=True, help="Path to embeddings/XXXXXX.pt")
    p.add_argument(
        "--model_name_or_path",
        default="",
        help="Base model dir for tokenizer (only needed if .pt has no token_surfaces)",
    )
    p.add_argument("--tsv", default="", help="Optional path to write index\\tid\\tsurface\\tis_completion")
    p.add_argument("--head", type=int, default=20, help="Print first N tokens (0 = all)")
    p.add_argument("--tail", type=int, default=10, help="Also print last N tokens")
    args = p.parse_args()

    obj = torch.load(args.pt, map_location="cpu", weights_only=False)
    ids = obj["input_ids"].tolist()
    labels = obj.get("labels")
    labels = labels.tolist() if labels is not None else [-100] * len(ids)

    surfaces = obj.get("token_surfaces")
    text = obj.get("text")
    if surfaces is None or text is None:
        if not args.model_name_or_path:
            raise SystemExit(
                "This .pt has no cached decode fields; pass --model_name_or_path "
                "(base Qwen dir; adapter NOT required)."
            )
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=True, local_files_only=True
        )
        surfaces = [tok.decode([tid], skip_special_tokens=False) for tid in ids]
        text = tok.decode(ids, skip_special_tokens=False)

    print("=" * 60)
    print(f"pt: {args.pt}")
    print(f"T={len(ids)}  H={tuple(obj['hidden'].shape)}  layer={obj.get('layer')}")
    print(f"adapter_path (info only): {obj.get('adapter_path')!r}")
    print("NOTE: decode = tokenizer only; adapter not used.")
    print("=" * 60)
    print(text)
    print("=" * 60)

    def row(i: int) -> str:
        lab = labels[i]
        flag = "COMP" if lab != -100 else "ctx "
        surf = surfaces[i].replace("\n", "\\n").replace("\t", "\\t")
        return f"{i:5d}\t{ids[i]:8d}\t{flag}\t{surf!r}"

    n = len(ids)
    head = n if args.head <= 0 else min(args.head, n)
    print("idx\tid\trole\tsurface")
    for i in range(head):
        print(row(i))
    if args.head > 0 and args.tail > 0 and head + args.tail < n:
        print("...")
        for i in range(max(head, n - args.tail), n):
            print(row(i))

    if args.tsv:
        out = Path(args.tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            f.write("index\tinput_id\tis_completion\tsurface\n")
            for i in range(n):
                surf = surfaces[i].replace("\n", "\\n").replace("\t", "\\t")
                f.write(f"{i}\t{ids[i]}\t{int(labels[i] != -100)}\t{surf}\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
