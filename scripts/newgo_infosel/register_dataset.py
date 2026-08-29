#!/usr/bin/env python3
"""Idempotently register the new-go teacher-labeled dataset in the global
dataset registry (configs/datasets.yaml) so run_train.py can resolve it.

The registry is the single source of truth for dataset paths; a training config
only references the dataset by KEY (data.train: <key>). This inserts the entry
right after the top-level `datasets:` line (comment-preserving text insert — a
YAML round-trip would drop the file's many comments).

Usage (from repo root):
  python scripts/newgo_infosel/register_dataset.py
  python scripts/newgo_infosel/register_dataset.py \
    --key newgo_infosel_5k_tlabeled \
    --path data/new_go_data/train_data/newgo_infosel_5k_tlabeled_compact.jsonl --n 5000
"""
import argparse
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_yaml", default=os.path.join(REPO, "configs/datasets.yaml"))
    ap.add_argument("--key", default="newgo_infosel_5k_tlabeled")
    ap.add_argument("--path", default="data/new_go_data/train_data/newgo_infosel_5k_tlabeled_compact.jsonl")
    ap.add_argument("--language", default="go")
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()

    with open(args.datasets_yaml, encoding="utf-8") as f:
        text = f.read()

    if f"\n  {args.key}:" in text or text.startswith(f"  {args.key}:"):
        print(f"[register] '{args.key}' already registered in {args.datasets_yaml}")
        return

    entry = (
        f"  # New enterprise-Go FIM training set, teacher-labeled (comp_teacher_nll)\n"
        f"  # for informative-token selection. Registered by scripts/newgo_infosel/register_dataset.py.\n"
        f"  {args.key}:\n"
        f"    language: {args.language}\n"
        f"    role: train\n"
        f"    format: compact\n"
        f"    has_tests: false\n"
        f"    path: {args.path}\n"
        f"    n: {args.n}\n"
    )

    lines = text.splitlines(keepends=True)
    out, inserted = [], False
    for line in lines:
        out.append(line)
        if not inserted and line.rstrip() == "datasets:":
            out.append("\n" + entry)
            inserted = True
    if not inserted:
        raise SystemExit(f"[register] could not find a top-level 'datasets:' line in {args.datasets_yaml}")

    with open(args.datasets_yaml, "w", encoding="utf-8") as f:
        f.write("".join(out))
    print(f"[register] added '{args.key}' -> {args.path} in {args.datasets_yaml}")


if __name__ == "__main__":
    main()
