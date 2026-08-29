#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

touch "$tmpdir/go.jsonl" "$tmpdir/java.data.jsonl"

out="$(
  TRAIN_DATA_1="$tmpdir/go.jsonl" \
  TRAIN_DATA_2="$tmpdir/java.data.jsonl" \
  MODEL_PATH=/models/base \
  ANNOTATE_MODEL=annotator \
  HUAWEI_BATCH_DRY_RUN=1 \
  bash scripts/data_process/run_huawei_batch_annotation.sh \
    --output-root "$tmpdir/out" \
    --workers 3
)"

grep -Fq "[huawei-batch] files=2" <<<"$out"
grep -Fq -- "--input $tmpdir/go.jsonl" <<<"$out"
grep -Fq -- "--input $tmpdir/java.data.jsonl" <<<"$out"
grep -Fq -- "--language auto" <<<"$out"
grep -Fq -- "--output-dir $tmpdir/out/go" <<<"$out"
grep -Fq -- "--output-dir $tmpdir/out/java_data" <<<"$out"
grep -Fq -- "--run-name go" <<<"$out"
grep -Fq -- "--run-name java_data" <<<"$out"
grep -Fq -- "--num-workers 3" <<<"$out"
