#!/usr/bin/env bash
# Train ce_only + ce_saliency on resp-edges compact data, then exact-match eval
# base / ce_only / ce_saliency on a compact (or chatml-with-ids) test set.
#
# Usage (on server):
#   cd /home/jiaxin/Empirical-Influence-Function/code-corr-annotation   # or this repo root
#   bash scripts/train_eval_resp_edges.sh
#
# Override any path/hyperparam via env vars before calling.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# ── Paths (edit / export to match your server layout) ─────────────────────────
MODEL="${MODEL:-/home/jiaxin/Empirical-Influence-Function/code-corr-annotation/models/Qwen2.5-Coder-7B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-/home/jiaxin/Empirical-Influence-Function/go_single_train_v2_graphsignal_10k_compact_resp_edges.jsonl}"
# Prefer compact test for EM; chatml that also carries input_ids+label also works.
TEST_DATA="${TEST_DATA:-/home/jiaxin/Empirical-Influence-Function/data/go_single/eval_data/codesearchnet_go_test_compact.jsonl}"

OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/resp_edges_ab}"
CE_OUT="${CE_OUT:-${OUT_ROOT}/ce_only}"
SAL_OUT="${SAL_OUT:-${OUT_ROOT}/ce_saliency}"
EVAL_ROOT="${EVAL_ROOT:-${OUT_ROOT}/em_eval}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# ── Train hyperparams ─────────────────────────────────────────────────────────
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29511}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MAX_LEN="${MAX_LEN:-8192}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SALIENCY_LAMBDA="${SALIENCY_LAMBDA:-1.0}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SEED="${SEED:-42}"

# ── Steps to run: train_ce | train_sal | eval | all ───────────────────────────
STAGE="${STAGE:-all}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"   # 0 = full test

echo "============================================================"
echo "ROOT       = ${ROOT}"
echo "MODEL      = ${MODEL}"
echo "TRAIN_DATA = ${TRAIN_DATA}"
echo "TEST_DATA  = ${TEST_DATA}"
echo "OUT_ROOT   = ${OUT_ROOT}"
echo "STAGE      = ${STAGE}"
echo "============================================================"

[[ -f "${MODEL}/config.json" || -f "${MODEL}/tokenizer_config.json" ]] || {
  echo "ERROR: MODEL not found: ${MODEL}" >&2; exit 1; }
[[ -f "${TRAIN_DATA}" ]] || { echo "ERROR: TRAIN_DATA not found: ${TRAIN_DATA}" >&2; exit 1; }
[[ -f "${TEST_DATA}" ]] || { echo "ERROR: TEST_DATA not found: ${TEST_DATA}" >&2; exit 1; }

mkdir -p "${OUT_ROOT}" "${CE_OUT}" "${SAL_OUT}" "${EVAL_ROOT}"

run_train() {
  local loss_mode="$1"
  local output_dir="$2"
  local run_name="$3"
  echo
  echo "########## TRAIN loss_mode=${loss_mode} -> ${output_dir} ##########"
  local launcher=(python)
  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    launcher=(torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}")
  fi
  "${launcher[@]}" src/train/train.py \
    --model_name_or_path "${MODEL}" \
    --data_path "${TRAIN_DATA}" \
    --output_dir "${output_dir}" \
    --loss_mode "${loss_mode}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --bf16 true \
    --eval_strategy "no" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 2 \
    --logging_steps "${LOGGING_STEPS}" \
    --dataloader_num_workers 4 \
    --report_to none \
    --run_name "${run_name}" \
    --max_len "${MAX_LEN}" \
    --remove_unused_columns False \
    --use_peft True \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --saliency_lambda "${SALIENCY_LAMBDA}" \
    --seed "${SEED}"
}

# Pick latest checkpoint-* under a run dir (fallback to run dir itself if adapter there).
latest_adapter() {
  local d="$1"
  local latest
  latest="$(ls -d "${d}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "${latest}" && -f "${latest}/adapter_config.json" ]]; then
    echo "${latest}"
  elif [[ -f "${d}/adapter_config.json" ]]; then
    echo "${d}"
  else
    echo "${latest:-${d}}"
  fi
}

run_em() {
  local tag="$1"
  local adapter="$2"
  local out="${EVAL_ROOT}/${tag}"
  echo
  echo "########## EM EVAL tag=${tag} adapter=${adapter} ##########"
  python scripts/eval_exact_match_compact.py \
    --model "${MODEL}" \
    --adapter "${adapter}" \
    --test "${TEST_DATA}" \
    --out_dir "${out}" \
    --limit "${EVAL_LIMIT}" \
    --dtype bf16 \
    --device_map auto
}

# ── Train ─────────────────────────────────────────────────────────────────────
if [[ "${STAGE}" == "all" || "${STAGE}" == "train_ce" || "${STAGE}" == "train" ]]; then
  run_train ce_only "${CE_OUT}" "resp_edges_ce_only"
fi
if [[ "${STAGE}" == "all" || "${STAGE}" == "train_sal" || "${STAGE}" == "train" ]]; then
  run_train ce_saliency "${SAL_OUT}" "resp_edges_ce_saliency"
fi

# ── Eval ──────────────────────────────────────────────────────────────────────
if [[ "${STAGE}" == "all" || "${STAGE}" == "eval" ]]; then
  run_em base none
  CE_ADAPTER="$(latest_adapter "${CE_OUT}")"
  SAL_ADAPTER="$(latest_adapter "${SAL_OUT}")"
  echo "CE_ADAPTER  = ${CE_ADAPTER}"
  echo "SAL_ADAPTER = ${SAL_ADAPTER}"
  run_em ce_only "${CE_ADAPTER}"
  run_em ce_saliency "${SAL_ADAPTER}"

  echo
  echo "########## EM TABLE ##########"
  export EVAL_ROOT
  python - <<'PY'
import json
from pathlib import Path
import os
root = Path(os.environ["EVAL_ROOT"])
rows = []
for tag in ("base", "ce_only", "ce_saliency"):
    p = root / tag / "summary.json"
    if not p.exists():
        rows.append((tag, None, "missing"))
        continue
    s = json.loads(p.read_text())
    rows.append((tag, s.get("exact_match_acc"), f"{s.get('n_exact')}/{s.get('n_eval')}"))
print(f"{'model':<14} {'EM_acc':>10} {'correct':>12}")
for tag, acc, frac in rows:
    acc_s = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
    print(f"{tag:<14} {acc_s:>10} {frac:>12}")
PY
fi

echo
echo "Done. Summaries under: ${EVAL_ROOT}/*/summary.json"
