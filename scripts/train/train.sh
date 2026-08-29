#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

export http_proxy="${http_proxy:-http://127.0.0.1:7890}"
export https_proxy="${https_proxy:-http://127.0.0.1:7890}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/nvme0n1/wenhao/hf_cache}"
export HF_HOME="${HF_HOME:-/mnt/nvme0n1/wenhao/hf_cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-code-corr-saliency}"

# Multi-GPU. Override these when launching, e.g. CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash scripts/train/train.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
MASTER_PORT="${MASTER_PORT:-29500}"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
DATA="${DATA:-/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/processed/go_singleline_fim_exp/train_data/go_single_train_v2_graphsignal_10k_compact.json}"
RUN_NAME="${RUN_NAME:-Qwen2.5-Coder-1.5B-Instruct-saliency}"
OUTPUT="${OUTPUT:-outputs/train/${RUN_NAME}}"
LANGUAGE="${LANGUAGE:-Go}"

mkdir -p "${OUTPUT}" runs/train

if [[ ! -f "${DATA}" ]]; then
  echo "Training data not found: ${DATA}" >&2
  echo "Set DATA=/path/to/compact_annotated.jsonl or .json before running." >&2
  exit 1
fi

torchrun   --nproc_per_node="${NPROC_PER_NODE}"   --master_port="${MASTER_PORT}"   src/train/train.py   --model_name_or_path "${MODEL}"   --data_path        "${DATA}"   --output_dir       "${OUTPUT}"   --num_train_epochs "${NUM_TRAIN_EPOCHS:-30}"   --per_device_train_batch_size  "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"   --gradient_accumulation_steps  "${GRADIENT_ACCUMULATION_STEPS:-4}"   --learning_rate    "${LEARNING_RATE:-2e-5}"   --warmup_ratio     "${WARMUP_RATIO:-0.03}"   --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}"   --max_grad_norm    "${MAX_GRAD_NORM:-1.0}"   --bf16             "${BF16:-true}"   --eval_strategy    "${EVAL_STRATEGY:-steps}"   --eval_steps       "${EVAL_STEPS:-50}"   --save_strategy    "${SAVE_STRATEGY:-steps}"   --save_steps       "${SAVE_STEPS:-100}"   --save_total_limit "${SAVE_TOTAL_LIMIT:-1}"   --logging_steps    "${LOGGING_STEPS:-10}"   --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}"   --report_to        "${REPORT_TO:-wandb}"   --run_name         "${RUN_NAME}"   --max_len          "${MAX_LEN:-8192}"   --remove_unused_columns False   --language         "${LANGUAGE}"   --use_flash_attention "${USE_FLASH_ATTENTION:-False}"   --use_peft         "${USE_PEFT:-True}"   --saliency_lambda  "${SALIENCY_LAMBDA:-1.0}"   --saliency_alpha   "${SALIENCY_ALPHA:-1.5}"   --saliency_eps     "${SALIENCY_EPS:-1e-8}"
