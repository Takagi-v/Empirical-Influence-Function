#!/usr/bin/env bash
# =============================================================================
# run_batch_experiments.sh
#
# Scan the entire test set in all-tokens mode.
# Every sample in TEST_DATA is treated as a wrong-prediction case to analyse.
#
# Usage:
#   bash run_batch_experiments.sh [options]
#
# Options (can also be set as env vars):
#   --model-path  PATH   Path to the model checkpoint directory
#   --train-data  FILE   Training JSONL  (default: sft_train.jsonl)
#   --test-data   FILE   Test JSONL      (default: sft_test.jsonl)
#   --indices     LIST   Comma/space-separated sample indices to run, e.g. 3,17,58
#   --start-idx   N      Start from sample index N (default: 0, for resume)
#   --end-idx     N      Stop after sample index N (inclusive, default: last)
#
# Examples:
#   # Full run with a new model
#   bash run_batch_experiments.sh \
#       --model-path /data/models/Qwen3-Coder-30B-Instruct \
#       --train-data /data/my_train.jsonl \
#       --test-data  /data/my_test.jsonl
#
#   # Resume from sample 10 after a crash
#   bash run_batch_experiments.sh \
#       --model-path /data/models/Qwen3-Coder-30B-Instruct \
#       --start-idx 10
#
#   # Run only selected samples
#   bash run_batch_experiments.sh \
#       --model-path /data/models/Qwen3-Coder-30B-Instruct \
#       --indices 3,17,58
# =============================================================================

set -uo pipefail   # -e intentionally omitted: one sample failing won't stop the batch

# ---------------------------------------------------------------------------
# Default configuration (override via flags or env vars)
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"
TRAIN_DATA="${TRAIN_DATA:-sft_train.jsonl}"
TEST_DATA="${TEST_DATA:-sft_test.jsonl}"
INDICES="${INDICES:-}"        # comma/space-separated explicit test sample indices
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-}"          # empty = auto-detect from file
TRAIN_LIMIT="${TRAIN_LIMIT:-}"  # empty = use all training samples
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"  # empty = intervention default
PRESCREEN_MAX_SEQ_LEN="${PRESCREEN_MAX_SEQ_LEN:-}"  # empty = intervention default; 0 disables guard
MAX_GPU_MEMORY="${MAX_GPU_MEMORY:-}"  # e.g. 26GiB to leave activation headroom
CORR_FEATURE_MODE="${CORR_FEATURE_MODE:-}"  # deprecated; accepted for compatibility
PRESCREEN_BATCH_SIZE="${PRESCREEN_BATCH_SIZE:-}"  # empty = 1
ALTI_GRAD_CHUNK_SIZE="${ALTI_GRAD_CHUNK_SIZE:-}"  # empty = intervention default
ALTI_GRAD_MAX_SEQ_LEN="${ALTI_GRAD_MAX_SEQ_LEN:-}"  # empty = intervention default; 0 disables
FINE_MATCH_PROJ="${FINE_MATCH_PROJ:-}"  # empty = qk; use qkvo/all for previous behavior
TOP_TARGETS="${TOP_TARGETS:-}"  # empty = intervention default
TOP_K_SOURCE_PER_TARGET="${TOP_K_SOURCE_PER_TARGET:-}"  # empty = intervention default
PRESCREEN_SKETCH_DIM="${PRESCREEN_SKETCH_DIM:-}"  # empty = intervention default; <=0 disables cache
PRESCREEN_SKETCH_SEED="${PRESCREEN_SKETCH_SEED:-}"  # empty = intervention default
PRESCREEN_SKETCH_CACHE_DIR="${PRESCREEN_SKETCH_CACHE_DIR:-}"  # legacy CE cache (unused)
SALIENCY_TRAIN_BANK_CACHE_DIR="${SALIENCY_TRAIN_BANK_CACHE_DIR:-}"  # empty = .cache/saliency_train_bank
IE_EXTRA_ARGS="${IE_EXTRA_ARGS:-}"  # optional raw passthrough args
PYTHON="${PYTHON:-python}"

# ---------------------------------------------------------------------------
# Parse command-line arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2";  shift 2 ;;
        --base-model-path) BASE_MODEL_PATH="$2"; shift 2 ;;
        --train-data) TRAIN_DATA="$2";  shift 2 ;;
        --test-data)  TEST_DATA="$2";   shift 2 ;;
        --indices)    INDICES="$2";     shift 2 ;;
        --start-idx)  START_IDX="$2";   shift 2 ;;
        --end-idx)    END_IDX="$2";     shift 2 ;;
        --train-limit) TRAIN_LIMIT="$2"; shift 2 ;;
        --attn-implementation) ATTN_IMPLEMENTATION="$2"; shift 2 ;;
        --prescreen-max-seq-len) PRESCREEN_MAX_SEQ_LEN="$2"; shift 2 ;;
        --max-gpu-memory) MAX_GPU_MEMORY="$2"; shift 2 ;;
        --corr-feature-mode) CORR_FEATURE_MODE="$2"; shift 2 ;;
        --prescreen-batch-size) PRESCREEN_BATCH_SIZE="$2"; shift 2 ;;
        --alti-grad-chunk-size) ALTI_GRAD_CHUNK_SIZE="$2"; shift 2 ;;
        --alti-grad-max-seq-len) ALTI_GRAD_MAX_SEQ_LEN="$2"; shift 2 ;;
        --fine-match-proj) FINE_MATCH_PROJ="$2"; shift 2 ;;
        --top-targets) TOP_TARGETS="$2"; shift 2 ;;
        --top-k-source-per-target) TOP_K_SOURCE_PER_TARGET="$2"; shift 2 ;;
        --prescreen-sketch-dim) PRESCREEN_SKETCH_DIM="$2"; shift 2 ;;
        --prescreen-sketch-seed) PRESCREEN_SKETCH_SEED="$2"; shift 2 ;;
        --prescreen-sketch-cache-dir) PRESCREEN_SKETCH_CACHE_DIR="$2"; shift 2 ;;
        --saliency-train-bank-cache-dir) SALIENCY_TRAIN_BANK_CACHE_DIR="$2"; shift 2 ;;
        *) echo "[batch] Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [[ -z "$MODEL_PATH" ]]; then
    echo "[batch] ERROR: --model-path is required."
    echo "        Example: bash run_batch_experiments.sh --model-path /data/models/Qwen3-Coder-30B-Instruct"
    exit 1
fi

if [[ ! -f "$TEST_DATA" ]]; then
    echo "[batch] ERROR: test data file not found: $TEST_DATA"
    exit 1
fi

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "[batch] ERROR: train data file not found: $TRAIN_DATA"
    exit 1
fi

# ---------------------------------------------------------------------------
# Read task_ids from test JSONL (one per non-empty line).
# Falls back to sequential index if a line has no task_id field.
# ---------------------------------------------------------------------------
mapfile -t TASK_IDS < <(python3 - "$TEST_DATA" <<'PYEOF'
import sys, json
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        print(obj.get("task_id") or f"test{i}")
PYEOF
)

TOTAL=${#TASK_IDS[@]}
if [[ -z "$END_IDX" ]]; then
    END_IDX=$((TOTAL - 1))
fi

if [[ -n "$INDICES" ]]; then
    # Accept "3,17,58" or "3 17 58". Empty chunks are ignored.
    IFS=', ' read -r -a RUN_INDICES <<< "$INDICES"
else
    mapfile -t RUN_INDICES < <(seq "$START_IDX" "$END_IDX")
fi

VALID_RUN_INDICES=()
for IDX in "${RUN_INDICES[@]}"; do
    [[ -z "$IDX" ]] && continue
    if ! [[ "$IDX" =~ ^[0-9]+$ ]]; then
        echo "[batch] ERROR: invalid sample index: $IDX"
        exit 1
    fi
    if (( IDX < 0 || IDX >= TOTAL )); then
        echo "[batch] ERROR: sample index out of range: $IDX (valid: 0..$((TOTAL - 1)))"
        exit 1
    fi
    VALID_RUN_INDICES+=("$IDX")
done

if [[ ${#VALID_RUN_INDICES[@]} -eq 0 ]]; then
    echo "[batch] ERROR: no sample indices selected."
    exit 1
fi

# ---------------------------------------------------------------------------
# Setup log directory
# ---------------------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# Derive model tag the same way as intervention_experiment.model_tag_from_path
MODEL_TAG="$(python3 - "$MODEL_PATH" <<'PYEOF'
import os, re, sys
path = sys.argv[1]
tag = os.path.basename(os.path.normpath(path)).strip()
tag = re.sub(r"[^\w.\-]+", "_", tag).strip("._") or "model"
print(tag)
PYEOF
)"

echo "================================================================="
echo "  Batch Experiment Runner  (all-tokens mode)"
echo "  model     : ${MODEL_PATH}"
echo "  base model: ${BASE_MODEL_PATH:-auto-resolve if adapter}"
echo "  model_tag : ${MODEL_TAG}  (appears in result filenames)"
echo "  train data: ${TRAIN_DATA}"
echo "  test data : ${TEST_DATA}  (${TOTAL} samples)"
if [[ -n "$INDICES" ]]; then
    echo "  indices   : ${VALID_RUN_INDICES[*]}"
else
    echo "  range     : [${START_IDX}, ${END_IDX}]"
fi
echo "  train_limit: ${TRAIN_LIMIT:-all}"
echo "  attention : ${ATTN_IMPLEMENTATION:-auto}"
echo "  prescreen max seq len: ${PRESCREEN_MAX_SEQ_LEN:-default}"
echo "  max gpu memory: ${MAX_GPU_MEMORY:-auto}"
echo "  fine match proj: ${FINE_MATCH_PROJ:-qk}"
echo "  alti grad chunk: ${ALTI_GRAD_CHUNK_SIZE:-default}"
echo "  prescreen batch size: ${PRESCREEN_BATCH_SIZE:-1}"
echo "  sketch dim: ${PRESCREEN_SKETCH_DIM:-8192}"
echo "  stage3 train targets: ${TOP_TARGETS:-all}  (first K valid answer tokens; empty=all)"
echo "  stage3 sources/target: ${TOP_K_SOURCE_PER_TARGET:-3}"
echo "  saliency train bank: ${SALIENCY_TRAIN_BANK_CACHE_DIR:-.cache/saliency_train_bank}"
echo "  legacy CE sketch (unused): ${PRESCREEN_SKETCH_CACHE_DIR:-.cache/prescreen_sketch}"
echo "  result pattern: correlation_matching_results_${MODEL_TAG}_<task_id>_all_tokens.json"
echo "  log dir   : ${LOG_DIR}"
echo "================================================================="

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
N_DONE=0
N_SKIP=0
N_FAIL=0
FAILED_INDICES=()

for IDX in "${VALID_RUN_INDICES[@]}"; do
    TASK_ID="${TASK_IDS[$IDX]}"
    RESULT_FILE="${ROOT_DIR}/correlation_matching_results_${MODEL_TAG}_${TASK_ID}_all_tokens.json"
    # Backward compatible: also skip if old filename without model tag exists
    LEGACY_RESULT_FILE="${ROOT_DIR}/correlation_matching_results_${TASK_ID}_all_tokens.json"

    # Resume: skip if result already exists
    if [[ -f "$RESULT_FILE" || -f "$LEGACY_RESULT_FILE" ]]; then
        echo "[$(date +%H:%M:%S)] [${IDX}/${END_IDX}] SKIP  ${MODEL_TAG}/${TASK_ID}  (result exists)"
        N_SKIP=$((N_SKIP + 1))
        continue
    fi

    echo ""
    echo "[$(date +%H:%M:%S)] [${IDX}/${END_IDX}] START  model=${MODEL_TAG} task_id=${TASK_ID}"

    LOG_FILE="${LOG_DIR}/${MODEL_TAG}_${TASK_ID}.log"

    "${PYTHON}" -m src.intervention_experiment \
        --model-path  "${MODEL_PATH}" \
        ${BASE_MODEL_PATH:+--base-model-path "${BASE_MODEL_PATH}"} \
        --train-data  "${TRAIN_DATA}" \
        --test-data   "${TEST_DATA}"  \
        --test-index  "${IDX}"        \
        ${TRAIN_LIMIT:+--train-limit "${TRAIN_LIMIT}"} \
        ${ATTN_IMPLEMENTATION:+--attn-implementation "${ATTN_IMPLEMENTATION}"} \
        ${PRESCREEN_MAX_SEQ_LEN:+--prescreen-max-seq-len "${PRESCREEN_MAX_SEQ_LEN}"} \
        ${MAX_GPU_MEMORY:+--max-gpu-memory "${MAX_GPU_MEMORY}"} \
        ${CORR_FEATURE_MODE:+--corr-feature-mode "${CORR_FEATURE_MODE}"} \
        ${PRESCREEN_BATCH_SIZE:+--prescreen-batch-size "${PRESCREEN_BATCH_SIZE}"} \
        ${ALTI_GRAD_CHUNK_SIZE:+--alti-grad-chunk-size "${ALTI_GRAD_CHUNK_SIZE}"} \
        ${ALTI_GRAD_MAX_SEQ_LEN:+--alti-grad-max-seq-len "${ALTI_GRAD_MAX_SEQ_LEN}"} \
        ${FINE_MATCH_PROJ:+--fine-match-proj "${FINE_MATCH_PROJ}"} \
        ${TOP_TARGETS:+--top-targets "${TOP_TARGETS}"} \
        ${TOP_K_SOURCE_PER_TARGET:+--top-k-source-per-target "${TOP_K_SOURCE_PER_TARGET}"} \
        ${PRESCREEN_SKETCH_DIM:+--prescreen-sketch-dim "${PRESCREEN_SKETCH_DIM}"} \
        ${PRESCREEN_SKETCH_SEED:+--prescreen-sketch-seed "${PRESCREEN_SKETCH_SEED}"} \
        ${PRESCREEN_SKETCH_CACHE_DIR:+--prescreen-sketch-cache-dir "${PRESCREEN_SKETCH_CACHE_DIR}"} \
        ${SALIENCY_TRAIN_BANK_CACHE_DIR:+--saliency-train-bank-cache-dir "${SALIENCY_TRAIN_BANK_CACHE_DIR}"} \
        ${IE_EXTRA_ARGS} \
        2>&1 | tee "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] [${IDX}/${END_IDX}] DONE  → ${RESULT_FILE}"
        N_DONE=$((N_DONE + 1))
    else
        echo "[$(date +%H:%M:%S)] [${IDX}/${END_IDX}] FAIL  ${TASK_ID}  (exit=${EXIT_CODE}, log: ${LOG_FILE})"
        N_FAIL=$((N_FAIL + 1))
        FAILED_INDICES+=("$TASK_ID")
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================="
echo "  Batch finished"
echo "  Done  : ${N_DONE}"
echo "  Skip  : ${N_SKIP}  (already existed)"
echo "  Failed: ${N_FAIL}"
if [[ ${#FAILED_INDICES[@]} -gt 0 ]]; then
    echo "  Failed task_ids: ${FAILED_INDICES[*]}"
    echo ""
    echo "  Logs for failed samples are in: ${LOG_DIR}/"
fi
echo "  Results : ${ROOT_DIR}/correlation_matching_results_<model>_*_all_tokens.json"
echo "  Logs    : ${LOG_DIR}/"
echo "================================================================="
