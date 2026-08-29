#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="full"
RAW_FILES=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/huawei_deploy/annotate_simple.sh [options] data1.jsonl [data2.jsonl ...]

Minimal Huawei full annotation wrapper.

Common options:
  --check                 Run small check mode instead of full mode.
  --full                  Run full mode. Default.
  --model-path PATH       Local tokenizer/base model path.
  --annotate-model NAME   Huawei/OpenAI-compatible model name.
  --base-url URL          OpenAI-compatible base URL.
  --api-key KEY           OpenAI API key.
  --hw-appkey KEY         Huawei X-HW-APPKEY.
  --hw-operator ID        Huawei operator/work-id.
  --workers N             Number of annotation workers. Default: 8.
  --out-dir DIR           Prepared/compact output dir. Default: /mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/interim/huawei_data/processed_full_clean.
  --run-dir DIR           Runtime/cache/log dir. Default: runs/huawei_deploy.
  --force-prepare         Rebuild prepared chatml/canonical data.
  --no-gofmt-filter       Disable gofmt full-code syntax quality filter.

Required in practice:
  data*.jsonl, ANNOTATE_MODEL, MODEL_PATH, OPENAI_BASE_URL, OPENAI_API_KEY,
  HW_APPKEY, HW_OPERATOR.

The script sets production-friendly defaults:
  oneshot annotation, Huawei template mode, stream response, timeout,
  cache reuse, FORCE_PREPARE=0, and no visualization in full mode.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      MODE="check"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --model-path)
      export MODEL_PATH="$2"
      shift 2
      ;;
    --annotate-model)
      export ANNOTATE_MODEL="$2"
      shift 2
      ;;
    --base-url)
      export OPENAI_BASE_URL="$2"
      shift 2
      ;;
    --api-key)
      export OPENAI_API_KEY="$2"
      shift 2
      ;;
    --hw-appkey)
      export HW_APPKEY="$2"
      shift 2
      ;;
    --hw-operator)
      export HW_OPERATOR="$2"
      shift 2
      ;;
    --workers)
      export NUM_WORKERS="$2"
      shift 2
      ;;
    --out-dir)
      export HUAWEI_PROCESSED_DIR="$2"
      export OUT_DIR="$2"
      shift 2
      ;;
    --run-dir)
      export RUN_DIR="$2"
      shift 2
      ;;
    --force-prepare)
      export FORCE_PREPARE=1
      shift
      ;;
    --no-gofmt-filter)
      export FILTER_GOFMT_VALID=0
      shift
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        RAW_FILES+=("$1")
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      RAW_FILES+=("$1")
      shift
      ;;
  esac
done

if [[ "${#RAW_FILES[@]}" -eq 0 ]]; then
  for name in TRAIN_DATA_1 TRAIN_DATA_2 TRAIN_DATA_3; do
    value="${!name:-}"
    if [[ -n "$value" ]]; then
      RAW_FILES+=("$value")
    fi
  done
fi

if [[ "${#RAW_FILES[@]}" -eq 0 ]]; then
  echo "No raw JSONL data paths provided. Pass data*.jsonl paths or set TRAIN_DATA_1..3." >&2
  exit 2
fi

export REQUIRE_HUAWEI_GATEWAY="${REQUIRE_HUAWEI_GATEWAY:-1}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://apigw-cn-south02.huawei.com/api/v1}"
export HW_ID="${HW_ID:-com.huawei.ipd.coretool.coreai}"
export HW_APP_ID="${HW_APP_ID:-$HW_ID}"
export HW_SCENE="${HW_SCENE:-test}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$HW_ID}"

export MODEL_PATH="${MODEL_PATH:-/home/model_project/CCCodeGenerationTrain/infer_format/}"
export ANNOTATE_MODEL="${ANNOTATE_MODEL:-fa6c020a-06e3-4a4f-8840-2951e5ef934d}"

export HUAWEI_PROCESSED_DIR="${HUAWEI_PROCESSED_DIR:-/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/interim/huawei_data/processed_full_clean}"
export OUT_DIR="${OUT_DIR:-$HUAWEI_PROCESSED_DIR}"
export RUN_DIR="${RUN_DIR:-runs/huawei_deploy}"
export VIS_OUT_DIR="${VIS_OUT_DIR:-outputs/huawei_deploy}"

export FORCE_PREPARE="${FORCE_PREPARE:-0}"
export PREPARE_MAX_ACCEPTED_ROWS="${PREPARE_MAX_ACCEPTED_ROWS:-0}"
if [[ "$MODE" == "check" ]]; then
  export CHECK_ROWS="${CHECK_ROWS:-10}"
  export PREPARE_MAX_ACCEPTED_ROWS="${PREPARE_MAX_ACCEPTED_ROWS:-10}"
  export ENABLE_VISUALIZE="${ENABLE_VISUALIZE:-1}"
else
  export ENABLE_VISUALIZE="${ENABLE_VISUALIZE:-0}"
fi

export ANNOTATION_MODE="${ANNOTATION_MODE:-oneshot}"
export MAX_ROUNDS="${MAX_ROUNDS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export ANNOTATE_MIN_REQUEST_INTERVAL="${ANNOTATE_MIN_REQUEST_INTERVAL:-1.0}"
export ANNOTATE_REQUEST_TIMEOUT="${ANNOTATE_REQUEST_TIMEOUT:-180}"
export ANNOTATE_MAX_RETRIES="${ANNOTATE_MAX_RETRIES:-3}"
export ANNOTATE_RETRY_BASE_SLEEP="${ANNOTATE_RETRY_BASE_SLEEP:-10}"

export ANNOTATE_HTTP_PROXY_NONE="${ANNOTATE_HTTP_PROXY_NONE:-1}"
export ANNOTATE_VERIFY_SSL="${ANNOTATE_VERIFY_SSL:-0}"
export HW_ENABLE_THINKING="${HW_ENABLE_THINKING:-0}"
export ANNOTATE_HUAWEI_TEMPLATE_MODE="${ANNOTATE_HUAWEI_TEMPLATE_MODE:-1}"
export ANNOTATE_HUAWEI_CONTENT_LIST="${ANNOTATE_HUAWEI_CONTENT_LIST:-1}"
export ANNOTATE_STREAM="${ANNOTATE_STREAM:-1}"
export ANNOTATE_FALLBACK_ON_CHAT_ERROR="${ANNOTATE_FALLBACK_ON_CHAT_ERROR:-1}"
export ANNOTATE_TEMPERATURE="${ANNOTATE_TEMPERATURE:-0.2}"

export STRIP_CJK_COMMENTS="${STRIP_CJK_COMMENTS:-1}"
export MAX_TARGET_NONEMPTY_LINES="${MAX_TARGET_NONEMPTY_LINES:-10}"
export MAX_TARGET_ROUGH_TOKENS="${MAX_TARGET_ROUGH_TOKENS:-192}"
export MAX_TARGET_CHARS="${MAX_TARGET_CHARS:-1024}"
export FILTER_GOFMT_VALID="${FILTER_GOFMT_VALID:-1}"
if [[ -n "${GOROOT:-}" && -x "$GOROOT/bin/gofmt" ]]; then
  export GOFMT_BIN="${GOFMT_BIN:-$GOROOT/bin/gofmt}"
else
  export GOFMT_BIN="${GOFMT_BIN:-gofmt}"
fi

: "${HW_APPKEY:?Set HW_APPKEY or pass --hw-appkey.}"
: "${HW_OPERATOR:?Set HW_OPERATOR or pass --hw-operator.}"

mkdir -p "$OUT_DIR" "$RUN_DIR" "$VIS_OUT_DIR"

echo "[annotate-simple] mode=$MODE"
echo "[annotate-simple] data_count=${#RAW_FILES[@]}"
echo "[annotate-simple] model_path=$MODEL_PATH"
echo "[annotate-simple] annotate_model=$ANNOTATE_MODEL"
echo "[annotate-simple] out_dir=$OUT_DIR"
echo "[annotate-simple] run_dir=$RUN_DIR"
echo "[annotate-simple] workers=$NUM_WORKERS mode=$ANNOTATION_MODE force_prepare=$FORCE_PREPARE"

exec bash scripts/huawei_deploy/annotate.sh "$MODE" "${RAW_FILES[@]}"
