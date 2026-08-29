#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-check}"
if [[ $# -gt 0 ]]; then
  shift
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

run_batch_if_requested() {
  local raw_files=()
  if [[ $# -gt 0 ]]; then
    raw_files=("$@")
  elif [[ -n "${HUAWEI_RAW_DATA_LIST:-}" ]]; then
    # Whitespace-separated list. Keep paths space-free on deployment machines.
    # For paths that contain spaces, pass them as positional args instead.
    read -r -a raw_files <<< "$HUAWEI_RAW_DATA_LIST"
  fi

  if [[ "${#raw_files[@]}" -eq 0 ]]; then
    return 1
  fi

  local idx=0
  for raw in "${raw_files[@]}"; do
    idx=$((idx + 1))
    printf '[huawei-annotate] batch %d/%d raw=%s\n' "$idx" "${#raw_files[@]}" "$raw"
    HUAWEI_RAW_DATA="$raw"     HUAWEI_RAW_DATA_LIST=""     HUAWEI_CHATML_DATA=""     HUAWEI_CANONICAL_DATA=""     HUAWEI_PREPARE_REPORT=""     CHECK_RUN_NAME=""     FULL_RUN_NAME=""     VIS_RUN_NAME=""     OUTPUT_PATH=""     CACHE_PATH=""       bash "$0" "$MODE"
  done
  return 0
}

if run_batch_if_requested "$@"; then
  exit 0
fi

RAW_DATA="${HUAWEI_RAW_DATA:-${RAW_DATA:-${TRAIN_DATA:-/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/external/huawei_data/cloud_core_test_25.JunJunly_GoOnly_length_filter.jsonl}}}"
PROCESSED_DIR="${HUAWEI_PROCESSED_DIR:-/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/interim/huawei_data/processed}"
RAW_BASENAME="$(basename "$RAW_DATA")"
RAW_STEM="${RAW_BASENAME%.*}"
CHATML_DATA="${HUAWEI_CHATML_DATA:-$PROCESSED_DIR/${RAW_STEM}_chatml.jsonl}"
CANONICAL_DATA="${HUAWEI_CANONICAL_DATA:-$PROCESSED_DIR/${RAW_STEM}_canonical.jsonl}"
REPORT_DATA="${HUAWEI_PREPARE_REPORT:-$PROCESSED_DIR/${RAW_STEM}_prepare_report.json}"
PREPARE_MAX_ROWS="${PREPARE_MAX_ROWS:-0}"
PREPARE_MAX_ACCEPTED_ROWS="${PREPARE_MAX_ACCEPTED_ROWS:-0}"
STRIP_CJK_COMMENTS="${STRIP_CJK_COMMENTS:-1}"
MAX_TARGET_NONEMPTY_LINES="${MAX_TARGET_NONEMPTY_LINES:-10}"
MAX_TARGET_ROUGH_TOKENS="${MAX_TARGET_ROUGH_TOKENS:-192}"
MAX_TARGET_CHARS="${MAX_TARGET_CHARS:-1024}"
FILTER_GOFMT_VALID="${FILTER_GOFMT_VALID:-1}"
GOFMT_BIN="${GOFMT_BIN:-gofmt}"
VALIDATE_LIMIT="${VALIDATE_LIMIT:-100}"
CHECK_ROWS="${CHECK_ROWS:-50}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
ENABLE_VISUALIZE="${ENABLE_VISUALIZE:-1}"
ANNOTATION_MODE="${ANNOTATION_MODE:-agent}"
CHECK_RUN_NAME="${CHECK_RUN_NAME:-${RAW_STEM}_check_huawei_${ANNOTATION_MODE}}"
FULL_RUN_NAME="${FULL_RUN_NAME:-${RAW_STEM}_full_huawei_${ANNOTATION_MODE}}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme0n1/wenhao/models/Empirical-Influence-Function/Qwen2.5-Coder-7B-Instruct}"
OUT_DIR="${OUT_DIR:-outputs/go_singleline_fim_exp/huawei_annotation}"
RUN_DIR="${RUN_DIR:-runs/go_singleline_fim_exp/huawei_annotation}"
VIS_OUT_DIR="${VIS_OUT_DIR:-outputs/huawei_deploy}"

usage() {
  cat <<'EOF'
Usage: bash scripts/huawei_deploy/annotate.sh [prepare|validate|check|full|visualize|all] [raw1.jsonl raw2.jsonl ...]

Modes:
  prepare    Convert Huawei raw prompt/response/task_id JSONL to project ChatML/FIM JSONL.
  validate   Validate converted ChatML/FIM rows.
  check      prepare -> validate -> annotate CHECK_ROWS rows -> visualize by default.
  full       prepare -> validate -> annotate all rows.
  visualize  Build rich-sample CSV and HTML viewer for CHECK_RUN_NAME output.
  all        prepare -> validate -> check -> optional visualize -> full annotation.

Batch input:
  Pass multiple raw JSONL paths after MODE, or set HUAWEI_RAW_DATA_LIST="a.jsonl b.jsonl".
  The script runs the selected MODE sequentially for each file and auto-derives
  per-file chatml/canonical/report/cache/output names from each raw basename.

Important env vars:
  HUAWEI_RAW_DATA       Raw Huawei JSONL path. Fallbacks: RAW_DATA, TRAIN_DATA, then local default.
  HUAWEI_RAW_DATA_LIST  Optional whitespace-separated raw JSONL list for sequential batch runs.
  HUAWEI_CHATML_DATA    Converted ChatML output path.
  HUAWEI_CANONICAL_DATA Converted canonical output path.
  FORCE_PREPARE=1      Rebuild converted data even if outputs already exist.
  PREPARE_MAX_ROWS=N   Read only N raw rows; 0 means all rows.
  PREPARE_MAX_ACCEPTED_ROWS=N Stop after N accepted rows; 0 means no accepted-row limit.
  STRIP_CJK_COMMENTS=1 Remove Go comments containing CJK characters during prepare; default: 1.
  MAX_TARGET_NONEMPTY_LINES=N  Reject long targets by non-empty line count; default: 10.
  MAX_TARGET_ROUGH_TOKENS=N    Reject long targets by rough token count; default: 192.
  MAX_TARGET_CHARS=N           Reject long targets by char count; default: 1024.
  FILTER_GOFMT_VALID=1 Reject samples whose prefix+target+suffix is not parseable by gofmt; default: 1.
  GOFMT_BIN=gofmt      gofmt executable for syntax quality filtering.
  CHECK_ROWS=50        Number of rows for check mode.
  ENABLE_VISUALIZE=1   Build viewer after check/all; default: 1.
  VIS_OUT_DIR=PATH     Visualization CSV/HTML output dir. Default: outputs/huawei_deploy

API env vars are required for check/full/all:
  OPENAI_API_KEY, OPENAI_BASE_URL, ANNOTATE_MODEL
  If REQUIRE_HUAWEI_GATEWAY=1, also require HW_APPKEY, HW_OPERATOR.
  For local non-Huawei OpenAI-compatible tests, set REQUIRE_HUAWEI_GATEWAY=0.
EOF
}

log() { printf '[huawei-annotate] %s\n' "$*"; }

prepare() {
  if [[ ! -f "$RAW_DATA" ]]; then
    echo "Raw data not found: $RAW_DATA" >&2
    exit 1
  fi
  if [[ "$FORCE_PREPARE" != "1" && -s "$CHATML_DATA" && -s "$CANONICAL_DATA" ]]; then
    log "prepare skipped: $CHATML_DATA and $CANONICAL_DATA already exist"
    return
  fi
  mkdir -p "$PROCESSED_DIR"
  local max_args=()
  if [[ "$PREPARE_MAX_ROWS" != "0" ]]; then
    max_args+=(--max-rows "$PREPARE_MAX_ROWS")
  fi
  if [[ "$PREPARE_MAX_ACCEPTED_ROWS" != "0" ]]; then
    max_args+=(--max-accepted-rows "$PREPARE_MAX_ACCEPTED_ROWS")
  fi
  if [[ "$STRIP_CJK_COMMENTS" == "1" ]]; then
    max_args+=(--strip-cjk-comments)
  fi
  if [[ "$MAX_TARGET_NONEMPTY_LINES" != "0" ]]; then
    max_args+=(--max-target-nonempty-lines "$MAX_TARGET_NONEMPTY_LINES")
  fi
  if [[ "$MAX_TARGET_ROUGH_TOKENS" != "0" ]]; then
    max_args+=(--max-target-rough-tokens "$MAX_TARGET_ROUGH_TOKENS")
  fi
  if [[ "$MAX_TARGET_CHARS" != "0" ]]; then
    max_args+=(--max-target-chars "$MAX_TARGET_CHARS")
  fi
  if [[ "$FILTER_GOFMT_VALID" == "1" ]]; then
    max_args+=(--filter-gofmt-valid --gofmt-bin "$GOFMT_BIN")
  fi
  log "prepare raw=$RAW_DATA"
  log "prepare chatml=$CHATML_DATA"
  log "prepare canonical=$CANONICAL_DATA"
  log "prepare cleaning strip_cjk_comments=$STRIP_CJK_COMMENTS max_lines=$MAX_TARGET_NONEMPTY_LINES max_tokens=$MAX_TARGET_ROUGH_TOKENS max_chars=$MAX_TARGET_CHARS filter_gofmt_valid=$FILTER_GOFMT_VALID gofmt_bin=$GOFMT_BIN max_accepted=$PREPARE_MAX_ACCEPTED_ROWS"
  python scripts/huawei_deploy/build_huawei_fim_chatml.py \
    --input-path "$RAW_DATA" \
    --chatml-output "$CHATML_DATA" \
    --canonical-output "$CANONICAL_DATA" \
    --report-output "$REPORT_DATA" \
    "${max_args[@]}"
}

validate() {
  log "validate input=$CHATML_DATA limit=$VALIDATE_LIMIT"
  python scripts/huawei_deploy/check_chatml_fim_input.py \
    --input_path "$CHATML_DATA" \
    --limit "$VALIDATE_LIMIT"
}

require_api_env() {
  : "${OPENAI_API_KEY:?Set OPENAI_API_KEY for the OpenAI-compatible endpoint.}"
  : "${ANNOTATE_MODEL:?Set ANNOTATE_MODEL to the deployed model name.}"
  local require_huawei="${REQUIRE_HUAWEI_GATEWAY:-1}"
  if [[ "$require_huawei" == "1" ]]; then
    : "${HW_APPKEY:?Set HW_APPKEY to the Huawei X-HW-APPKEY value, or set REQUIRE_HUAWEI_GATEWAY=0 for local non-Huawei tests.}"
    : "${HW_OPERATOR:?Set HW_OPERATOR to your work-id/operator value, or set REQUIRE_HUAWEI_GATEWAY=0 for local non-Huawei tests.}"
  else
    : "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL for local non-Huawei OpenAI-compatible tests.}"
  fi
}

configure_api_env() {
  local require_huawei="${REQUIRE_HUAWEI_GATEWAY:-1}"
  if [[ "$require_huawei" == "1" ]]; then
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://apigw-cn-south02.huawei.com/api/v1}"
    export HW_ID="${HW_ID:-com.huawei.ipd.coretool.coreai}"
    export HW_APP_ID="${HW_APP_ID:-$HW_ID}"
    export HW_SCENE="${HW_SCENE:-test}"
    export HW_ENABLE_THINKING="${HW_ENABLE_THINKING:-0}"
    export ANNOTATE_HUAWEI_TEMPLATE_MODE="${ANNOTATE_HUAWEI_TEMPLATE_MODE:-1}"
    export ANNOTATE_HUAWEI_CONTENT_LIST="${ANNOTATE_HUAWEI_CONTENT_LIST:-1}"
    export ANNOTATE_STREAM="${ANNOTATE_STREAM:-1}"
    export ANNOTATE_FALLBACK_ON_CHAT_ERROR="${ANNOTATE_FALLBACK_ON_CHAT_ERROR:-1}"
    export ANNOTATE_HTTP_PROXY_NONE="${ANNOTATE_HTTP_PROXY_NONE:-1}"
    export ANNOTATE_VERIFY_SSL="${ANNOTATE_VERIFY_SSL:-0}"
  else
    unset HW_ID HW_APPKEY HW_APP_ID HW_SCENE HW_OPERATOR
    unset HUAWEI_HW_ID HUAWEI_HW_APPKEY HUAWEI_APP_ID HUAWEI_SCENE HUAWEI_OPERATOR
    unset ANNOTATE_EXTRA_HEADERS_JSON ANNOTATE_EXTRA_BODY_JSON
    export HW_ENABLE_THINKING="${HW_ENABLE_THINKING:-0}"
    export ANNOTATE_HUAWEI_TEMPLATE_MODE="${ANNOTATE_HUAWEI_TEMPLATE_MODE:-0}"
    export ANNOTATE_HUAWEI_CONTENT_LIST="${ANNOTATE_HUAWEI_CONTENT_LIST:-0}"
    export ANNOTATE_STREAM="${ANNOTATE_STREAM:-0}"
    export ANNOTATE_FALLBACK_ON_CHAT_ERROR="${ANNOTATE_FALLBACK_ON_CHAT_ERROR:-0}"
    export ANNOTATE_HTTP_PROXY_NONE="${ANNOTATE_HTTP_PROXY_NONE:-0}"
    export ANNOTATE_VERIFY_SSL="${ANNOTATE_VERIFY_SSL:-1}"
  fi

  export ANNOTATE_TEMPERATURE="${ANNOTATE_TEMPERATURE:-0.2}"
  export ANNOTATE_MIN_REQUEST_INTERVAL="${ANNOTATE_MIN_REQUEST_INTERVAL:-1.0}"
  export ANNOTATE_MAX_RETRIES="${ANNOTATE_MAX_RETRIES:-8}"
  export ANNOTATE_RETRY_BASE_SLEEP="${ANNOTATE_RETRY_BASE_SLEEP:-10}"
  export ANNOTATE_MAX_TOKENS="${ANNOTATE_MAX_TOKENS:-2048}"
}

run_annotation() {
  local mode="$1"
  local run_name="$2"
  local max_rows="$3"
  local overwrite_args=()
  local max_rows_args=()
  if [[ "$mode" == "check" ]]; then
    max_rows_args=(--max_rows "$max_rows")
    overwrite_args=(--overwrite_cache)
  elif [[ "$mode" != "full" ]]; then
    echo "run_annotation mode must be check or full, got: $mode" >&2
    exit 2
  fi

  configure_api_env
  mkdir -p "$OUT_DIR" "$RUN_DIR"
  local output_path="${OUTPUT_PATH:-$OUT_DIR/${run_name}_compact.jsonl}"
  local cache_path="${CACHE_PATH:-$RUN_DIR/${run_name}.annotation_cache.jsonl}"

  echo "mode=$mode"
  echo "input=$CHATML_DATA"
  echo "output=$output_path"
  echo "cache=$cache_path"
  echo "model_path=$MODEL_PATH"
  echo "annotate_model=$ANNOTATE_MODEL"
  echo "require_huawei_gateway=${REQUIRE_HUAWEI_GATEWAY:-1}"
  echo "huawei_template_mode=${ANNOTATE_HUAWEI_TEMPLATE_MODE:-0}"
  echo "huawei_content_list=${ANNOTATE_HUAWEI_CONTENT_LIST:-0}"
  echo "annotate_stream=${ANNOTATE_STREAM:-0}"
  echo "fallback_on_chat_error=${ANNOTATE_FALLBACK_ON_CHAT_ERROR:-0}"
  echo "workers=${NUM_WORKERS:-4}"

  python scripts/go_singleline_fim_exp/annotate_chatml_with_src_annotate.py     --input_path "$CHATML_DATA"     --output_path "$output_path"     --model_name_or_path "$MODEL_PATH"     "${max_rows_args[@]}"     --num_workers "${NUM_WORKERS:-4}"     --annotation_mode "$ANNOTATION_MODE"     --max_rounds "${MAX_ROUNDS:-6}"     --annotation_cache_path "$cache_path"     --flush_every "${FLUSH_EVERY:-20}"     "${overwrite_args[@]}"
}

annotate_check() {
  require_api_env
  log "annotate check rows=$CHECK_ROWS run=$CHECK_RUN_NAME"
  run_annotation check "$CHECK_RUN_NAME" "$CHECK_ROWS"
}

annotate_full() {
  require_api_env
  log "annotate full run=$FULL_RUN_NAME"
  run_annotation full "$FULL_RUN_NAME" "0"
}

visualize() {
  local run_name="${VIS_RUN_NAME:-$CHECK_RUN_NAME}"
  local compact="$OUT_DIR/${run_name}_compact.jsonl"
  local csv="$VIS_OUT_DIR/${run_name}_rich_samples.csv"
  local html="$VIS_OUT_DIR/${run_name}_viewer.html"
  if [[ ! -s "$compact" ]]; then
    echo "Compact output not found for visualization: $compact" >&2
    exit 1
  fi
  mkdir -p "$VIS_OUT_DIR"
  log "visualize compact=$compact"
  log "visualize csv=$csv"
  log "visualize html=$html"
  python src/viz/viz_annotation/find_annotation_rich_samples.py \
    --raw_data_path "$CHATML_DATA" \
    --edge_data_path "$compact" \
    --model_path "$MODEL_PATH" \
    --top_n 50 \
    --sort_by total \
    --output_csv "$csv" \
    --local_files_only
  python src/viz/viz_annotation/build_dynamic_annotation_viewer.py \
    --edge_data_path "$compact" \
    --samples_csv "$csv" \
    --top_n_from_csv 50 \
    --model_path "$MODEL_PATH" \
    --output_path "$html" \
    --local_files_only
  log "viewer=$html"
  log "serve from repo root: python -m http.server 8000"
  log "open: http://<server>:8000/$html"
}

case "$MODE" in
  prepare)
    prepare
    ;;
  validate)
    prepare
    validate
    ;;
  check)
    prepare
    validate
    annotate_check
    if [[ "$ENABLE_VISUALIZE" == "1" ]]; then visualize; fi
    ;;
  full)
    prepare
    validate
    annotate_full
    ;;
  visualize)
    prepare
    validate
    visualize
    ;;
  all)
    prepare
    validate
    annotate_check
    if [[ "$ENABLE_VISUALIZE" == "1" ]]; then visualize; fi
    annotate_full
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
