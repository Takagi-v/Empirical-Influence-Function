#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${HUAWEI_PROCESSED_DIR:-data/huawei_data/processed_full_clean}"
MODEL_PATH_VALUE="${MODEL_PATH:-}"
ANNOTATE_MODEL_VALUE="${ANNOTATE_MODEL:-}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LANGUAGE="auto"
RUN_PREFIX=""
MAX_ROWS="${PREPARE_MAX_ROWS:-0}"
MAX_ACCEPTED_ROWS="${PREPARE_MAX_ACCEPTED_ROWS:-0}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-4096}"
MAX_TEACHER_EDGES="${MAX_TEACHER_EDGES:-64}"
STRIP_CJK_COMMENTS="${STRIP_CJK_COMMENTS:-1}"
STRUCTURAL_ONLY="${STRUCTURAL_ONLY:-0}"
SKIP_ANNOTATION="${SKIP_ANNOTATION:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${HUAWEI_BATCH_DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_FILES=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/data_process/run_huawei_batch_annotation.sh [options] [raw1.jsonl raw2.jsonl ...]

Serially run scripts/data_process/build_fim_annotation_data.py for multiple
Huawei FIM JSONL files. Language is intentionally fixed to auto by default.

Input discovery:
  1. Positional raw JSONL paths, if provided.
  2. HUAWEI_RAW_DATA_LIST, whitespace-separated.
  3. TRAIN_DATA_1 ... TRAIN_DATA_20.

Common options:
  --output-root DIR          Parent output dir. Default: $HUAWEI_PROCESSED_DIR or data/huawei_data/processed_full_clean.
  --model-path PATH          Tokenizer/base model path. Default: $MODEL_PATH.
  --annotate-model NAME      OpenAI-compatible annotation model. Default: $ANNOTATE_MODEL.
  --workers N                Annotation workers. Default: $NUM_WORKERS or 8.
  --run-prefix NAME          Prefix run/output dirs with NAME_.
  --max-rows N               Read at most N rows per file; 0 = all.
  --max-accepted-rows N      Stop after N accepted rows per file; 0 = all.
  --model-max-length N       Tokenized ChatML max length. Default: 4096.
  --max-teacher-edges N      Max LLM-added edges per row. Default: 64.
  --structural-only          Do not call LLM; keep tree-sitter/local bridge edges.
  --skip-annotation          Only write canonical/chatml/report.
  --force                    Ignore annotation cache.
  --no-strip-cjk-comments    Keep comments containing CJK chars.
  --dry-run                  Print commands without running.

Outputs per input:
  <output-root>/<run-name>/<run-name>_canonical.jsonl
  <output-root>/<run-name>/<run-name>_chatml.jsonl
  <output-root>/<run-name>/<run-name>_compact.jsonl
  <output-root>/<run-name>/<run-name>_report.json
EOF
}

sanitize_stem() {
  local name="$1"
  name="${name%.jsonl}"
  name="${name%.json}"
  name="${name//[^A-Za-z0-9_]/_}"
  name="$(printf '%s' "$name" | sed -E 's/_+/_/g; s/^_//; s/_$//')"
  if [[ -z "$name" ]]; then
    name="huawei_data"
  fi
  printf '%s' "$name"
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH_VALUE="$2"
      shift 2
      ;;
    --annotate-model)
      ANNOTATE_MODEL_VALUE="$2"
      shift 2
      ;;
    --workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --run-prefix)
      RUN_PREFIX="$2"
      shift 2
      ;;
    --max-rows)
      MAX_ROWS="$2"
      shift 2
      ;;
    --max-accepted-rows)
      MAX_ACCEPTED_ROWS="$2"
      shift 2
      ;;
    --model-max-length)
      MODEL_MAX_LENGTH="$2"
      shift 2
      ;;
    --max-teacher-edges)
      MAX_TEACHER_EDGES="$2"
      shift 2
      ;;
    --structural-only)
      STRUCTURAL_ONLY=1
      shift
      ;;
    --skip-annotation)
      SKIP_ANNOTATION=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --no-strip-cjk-comments)
      STRIP_CJK_COMMENTS=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

if [[ "${#RAW_FILES[@]}" -eq 0 && -n "${HUAWEI_RAW_DATA_LIST:-}" ]]; then
  read -r -a RAW_FILES <<< "$HUAWEI_RAW_DATA_LIST"
fi

if [[ "${#RAW_FILES[@]}" -eq 0 ]]; then
  for i in $(seq 1 20); do
    var="TRAIN_DATA_$i"
    value="${!var:-}"
    if [[ -n "$value" ]]; then
      RAW_FILES+=("$value")
    fi
  done
fi

if [[ "${#RAW_FILES[@]}" -eq 0 ]]; then
  echo "No input JSONL files. Pass paths or set HUAWEI_RAW_DATA_LIST / TRAIN_DATA_1..20." >&2
  exit 2
fi
if [[ -z "$MODEL_PATH_VALUE" ]]; then
  echo "MODEL_PATH is required. Set MODEL_PATH or pass --model-path." >&2
  exit 2
fi
if [[ -z "$ANNOTATE_MODEL_VALUE" ]]; then
  echo "ANNOTATE_MODEL is required. Set ANNOTATE_MODEL or pass --annotate-model." >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
echo "[huawei-batch] files=${#RAW_FILES[@]}"
echo "[huawei-batch] output_root=$OUTPUT_ROOT"
echo "[huawei-batch] language=$LANGUAGE workers=$NUM_WORKERS dry_run=$DRY_RUN"

declare -A SEEN_RUN_NAMES=()
idx=0
for raw in "${RAW_FILES[@]}"; do
  idx=$((idx + 1))
  if [[ ! -f "$raw" ]]; then
    echo "[huawei-batch] missing input: $raw" >&2
    exit 1
  fi

  stem="$(sanitize_stem "$(basename "$raw")")"
  if [[ -n "$RUN_PREFIX" ]]; then
    stem="$(sanitize_stem "${RUN_PREFIX}_${stem}")"
  fi
  run_name="$stem"
  if [[ -n "${SEEN_RUN_NAMES[$run_name]:-}" ]]; then
    run_name="${stem}_${idx}"
  fi
  SEEN_RUN_NAMES[$run_name]=1

  out_dir="$OUTPUT_ROOT/$run_name"
  cmd=(
    "$PYTHON_BIN" scripts/data_process/build_fim_annotation_data.py
    --input "$raw"
    --output-dir "$out_dir"
    --model-path "$MODEL_PATH_VALUE"
    --annotate-model "$ANNOTATE_MODEL_VALUE"
    --language "$LANGUAGE"
    --run-name "$run_name"
    --num-workers "$NUM_WORKERS"
    --model-max-length "$MODEL_MAX_LENGTH"
    --max-teacher-edges "$MAX_TEACHER_EDGES"
  )
  if [[ "$MAX_ROWS" != "0" ]]; then
    cmd+=(--max-rows "$MAX_ROWS")
  fi
  if [[ "$MAX_ACCEPTED_ROWS" != "0" ]]; then
    cmd+=(--max-accepted-rows "$MAX_ACCEPTED_ROWS")
  fi
  if [[ "$STRUCTURAL_ONLY" == "1" ]]; then
    cmd+=(--structural-only)
  fi
  if [[ "$SKIP_ANNOTATION" == "1" ]]; then
    cmd+=(--skip-annotation)
  fi
  if [[ "$FORCE" == "1" ]]; then
    cmd+=(--force)
  fi
  if [[ "$STRIP_CJK_COMMENTS" == "0" ]]; then
    cmd+=(--no-strip-cjk-comments)
  fi

  echo "[huawei-batch] $idx/${#RAW_FILES[@]} raw=$raw run=$run_name"
  echo "[huawei-batch] command:"
  print_cmd "${cmd[@]}"
  if [[ "$DRY_RUN" != "1" ]]; then
    "${cmd[@]}"
  fi
done
