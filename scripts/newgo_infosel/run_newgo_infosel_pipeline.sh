#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_newgo_infosel_pipeline.sh — FULLY-AUTOMATIC infosel pipeline for the new
# enterprise-Go FIM dataset, end to end on this A100 box:
#
#   Phase 0  extract the AI4Go eval framework (idempotent)
#   Phase 1  convert {prompt,response} train rows -> compact input_ids/label      [env: build]
#   Phase 2  teacher-NLL label the compact rows (comp_teacher_nll) with vLLM      [env: vllm ]
#   Phase 3  LoRA-train the infosel run (token_select ON) + CE baseline (OFF)     [env: build]
#   Phase 4  generate on the 301 test prompts + score with the AI4Go metric       [env: build]
#   Phase 5  print the infosel-vs-CE metric comparison
#
# Phases are skip-if-done (set FORCE=1 to redo). GPU is used one phase at a time.
#
# Usage:
#   bash scripts/newgo_infosel/run_newgo_infosel_pipeline.sh
#   FORCE=1 TARGET=models/Qwen2.5-Coder-7B-Instruct bash scripts/newgo_infosel/run_newgo_infosel_pipeline.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
REPO=/home/v-murongma/code/code-corr-annotation
CONDA=/home/v-murongma/miniconda3
cd "$REPO"

ENV_BUILD=${ENV_BUILD:-code-attribute}     # torch+transformers+peft+eval deps
ENV_VLLM=${ENV_VLLM:-vllm}                 # vLLM teacher labeling
TEACHER=${TEACHER:-models/Qwen3-Coder-30B-A3B-Instruct}
TARGET=${TARGET:-models/Qwen2.5-Coder-7B-Instruct}
SEED=${SEED:-42}
THRESHOLD=${THRESHOLD:-2.0}
MAX_LEN=${MAX_LEN:-2048}                    # target train seq len
TEACHER_MAX_LEN=${TEACHER_MAX_LEN:-4096}
SUBSAMPLE=${SUBSAMPLE:-5000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MAX_NBT=${MAX_NBT:-2048}                    # vLLM max_num_batched_tokens (bounds logprobs OOM)
FORCE=${FORCE:-0}

TRAIN_RAW=${TRAIN_RAW:-data/new_go_data/train/cloud_core_test_25.JunJunly_GoOnly_length_filter.jsonl}
TEST=${TEST:-data/new_go_data/test/processed_part1.jsonl}
# RAR auto-discovers the AI4Go eval archive in the repo root; AI4GO_DIR is where
# it gets extracted. Both are env-overridable.
RAR=${RAR:-$(ls "$REPO"/*.rar 2>/dev/null | head -1)}
AI4GO_DIR=${AI4GO_DIR:-$HOME/ai4go_eval/framework}

TD=data/new_go_data/train_data
COMPACT=${COMPACT:-$TD/newgo_infosel_${SUBSAMPLE/000/k}_compact.jsonl}
TLABELED=${TLABELED:-$TD/newgo_infosel_${SUBSAMPLE/000/k}_tlabeled_compact.jsonl}
INFOSEL_CFG=${INFOSEL_CFG:-scripts/newgo_infosel/configs/lora_infosel_newgo5k_7b_s${SEED}.yaml}
CE_CFG=${CE_CFG:-scripts/newgo_infosel/configs/lora_ce_newgo5k_7b_s${SEED}.yaml}
OUTDIR=outputs/newgo_eval
mkdir -p "$OUTDIR" "$TD"

S=$(date +%s); LOG(){ echo "[$(date +%T) +$(( $(date +%s)-S ))s] $*"; }
run_env(){ local e="$1"; shift; "$CONDA/bin/conda" run --no-capture-output -n "$e" "$@"; }
ckpt_of(){ ls -d "outputs/$1/checkpoints/checkpoint-"* 2>/dev/null | sort -t- -k2 -n | tail -1; }

# ── Phase 0: AI4Go eval framework ────────────────────────────────────────────
if [ ! -f "$AI4GO_DIR/eval/eval.py" ]; then
  LOG "Phase 0: extracting AI4Go eval framework -> $AI4GO_DIR"
  [ -n "$RAR" ] || { LOG "FATAL: no .rar found in $REPO (set RAR=...)"; exit 1; }
  mkdir -p "$AI4GO_DIR"; bsdtar -xf "$RAR" --strip-components=1 -C "$AI4GO_DIR"
fi
[ -f "$AI4GO_DIR/eval/eval.py" ] || { LOG "FATAL: AI4Go eval.py missing"; exit 1; }

# ── Phase 1: convert to compact ──────────────────────────────────────────────
if [ "$FORCE" = 1 ] || [ ! -s "$COMPACT" ]; then
  LOG "Phase 1: build compact ($SUBSAMPLE rows, target=$TARGET)"
  run_env "$ENV_BUILD" python3 scripts/newgo_infosel/build_newgo_compact.py \
    --in "$TRAIN_RAW" --out "$COMPACT" --target_model "$TARGET" \
    --subsample "$SUBSAMPLE" --seed "$SEED" --max_len "$MAX_LEN" || { LOG "build FAILED"; exit 1; }
else LOG "Phase 1: skip (exists $COMPACT)"; fi

# ── Phase 2: teacher-NLL labeling (vLLM) ─────────────────────────────────────
scored_frac(){ python3 - "$1" <<'PY'
import json,sys
n=s=0
for i,l in enumerate(open(sys.argv[1])):
    if i>=500: break
    n+=1; s+= 1 if json.loads(l).get("comp_teacher_nll") else 0
print(f"{s}/{n}")
PY
}
if [ "$FORCE" = 1 ] || [ ! -s "$TLABELED" ]; then
  LOG "Phase 2: teacher-NLL labeling with $TEACHER (vLLM)"
  HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  run_env "$ENV_VLLM" python3 scripts/newgo_infosel/label_newgo_nll_vllm.py \
    --in "$COMPACT" --out "$TLABELED" --teacher "$TEACHER" \
    --tp 1 --max_len "$TEACHER_MAX_LEN" --chunk 64 \
    --gpu_mem_util "$GPU_MEM_UTIL" --max_num_batched_tokens "$MAX_NBT" || { LOG "label FAILED"; exit 1; }
else LOG "Phase 2: skip (exists $TLABELED)"; fi
LOG "labels present; first500 scored=$(scored_frac "$TLABELED")"
[ -s "$TLABELED" ] || { LOG "FATAL: no labeled file"; exit 1; }

# ── Phase 3: LoRA train infosel + CE baseline ────────────────────────────────
# Register the labeled set in the global dataset registry (idempotent) so
# run_train.py can resolve data.train by key.
run_env "$ENV_BUILD" python3 scripts/newgo_infosel/register_dataset.py \
  --key "$(basename "$TLABELED" _compact.jsonl)" --path "$TLABELED" \
  --n "$(wc -l < "$TLABELED" 2>/dev/null || echo 0)" || true
export WANDB_DISABLED=true HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for cfg in "$INFOSEL_CFG" "$CE_CFG"; do
  name=$(basename "$cfg" .yaml)
  if [ "$FORCE" = 1 ] || [ -z "$(ckpt_of "$name")" ]; then
    LOG "Phase 3: train $name"
    run_env "$ENV_BUILD" env WANDB_DISABLED=true HF_HUB_OFFLINE=1 \
      python3 src/train/run_train.py --config "$cfg" || { LOG "train $name FAILED"; exit 1; }
  else LOG "Phase 3: skip $name (ckpt $(ckpt_of "$name"))"; fi
done

# ── Phase 4: generate on test + AI4Go metric ─────────────────────────────────
declare -A METRICS
for name in "$(basename "$INFOSEL_CFG" .yaml)" "$(basename "$CE_CFG" .yaml)"; do
  ck=$(ckpt_of "$name"); tag=${name%%_newgo*}; tag=${tag#lora_}
  mfile="$OUTDIR/${name}_metrics.json"
  if [ "$FORCE" = 1 ] || [ ! -s "$mfile" ]; then
    LOG "Phase 4: eval $name (adapter=$ck)"
    run_env "$ENV_BUILD" python3 scripts/newgo_infosel/eval_newgo.py \
      --base_model "$TARGET" --adapter "$ck" --test "$TEST" \
      --out_predictions "$OUTDIR/${name}_predictions.jsonl" \
      --out_metrics "$mfile" --ai4go_dir "$AI4GO_DIR" || { LOG "eval $name FAILED"; exit 1; }
  else LOG "Phase 4: skip $name (metrics exist)"; fi
  METRICS[$name]=$mfile
done

# ── Phase 5: comparison ──────────────────────────────────────────────────────
LOG "Phase 5: RESULTS (new-go test, $(wc -l < "$TEST") rows)"
python3 - "$OUTDIR/$(basename "$INFOSEL_CFG" .yaml)_metrics.json" \
          "$OUTDIR/$(basename "$CE_CFG" .yaml)_metrics.json" <<'PY'
import json,sys
def load(p):
    d=json.load(open(p))
    return {k:(v[0] if isinstance(v,list) else v) for k,v in d.items()}
inf=load(sys.argv[1]); ce=load(sys.argv[2])
keys=["avg_es","avg_bleu","rougeL","line_hit_pre","line_hit_rec","hit@0.5","acc"]
print(f"{'metric':14s} {'infosel':>10s} {'ce':>10s} {'delta':>10s}")
for k in keys:
    a=float(inf.get(k,0)); b=float(ce.get(k,0))
    print(f"{k:14s} {a:10.3f} {b:10.3f} {a-b:+10.3f}")
PY
LOG "DONE. predictions+metrics in $OUTDIR/"
