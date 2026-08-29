# new-go infosel pipeline

End-to-end **informative-token-selection (infosel)** training + evaluation for the new
enterprise-Go FIM dataset, on a single GPU. Everything needed to run the pipeline lives in
this directory.

**infosel** = train the target model with cross-entropy but **drop the completion tokens a
strong teacher finds "missing-info"** (teacher FIM NLL > threshold). The idea: tokens the
teacher can't predict from the context are un-inferable / memorization noise, so excluding them
from the loss can improve the student's FIM accuracy. We compare **infosel vs a matched CE
baseline** (same data, no token dropping) on the held-out test set.

```
{prompt,response}                  teacher NLL                infosel LoRA          AI4Go metric
  raw train  ──build──▶  compact  ──label(vLLM)──▶ tlabeled  ──train──▶ adapter ──eval──▶ es/bleu/rougeL/acc
  (199,923)   input_ids/label       comp_teacher_nll         (+CE base)          (301 test rows)
```

---

## Contents

| File | Role |
|------|------|
| `run_newgo_infosel_pipeline.sh` | **Orchestrator** — runs all phases end-to-end, skip-if-done |
| `setup_env.sh` | Build the two conda envs (train/eval + isolated vLLM) |
| `build_newgo_compact.py` | Phase 1 — convert `{prompt,response,task_id}` → compact `input_ids`/`label` (+ response char-spans) |
| `register_dataset.py` | Register the labeled set in the global `configs/datasets.yaml` (idempotent) |
| `label_newgo_nll_vllm.py` | Phase 2 — teacher-NLL labeling with vLLM → `comp_teacher_nll` |
| `eval_newgo.py` | Phase 4 — generate on the test prompts + score with the AI4Go metric |
| `configs/lora_infosel_newgo5k_7b_s42.yaml` | Training config — infosel (`token_select` ON, threshold 2.0) |
| `configs/lora_ce_newgo5k_7b_s42.yaml` | Training config — matched CE baseline (`token_select` OFF) |

Training itself uses the repo's `src/train/run_train.py` (unchanged); these configs point it at
the registered dataset.

---

## Prerequisites

- **GPU**: 1× A100 80GB (or similar). The teacher (30B-A3B) and 7B LoRA run one phase at a time.
- **Models** (local, under `models/`):
  - Teacher: `models/Qwen3-Coder-30B-A3B-Instruct` (30B-total / 3B-active MoE).
  - Target: `models/Qwen2.5-Coder-7B-Instruct` (the model we fine-tune + evaluate).
- **Data** (already in the repo):
  - Train: `data/new_go_data/train/cloud_core_test_25.JunJunly_GoOnly_length_filter.jsonl` (199,923 rows).
  - Test:  `data/new_go_data/test/processed_part1.jsonl` (301 rows).
- **Eval framework**: the AI4Go eval `.rar` at the repo root (the metric code).
  The orchestrator auto-discovers it and extracts to `AI4GO_DIR` (default `~/ai4go_eval/framework`) on first run.

---

## Quick start (one command)

```bash
cd /home/v-murongma/code/code-corr-annotation
bash scripts/newgo_infosel/setup_env.sh                       # once: build envs
bash scripts/newgo_infosel/run_newgo_infosel_pipeline.sh      # runs phases 0–5
```

Results print at the end (infosel vs CE) and land in `outputs/newgo_eval/`.

---

## Step-by-step (what the orchestrator does)

### 0. Build the two conda envs — `setup_env.sh`
Two envs are used **on purpose**:
- `code-attribute` (build + train + eval): the repo training stack (torch + transformers + peft)
  plus the AI4Go metric deps (`fuzzywuzzy python-Levenshtein rouge nltk pandas`).
- `vllm` (labeling only): `pip install vllm` pins torch/transformers, so it is isolated to avoid
  downgrading/breaking the training env.

```bash
bash scripts/newgo_infosel/setup_env.sh
# override env names:  ENV_BUILD=myenv ENV_VLLM=myvllm bash .../setup_env.sh
```
> Use `python3 -m pip` (a stray `~/.local/bin/pip` can shadow the env pip and install to the wrong python).

### 1. Convert raw → compact — `build_newgo_compact.py`
`{prompt, response, task_id}` → compact rows the trainer + labeler consume. The `prompt` already
carries the full FIM context (`<PRE>/<SUF>/<MID>` + `### Response:`), so **no chat template** is
added — training on `prompt → response` is exactly what eval feeds the model.

```bash
conda run -n code-attribute python3 scripts/newgo_infosel/build_newgo_compact.py \
  --in  data/new_go_data/train/cloud_core_test_25.JunJunly_GoOnly_length_filter.jsonl \
  --out data/new_go_data/train_data/newgo_infosel_5k_compact.jsonl \
  --target_model models/Qwen2.5-Coder-7B-Instruct \
  --subsample 5000 --seed 42 --max_len 2048
```
Output row: `{task_id, input_ids, label, prompt_text, comp_text, comp_spans, comp_q}`.
`input_ids = tok(prompt)+tok(response)+EOS`; `label` masks the prompt (IGNORE) and supervises the
response + EOS; the prompt is **left-truncated** if over `max_len` (the response is never dropped).
`comp_spans`/`comp_q` let the teacher align its own tokenization back to the target token positions.

### 2. Register the dataset — `register_dataset.py`
The trainer resolves datasets by KEY from the global registry `configs/datasets.yaml`. This adds
(idempotently) `newgo_infosel_5k_tlabeled` → the labeled file path:

```bash
python3 scripts/newgo_infosel/register_dataset.py
```
Equivalent manual entry (under the top-level `datasets:` in `configs/datasets.yaml`):
```yaml
  newgo_infosel_5k_tlabeled:
    language: go
    role: train
    format: compact
    has_tests: false
    path: data/new_go_data/train_data/newgo_infosel_5k_tlabeled_compact.jsonl
    n: 5000
```

### 3. Teacher-NLL labeling — `label_newgo_nll_vllm.py`
vLLM teacher-forcing: feed `prompt + response`, read `prompt_logprobs`, char-span-align to the
target response tokens → `comp_teacher_nll = [[q, nll], ...]`. Runs in the `vllm` env.

```bash
conda run -n vllm env HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/newgo_infosel/label_newgo_nll_vllm.py \
    --in  data/new_go_data/train_data/newgo_infosel_5k_compact.jsonl \
    --out data/new_go_data/train_data/newgo_infosel_5k_tlabeled_compact.jsonl \
    --teacher models/Qwen3-Coder-30B-A3B-Instruct \
    --tp 1 --max_len 4096 --chunk 64 --gpu_mem_util 0.85 --max_num_batched_tokens 2048
```
Output row (lean, ready for training): `{task_id, input_ids, label, comp_teacher_nll}`.
~9.5 rows/s on one A100; token_select@2.0 drops ~13.5% of tokens.

> **vLLM OOM knobs (important on a single 80GB GPU):** `prompt_logprobs` materialises a
> `[positions × 151k vocab]` transient. Keep `--gpu_mem_util 0.85` (headroom for that transient)
> and `--max_num_batched_tokens 2048` (bounds it). Higher values → `CUDA out of memory` in
> `_get_prompt_logprobs_dict`.

### 4. Train — `src/train/run_train.py` + `configs/`
Two runs, identical recipe (Qwen2.5-Coder-7B LoRA r16/a32, 1 epoch, lr 2e-4, bs1/ga8, max_len 2048):
- **infosel**: `configs/lora_infosel_newgo5k_7b_s42.yaml` — `token_select {enabled: true, threshold: 2.0}`.
- **CE baseline**: `configs/lora_ce_newgo5k_7b_s42.yaml` — `token_select {enabled: false}` (plain CE).

```bash
conda run -n code-attribute env WANDB_DISABLED=true HF_HUB_OFFLINE=1 \
  python3 src/train/run_train.py --config scripts/newgo_infosel/configs/lora_infosel_newgo5k_7b_s42.yaml
```
Checkpoint → `outputs/<name>/checkpoints/checkpoint-<steps>` (625 steps for 5k @ bs1/ga8).

### 5. Evaluate — `eval_newgo.py`
Load base + LoRA adapter, greedy-generate on the 301 test prompts, write `{task_id, label, predict}`,
then score with the **AI4Go metric** (`eval_main`).

```bash
conda run -n code-attribute python3 scripts/newgo_infosel/eval_newgo.py \
  --base_model models/Qwen2.5-Coder-7B-Instruct \
  --adapter outputs/lora_infosel_newgo5k_7b_s42/checkpoints/checkpoint-625 \
  --test data/new_go_data/test/processed_part1.jsonl \
  --out_predictions outputs/newgo_eval/infosel_predictions.jsonl \
  --out_metrics     outputs/newgo_eval/infosel_metrics.json \
  --ai4go_dir <path to extracted AI4GoPlayGround framework>
```
Metrics: `es` (edit-sim, `fuzz.ratio`), `bleu` (BLEU-4, Tokenizer13a), `rougeL` (rouge-L F),
`line_hit_pre/rec`, `hit@0.5`, and **`acc = 0.5·es + 0.2·bleu + 0.3·rougeL`**.

---

## Orchestrator — `run_newgo_infosel_pipeline.sh`

Runs phases 0 (extract RAR) → 1 build → 2 label → 3 train (infosel + CE) → 4 eval → 5 compare.
Every phase is **skip-if-done**; set `FORCE=1` to redo. GPU is used one phase at a time.

Env-var knobs (all optional):

| var | default | meaning |
|-----|---------|---------|
| `TEACHER` | `models/Qwen3-Coder-30B-A3B-Instruct` | labeling teacher |
| `TARGET` | `models/Qwen2.5-Coder-7B-Instruct` | model fine-tuned + evaluated |
| `SUBSAMPLE` | `5000` | rows sampled from the 199,923 train set |
| `SEED` | `42` | subsample + train seed |
| `THRESHOLD` | `2.0` | token_select NLL cutoff |
| `MAX_LEN` | `2048` | target train seq len |
| `TEACHER_MAX_LEN` | `4096` | teacher context |
| `GPU_MEM_UTIL` / `MAX_NBT` | `0.85` / `2048` | vLLM memory knobs (see OOM note) |
| `ENV_BUILD` / `ENV_VLLM` | `code-attribute` / `vllm` | conda envs |
| `AI4GO_DIR` | `~/ai4go_eval/framework` | extracted eval framework |
| `FORCE` | `0` | `1` = ignore skip-if-done |

```bash
# full 200k run on a bigger GPU budget, forced redo:
FORCE=1 SUBSAMPLE=199923 GPU_MEM_UTIL=0.88 bash scripts/newgo_infosel/run_newgo_infosel_pipeline.sh
```

---

## Outputs

```
data/new_go_data/train_data/
    newgo_infosel_5k_compact.jsonl            # Phase 1
    newgo_infosel_5k_tlabeled_compact.jsonl   # Phase 2 (+ comp_teacher_nll)
outputs/
    lora_infosel_newgo5k_7b_s42/checkpoints/  # Phase 3 (adapter)
    lora_ce_newgo5k_7b_s42/checkpoints/
    newgo_eval/
        lora_infosel_newgo5k_7b_s42_predictions.jsonl   # Phase 4 {task_id,label,predict}
        lora_infosel_newgo5k_7b_s42_metrics.json
        lora_ce_newgo5k_7b_s42_{predictions,metrics}...
```

---

## Fine-tuning a 30B-A3B MoE on 8 GPUs

The pipeline supports a much larger **target** (e.g. `Qwen3-Coder-30B-A3B-Instruct`) on a
multi-GPU box (e.g. 8×~70 GB). infosel is a LoRA + label-masking method, so it transfers
directly — only the target model, tokenization, and GPU placement change. Ready configs:
`configs/lora_{infosel,ce}_newgo5k_30bmoe_s42.yaml`.

### Config knobs (in `train:`)
| knob | meaning |
|------|---------|
| `device_map: auto` | shard ONE model across all visible GPUs (naive model-parallel, single process) — **simplest, recommended** |
| `nproc_per_node: 8` + `fsdp: "full_shard auto_wrap"` + `fsdp_wrap: Qwen3MoeDecoderLayer` | launch via `torchrun` + FSDP (data-parallel **and** sharded) — faster, advanced |
| `lora.target_modules: [q_proj, k_proj, v_proj, o_proj]` | **attention-only LoRA** — required for MoE (else adapters attach to every expert) |
| `gradient_checkpointing: true` | needed to fit 30B activations |

`device_map` and `nproc_per_node+fsdp` are **mutually exclusive**. Memory (30B-A3B ≈ 61 GB bf16):
LoRA + `device_map=auto` ≈ **8 GB weights/GPU** + activations (comfortable on 70 GB); FSDP is
similar per-GPU but faster (data-parallel). Full fine-tune needs DeepSpeed ZeRO-3 and is tight — LoRA is preferred.

### Run it (orchestrator, one command)
The data is tokenized with the **target** tokenizer, so it must be **rebuilt + relabeled** with the
30B tokenizer (new key `newgo_infosel_5k_30bmoe_tlabeled`, auto-derived from the TLABELED filename
and auto-registered). Point the orchestrator at the 30B target + configs:

```bash
TARGET=models/Qwen3-Coder-30B-A3B-Instruct \
COMPACT=data/new_go_data/train_data/newgo_infosel_5k_30bmoe_compact.jsonl \
TLABELED=data/new_go_data/train_data/newgo_infosel_5k_30bmoe_tlabeled_compact.jsonl \
INFOSEL_CFG=scripts/newgo_infosel/configs/lora_infosel_newgo5k_30bmoe_s42.yaml \
CE_CFG=scripts/newgo_infosel/configs/lora_ce_newgo5k_30bmoe_s42.yaml \
bash scripts/newgo_infosel/run_newgo_infosel_pipeline.sh
```

Manual per-phase (same steps, explicit):
```bash
# 1) rebuild compact with the 30B tokenizer
conda run -n code-attribute python3 scripts/newgo_infosel/build_newgo_compact.py \
  --in  data/new_go_data/train/cloud_core_test_25.JunJunly_GoOnly_length_filter.jsonl \
  --out data/new_go_data/train_data/newgo_infosel_5k_30bmoe_compact.jsonl \
  --target_model models/Qwen3-Coder-30B-A3B-Instruct --subsample 5000 --seed 42 --max_len 2048
# 2) relabel (teacher can stay 30B-A3B, or use a stronger teacher); --tp 8 for speed on 8 GPUs
conda run -n vllm env HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/newgo_infosel/label_newgo_nll_vllm.py \
    --in  data/new_go_data/train_data/newgo_infosel_5k_30bmoe_compact.jsonl \
    --out data/new_go_data/train_data/newgo_infosel_5k_30bmoe_tlabeled_compact.jsonl \
    --teacher models/Qwen3-Coder-30B-A3B-Instruct --tp 8 --max_len 4096 --gpu_mem_util 0.85
# 3) register + train (device_map=auto shards the 30B across all 8 GPUs)
python3 scripts/newgo_infosel/register_dataset.py \
  --key newgo_infosel_5k_30bmoe_tlabeled \
  --path data/new_go_data/train_data/newgo_infosel_5k_30bmoe_tlabeled_compact.jsonl
conda run -n code-attribute env WANDB_DISABLED=true HF_HUB_OFFLINE=1 \
  python3 src/train/run_train.py --config scripts/newgo_infosel/configs/lora_infosel_newgo5k_30bmoe_s42.yaml
# 4) eval — eval_newgo.py --device_map defaults to auto, so the 30B spreads across GPUs
```

### MoE-specific notes
- **Attention-only LoRA is mandatory**: the dense names `gate_proj/up_proj/down_proj` also live
  inside every MoE expert, so the default 7-module list would attach LoRA to all ~128 experts × layers.
- **`fsdp_wrap`** must be the decoder-layer class — `Qwen3MoeDecoderLayer` for Qwen3 MoE (verify with
  `type(model.model.layers[0]).__name__`).
- **Target == teacher** (both 30B-A3B) ⇒ labeling is *self*-informativeness (the model scores its own
  completions); valid, but a stronger teacher usually gives a cleaner signal.
- **ROCm (AMD MI300)**: transformers MoE forward uses `torch._grouped_mm`, which needs the loop fallback
  on ROCm; on CUDA (A100/H100) native kernels work. `ce_only` allows sdpa/flash attention.
- Under the hood (verified via `--dry_run`): `device_map: auto` → `python … --device_map auto
  --lora_target_modules q_proj,k_proj,v_proj,o_proj`; the FSDP knobs → `torchrun --nproc_per_node=8 …
  --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap Qwen3MoeDecoderLayer`.

---

## Notes & gotchas

- **Scaling / other models**: change `SUBSAMPLE` (and `TARGET`) — the 7B configs + registry key are
  named for `newgo5k` / 7B; for a different size/model copy a config, adjust `model` + `data.train`
  (+ `device_map`/`lora.target_modules` for big/MoE models), and register a new `--key`/`--path`.
  See "Fine-tuning a 30B-A3B MoE on 8 GPUs" above.
- **Two envs are required**: labeling (vLLM) and training/eval must not share one env (vLLM's
  torch/transformers pins conflict with the repo training stack).
- **`n_judged`/empty labels**: the labeler fail-fasts if < 1/5 of rows get any `comp_teacher_nll`.
- **AI4Go metric only**: the framework's own model backends (internal HTTP endpoints) are not
  used — we generate with our own LoRA. The framework's hardcoded credentials were scrubbed
  (endpoints/tokens now read from env vars), so a metric-only import pulls no secrets.
- **The `.rar`** is git-ignored in this repo; the extracted framework is referenced via `AI4GO_DIR`.
