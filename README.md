# code-corr-annotation

Training and evaluation for **annotation-guided code completion** on Qwen2.5-Coder.
Task: Go single-line completion (a statement is masked out; the model fills it in).
The training data carries **annotation edges** (`attention_edges`) that mark which
source tokens the target genuinely depends on — used by the saliency / cfmask
objectives to reduce reliance on spurious shortcuts.

Everything is **config-driven**: one YAML per experiment, two entry points
(`run_train.py`, `run_eval.py`). Adding an experiment = adding a YAML, not a shell
script.

> The previous, long-form research log lives in [`README.legacy.md`](README.legacy.md).

---

## 1. Setup

Single A100 80GB, conda env `code-attribute`. Run this prelude before any command:

```bash
source /home/v-murongma/miniconda3/etc/profile.d/conda.sh
conda activate code-attribute
cd /home/v-murongma/code/code-corr-annotation
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_DISABLED=true
```

Validate the dataset registry (prints every dataset + checks all paths exist):

```bash
python src/data/registry.py
```

---

## 2. Quickstart

```bash
# Train an experiment (reads the YAML, writes outputs/<name>/checkpoints + train.log)
python src/train/run_train.py --config configs/experiments/cfmask_A_rate30_10k.yaml

# Evaluate that experiment's trained checkpoint on its configured datasets
python src/eval/run_eval.py --config configs/experiments/cfmask_A_rate30_10k.yaml

# Read the result table
cat outputs/cfmask_A_rate30_10k/eval/summary.json
```

Long runs: detach so they survive the terminal closing (plain `nohup` has been
unreliable here):

```bash
setsid bash -c 'source /home/v-murongma/miniconda3/etc/profile.d/conda.sh; conda activate code-attribute; \
  cd /home/v-murongma/code/code-corr-annotation; \
  export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_DISABLED=true; \
  python src/train/run_train.py --config configs/experiments/<exp>.yaml' \
  > outputs/<exp>_train.out 2>&1 < /dev/null &
```

---

## 3. Data formats (READ THIS)

Two physical formats coexist **by purpose — do not mix them**. The registry
(`configs/datasets.yaml`) records which format each dataset uses.

### 3.1 `compact` — TRAINING & teacher-forcing diagnostics

Pre-tokenized, one JSON object per line. Fields:

| field | type | meaning |
|---|---|---|
| `input_ids` | `list[int]` | full tokenized sequence (chat template already applied) |
| `label` | `list[int]` | same length as `input_ids`; `-100` = masked (prompt), else = target token to learn |
| `attention_edges` | `list[{src,dst,subtype}]` | annotation edges; `src`/`dst` index into `input_ids`; `subtype` ∈ dataflow, defuse, call, semantic, api, type, bracket, return |
| `length` | `int` | `len(input_ids)` |
| `uid` | `str` | stable example id |
| `language` | `str` | e.g. `go` |
| `annotations` | `list` | human-readable annotation records (optional) |

Used by: **all training** (`loss_mode` = ce_only / ce_shortcut_mask / ce_saliency)
and teacher-forcing diagnostics (saliency mAP, counterfactual robustness). The
train-time valid set is also compact (for next-token-prediction valid loss).

### 3.2 `chatml` — GENERATION eval (pass@k + CodeBLEU)

Raw text with clean prefix/suffix, one JSON object per line. Fields:

| field | type | meaning |
|---|---|---|
| `messages` | `list[{role,content}]` | system / user / assistant chat turns; the prompt is everything up to the assistant turn |
| `prefix`, `target`, `suffix` | `str` | the code before the mask, the gold completion, the code after |
| `judge_payload` | `dict` | execution-judge inputs (`test`, `entry_point`, `judge_prefix/suffix`, …); only meaningful when `has_tests=true` |
| `source_dataset` | `str` | judge-routing key (`codesearchnet_go`, `mceval_go`, `humaneval_x_go`) — must match the dispatcher in `scripts/benchmark/eval_judges.py` |
| `uid`, `language` | `str` | id / language |

Used by: **all generation eval**. Clean prefix/suffix avoids the compact
trailing-token bug that collapses CodeBLEU.

### 3.3 Which role uses which format

| role | format | purpose |
|---|---|---|
| `train` | **compact** | teacher-forcing training (required) |
| `valid` | **compact** (+ optional `chatml_path`) | train-time valid loss; optional CodeBLEU |
| `test` | **chatml** (+ optional `compact_path`) | generation eval; compact only for teacher-forcing diagnostics |

`run_train.py` **rejects** a non-compact train set. `run_eval.py` always reads the
chatml representation for generation.

### 3.4 Dataset registry — `configs/datasets.yaml`

Single source of truth for paths. Never hardcode paths elsewhere — add an entry:

```yaml
datasets:
  my_test:
    language: go
    role: test                 # train | valid | test
    format: chatml             # compact | chatml
    has_tests: true            # true -> pass@1/pass@5/CodeBLEU; false -> CodeBLEU only
    source_dataset: mceval_go  # judge-routing key (executable sets only)
    path: data/.../my_test_chatml.jsonl
    compact_path: data/.../my_test_compact.json   # optional, for diagnostics
    n: 123
```

Currently registered: `csn_train_5k`, `csn_train_10k` (train); `csn_valid` (valid);
`csn_test` (CodeBLEU only, N=7979), `mceval` (executable, N=45), `humaneval_x`
(executable, N=129).

---

## 4. Training — `src/train/run_train.py`

```bash
python src/train/run_train.py --config configs/experiments/<exp>.yaml [--dry_run]
```

`--dry_run` prints the fully-resolved `train.py` command without running it.
Output: `outputs/<name>/checkpoints/` (+ `outputs/<name>/train/train.log`).
`max_steps` is computed from `epochs × ceil(n / (batch_size × grad_accum))`.

### Experiment YAML schema

```yaml
name: cfmask_A_rate30_10k          # -> outputs/cfmask_A_rate30_10k/
model: models/Qwen2.5-Coder-7B-Instruct   # optional; default in datasets.yaml

data:
  train: csn_train_10k             # registry name (MUST be compact)
  valid: csn_valid                 # registry name (optional)

train:
  loss_mode: ce_shortcut_mask      # ce_only | ce_shortcut_mask | ce_saliency | saliency_only
  epochs: 3
  lr: 2.0e-4
  batch_size: 1
  grad_accum: 8
  max_len: 8192
  gradient_checkpointing: true     # needed for 14B/32B; optional for 7B
  lora: {r: 32, alpha: 64, dropout: 0.05}   # omit / set `full_finetune: true` for full FT
  save_total_limit: 2
  eval_steps: 625                  # train-time valid-loss cadence
  eval_max_samples: 500
  eval_codebleu_samples: 64        # train-time greedy CodeBLEU on valid (optional)

  # loss-specific block (only the one matching loss_mode is read):
  cfmask:                          # ce_shortcut_mask (+ ce_shortcut_mask_saliency)
    rate: 0.3                      # fraction of shortcut-candidate tokens masked
    recency_window: 8              # protect the N tokens before each target
    protect_prefix: 16             # protect the first N prompt tokens
    exclude_special: true          # protect special / role tokens + BOS sink
    per_target: true               # per-target leaky mask (4-D); false = one global key mask
    weight_aware: false            # true = mask prob p_max*(1-w)^gamma by augmented node weight
    invariance_beta: 0.0           # 0 = Variant A (1 forward); >0 = Variant B (CE + beta*KL, 2 forwards)
  # saliency:                      # ce_saliency / saliency_only / ce_shortcut_mask_saliency
  #   loss_type: contrastive       # contrastive | ranknet | margin_bce | softmax
  #   lambda: 1.5                   # weight of the saliency term (combined mode: 0.25 keeps pass-rate)
  #   alpha: 1.0
  #   margin_plus: 2.0
  #   neg_sample_k: 64
  #   layer: -1
  #   detail_log_steps: 50         # log recall@k / mAP@k of annotation edges every N steps (§9.3)

eval:                              # consumed by run_eval.py (see §5)
  datasets: [csn_test, mceval, humaneval_x]
  num_samples: 5
  codebleu: true
```

### Loss modes

| `loss_mode` | what it optimizes | needs eager attn? |
|---|---|---|
| `ce_only` | standard next-token CE | no (SDPA) |
| `ce_shortcut_mask` | CE on input with shortcut tokens masked out (**cfmask**) | no (SDPA) |
| `ce_saliency` | CE + saliency alignment loss (reads attention probs) | **yes (eager)** |
| `ce_shortcut_mask_saliency` | **cfmask masked-CE + λ·contrastive saliency** — pass-rate *and* alignment (§9.2) | **yes (eager)** |
| `saliency_only` | saliency loss alone | **yes (eager)** |

`attn_implementation` is selected automatically by `loss_mode`: saliency modes
load with `eager` (needed to expose attention weights), everything else uses
memory-efficient `sdpa` (essential for 14B/32B).

---

## 5. Evaluation — `src/eval/run_eval.py`

Two modes. Both write a uniform `outputs/<name>/eval/summary.json` (one row per
dataset) plus per-dataset detail under `outputs/<name>/eval/`.

```bash
# Config mode: evaluate the experiment's TRAINED checkpoint on its eval.datasets
python src/eval/run_eval.py --config configs/experiments/<exp>.yaml

# Ad-hoc mode: any checkpoint (or 'base') on any registry datasets
python src/eval/run_eval.py --model outputs/<exp>/checkpoints --name <tag> \
    --datasets csn_test,mceval,humaneval_x [--num_samples 5]
```

Useful flags:

| flag | effect |
|---|---|
| `--datasets a,b` | override the dataset list (e.g. `mceval,humaneval_x` to skip the slow `csn_test`) |
| `--num_samples K` | pass@K sampling count (default 5) |
| `--no_codebleu` | skip CodeBLEU |
| `--max_rows N` | cap rows per dataset (debug) |
| `--model base` | evaluate the registry default base model |

Config mode always targets `outputs/<name>/checkpoints` — never the training base.
Re-running with a subset of datasets **merges** into the existing summary (a
previously-computed `csn_test` is preserved).

### Metrics & per-dataset policy

Decided by the dataset's `has_tests` flag in the registry:

| `has_tests` | reported metrics | how |
|---|---|---|
| `true` (mceval, humaneval_x) | **pass@1**, **pass@K**, **CodeBLEU** | greedy → pass@1; K samples (temp 0.2, top-p 0.95) → any-of-K pass@K; greedy vs gold → CodeBLEU |
| `false` (csn_test) | **CodeBLEU only** | greedy vs gold (no executable tests) |

- **pass@1** — greedy completion stitched into `prefix + pred + suffix`, executed against `judge_payload`.
- **pass@K** — K sampled completions; passes if **any** of the K passes. `summary.json` reports it under the key `pass@<num_samples>`.
- **CodeBLEU** — token + AST + dataflow similarity of the greedy completion vs the gold `target`. Note: on short single-line targets the dataflow sub-score often degenerates to 0 (a benign library warning — the term is substituted, not penalized); the score is dominated by n-gram + AST and is comparable across models on the same set.

Generation uses each dataset's **chatml** representation via
`scripts/benchmark/benchmark_eval.py`. The execution judge routes on
`source_dataset` (see `scripts/benchmark/eval_judges.py`).

### `summary.json` shape

```json
{
  "name": "cfmask_A_rate30_10k",
  "model": ".../checkpoints",
  "num_samples": 5,
  "results": [
    {"dataset": "mceval", "has_tests": true, "n_total": 45,
     "pass@1": 0.80, "pass@5": 0.84, "codebleu": 0.72},
    {"dataset": "humaneval_x", "has_tests": true, "n_total": 129,
     "pass@1": 0.85, "pass@5": 0.86, "codebleu": 0.74},
    {"dataset": "csn_test", "has_tests": false, "n_total": 7979,
     "pass@1": null, "pass@5": null, "codebleu": 0.60}
  ]
}
```

---

## 6. Outputs layout

```
outputs/<name>/
  checkpoints/        # trainer output (LoRA adapter or full model) + intermediate checkpoint-*/
  train/train.log     # training log
  eval/
    summary.json      # one row per dataset (the table you read)
    <dataset>.json    # per-dataset cleaned detail
    raw/<dataset>/    # raw benchmark_eval outputs + run.log
```

---

## 7. Repository layout

```
configs/
  datasets.yaml                 # dataset registry (paths, format, has_tests)
  experiments/*.yaml            # one YAML per experiment
src/
  data/registry.py              # DatasetSpec + get_dataset() + self-check
  data/edge_augment.py          # transitive-closure annotation augmentation (§9.1)
  common.py                     # load_experiment(), ExpPaths, compute_max_steps()
  train/run_train.py            # training launcher (YAML -> train.py argv)
  train/train.py                # trainer (loss modes, cfmask, saliency)
  train/loss.py                 # build_shortcut_mask, saliency losses
  eval/run_eval.py              # unified evaluation entry point
  eval/diagnostics/             # cf_robustness, honest-mAP (teacher-forcing)
scripts/benchmark/
  benchmark_eval.py             # generation + judge + CodeBLEU
  eval_judges.py                # per-language execution judge + source_dataset routing
  recall_diag.py                # offline saliency recall@k for any checkpoint (§9.3)
scripts/data_process/           # single-line FIM eval builders (HumanEval-X, MCEval), CSN train builder
tools/visual_annotation/        # annotation viewer (build_annotation_showcase_viewer.py)
viz/                            # interactive saliency explorer (serve.sh + precompute.py)
outputs/docs/                   # reports
```

---

## 8. Diagnostics & visualization (optional)

```bash
# Annotation viewer: render annotation edges + model predictions for chosen uids
python tools/visual_annotation/build_annotation_showcase_viewer.py \
  --raw_data_path data/.../codesearchnet_go_test_full_chatml.jsonl \
  --edge_data_path data/.../codesearchnet_go_test_full_srcannotate_compact.json \
  --model_path models/Qwen2.5-Coder-7B-Instruct --languages go \
  --uids <uid1>,<uid2> --predictions_json <preds.json> --output_html out.html

# Saliency explorer (precompute once with GPU, then serve statically)
python viz/precompute.py --models cfmask_A_rate30_10k ce_only_10k --dataset csn10k --uids <uids> --layer -1
bash viz/serve.sh 8011    # open the forwarded port in your browser
```

---

## 9. Annotation augmentation, combined loss & saliency alignment

The most recent line of work: (a) **densify** the sparse annotation graph, (b) a
**combined objective** that raises pass-rate *and* aligns attention saliency to the
annotation, (c) a **saliency-alignment metric**, and (d) a subtle but important
**eval correctness fix**.

### 9.1 Annotation augmentation — `data.edge_augment`

The raw graph signal is single-hop. `augment_edges()`
([`src/data/edge_augment.py`](src/data/edge_augment.py)) takes the **transitive
closure** of the position-ordered DAG so descendants inherit their ancestors'
annotation, with a per-hop `decay` weight:

```yaml
data:
  edge_augment:
    enabled: true
    mode: undirected   # directed = ancestor->descendant; undirected = every pair in a connected component
    decay: 1.0         # inherited-edge weight = decay^(hops-1); 1.0 = full inheritance
    max_hops: 3        # 0 = unlimited; k = keep nodes within <=k hops
```

The **same augmented `annot_pairs`** feed *both* the cfmask (which tokens to
protect) and the contrastive positives. Density on the python data (vs direct
edges): directed ∞ ≈ ×1.66 (cap barely matters), but **undirected** cap-2 ×4.4,
cap-3 ×7.2, cap-4 ×13.3, ∞ ×28.9 — so under `undirected` the hop cap is a strong
lever and ∞ is near-fully-connected. `undirected, max_hops: 3` is recommended.

### 9.2 Combined objective — `loss_mode: ce_shortcut_mask_saliency`

cfmask alone raises pass-rate but does **not** align saliency; the contrastive
saliency aligns but does **not** raise pass-rate. The combined mode does both:

```
total = cfmask_masked_CE  +  saliency.lambda · contrastive_saliency
```

The masked-CE drives pass-rate (recover the target from annotation tokens); the
contrastive saliency is computed on a **clean** forward (the masked forward would
trivially satisfy it) and pulls attention onto the augmented annotation. Needs
eager attention (two forwards). `saliency.lambda` is the key knob: **0.25** keeps
pass-rate reasonable; 1.0 collapses it.

### 9.3 Saliency-alignment metric — recall@k

Set `saliency.detail_log_steps: N` to log **recall@k / mAP@k** of annotation edges
among each target's top-k most-salient sources, to `train.log` and
`train/saliency_detail.jsonl`. To score any checkpoint offline:

```bash
python scripts/benchmark/recall_diag.py     # edit the CKPTS list inside
```

### 9.4 Eval correctness fix — de-indent the prediction

The single-line FIM builders move the masked line's **leading indentation into the
prefix** (the target is de-indented). A model trained on indentation-bearing
targets re-emits it, double-indenting the reconstruction → `IndentationError` in
Python (cosmetic for brace languages). The judge
([`eval_judges.py`](scripts/benchmark/eval_judges.py)) now `lstrip()`s the
prediction before reconstruction — a no-op for correct predictions, never turning a
passing case into a failure. This removed a large **python-only** "regression" that
was purely a train/eval format mismatch.

### 9.5 Headline findings (1.5B LoRA, python single-line FIM, 1 epoch, seed 42, patched eval)

| config | pass@1 (HE-X / MCEval) | saliency recall@10 |
|---|---|---|
| base (untrained) | 0.227 / 0.300 | 0.22 |
| ce_only | 0.405 / 0.511 | 0.27 |
| cfmask + undirected-h3 aug | **0.509 / 0.578** | 0.26 |
| &nbsp;&nbsp;+ contrastive (λ=1.0) | 0.307 / 0.433 | — |
| &nbsp;&nbsp;+ contrastive (λ=0.25) | 0.429 / 0.500 | **0.61** |

- **Pass-rate** is driven by cfmask + undirected augmentation; the hop cap (3/4/5)
  is within seed noise.
- **Alignment** comes *only* from the contrastive: every non-contrastive model sits
  at recall@10 ≈ 0.22–0.27 (untrained level); the contrastive lifts it to ~0.61.
- The combined loss buys alignment at a ~0.08 pass-rate cost; loss / grad-norm
  curves show this is a genuine generation effect, **not** under-convergence.
- Across 3 seeds the undirected-aug pass-rate gain is real but modest once seed
  variance (±0.04–0.06) is accounted for.
