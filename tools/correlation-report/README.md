# Attribution Analysis Visualizer

React/Vite frontend for inspecting **token-level ALTI saliency** and **training correlation** reports produced by this repo’s all-tokens pipeline (`src/intervention_experiment.py`).

## What the current pipeline does

For each semantic **test output token** \(t\):

1. **ALTI** → top source tokens (UI: Top Correlations, up to 4).
2. User (or offline precompute) focuses on **one** edge \((s \rightarrow t)\).
3. **Train retrieval (viz-aligned)**  
   \(L_{\text{probe}} = -\log(C[t,s]+\varepsilon)\),  
   \(g_{\text{probe}} = \nabla_{\theta_{\text{LoRA}}} L_{\text{probe}}\),  
   score trains by \(\cos(g_{\text{probe}}, g_{\text{train}})\) → **Top-10** samples.  
   - `ce_only` bank: CE on LoRA  
   - `ce_saliency` bank: CE + λ · contrastive saliency on LoRA (needs `attention_edges`)
4. **Pair matching (Stage 3)** on those trains: scan answer tokens (default 3 targets × 3 sources), match with \(\cos(\nabla C_{\text{test}}, \nabla C_{\text{train}})\) on the **same LoRA** space.

Report filenames:

```text
correlation_matching_results_{model_tag}_{task_id}_all_tokens.json
```

Example: `correlation_matching_results_ce_saliency_codesearchnet_go_test_a443fde88b8c161d_all_tokens.json`  
(Do **not** load `*_prescreen.json` into the viewer.)

---

## Run the visualizer

```bash
cd tools/correlation-report
pnpm install
pnpm dev
```

Open `http://localhost:5173`.

Static build:

```bash
cd tools/correlation-report
pnpm run build
cd dist
python3 -m http.server 5173 --bind 0.0.0.0
```

The app can load bundled files under `dist/data/` and also **import JSON in the browser** (nothing is uploaded to a server).

---

## How to use New View

### Single model

1. Import one `*_all_tokens.json`.
2. Click an **output token** in Model Output (orange = target).
3. Left: **Top Correlations** (`source → target` + saliency).
4. **Click one** Top Correlation edge.
5. Right: **Top-10 related trains** for that edge, with matched train pairs and `cos_sim`.

Until you select a Top Correlation, the right panel stays empty (by design).

### Dual-model compare (e.g. `ce_only` vs `ce_saliency`)

1. Load left JSON and right JSON (two imports).
2. Panels **scroll independently**; token linking is **off** by default (enable Link only if you want synced clicks).
3. Same click flow on each side.

### Reading scores

| UI label | Meaning |
|----------|---------|
| saliency | ALTI \(C[t,s]\) on the test side |
| probe / `coarse` on a train group | Retrieval score \(\cos(g_{\text{probe}}, g_{\text{train}})\) (stored as `coarse_cos_sim`) |
| pair `cos_sim` | \(\cos(\nabla C_{\text{test}}, \nabla C_{\text{train}})\) for one matched edge pair |
| group `best` | Max pair `cos_sim` among currently filtered pairs for that train |

---

## Generating reports (backend)

Prefer **LoRA adapters** (not merged full weights) for attribution accuracy:

```bash
# From repo root
bash run_batch_experiments.sh \
  --model-path  /path/to/code-corr-annotation/outputs/go_single/models/ce_saliency \
  --base-model-path /path/to/code-corr-annotation/models/Qwen2.5-Coder-7B-Instruct \
  --train-data  /path/to/csn10k_train_chat.jsonl \
  --test-data   /path/to/csn500_test_chat.jsonl \
  --indices 106
```

Use `.../models/ce_only` the same way for the CE-only adapter.

Notes:

- Detects `adapter_config.json` → loads base + Peft; grads on all `lora_*` params.
- First run builds `.cache/saliency_train_bank/` (slow); later runs reuse it. Legacy `.cache/prescreen_sketch/` is unused.
- For `ce_saliency`, train JSONL must keep **`attention_edges`** (re-convert with updated `tools/compact_to_chat_jsonl.py` if needed).
- Sync this repo to the machine that runs experiments; filenames include `model_tag` from the adapter folder name.

---

## Importing reports in the UI

- Drag a JSON file, or choose a file.
- Paste a JSON URL → `Load URL`.
- Or open with `?reportUrl=...` (remote host must allow CORS).

### 1. Native all-token correlation report (primary)

```text
correlation_matching_results_{model_tag}_{task_id}_all_tokens.json
```

Useful `experiment_meta` fields:

- `model_name` / `model_path`
- `screening`: `saliency_probe_bank`
- `config.GRAD_SPACE`: `lora` (preferred) or `fine_attn`
- `config.BANK_LOSS_MODE`: `ce_only` | `ce_saliency`
- `config.PROBE`: `L_probe=-log(C+eps)`

Minimal shape:

```json
{
  "experiment_meta": {
    "test_sample_index": 106,
    "task_id": "codesearchnet_go_test_…",
    "model_name": "ce_saliency",
    "mode": "all_tokens",
    "screening": "saliency_probe_bank",
    "tokens_analyzed": 30
  },
  "test_sample_baseline": {
    "full_tokens": ["…"],
    "correct_full_tokens": ["…"],
    "prompt_len": 100
  },
  "per_token_results": [
    {
      "target_token_index": 108,
      "target_token": " err",
      "top_correlations": [
        {
          "source_token": "func",
          "source_token_index": 42,
          "target_token": " err",
          "target_token_index": 108,
          "saliency_score": 0.11
        }
      ],
      "correlation_pairs": []
    }
  ],
  "train_sample_details": {}
}
```

If `correlation_pairs` / `train_sample_details` are present, the training panel is enabled. Empty pairs still show saliency + top sources.

### 2. Generic saliency-only format

When an external method only has one saliency vector per target:

```json
{
  "sample_id": "case-001",
  "tokens": ["def", "Ġfoo", "(", ")", ":", "Ċ", "Ġ", "Ġreturn", "Ġbar"],
  "prompt_len": 6,
  "correct_tokens": ["def", "Ġfoo", "(", ")", ":", "Ċ", "Ġ", "Ġreturn", "Ġbaz"],
  "saliency_list": [
    {
      "target_token_index": 7,
      "scores": [0.12, 0.04, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0]
    }
  ]
}
```

Aliases: `full_tokens` / `token_list`; `start_index` / `answer_start_index`; `saliency` / `targets`; `index` / `target_index`; object map `saliency_by_target`.

### 3. Legacy `latest_saliency.json`

Still accepted; importer maps it to the generic saliency-only view.

---

## Stage 3 train-side candidates (for reading the UI)

On each retrieved train sample (defaults):

- Up to **3** non-trivial answer **targets** \(t'\) (scanned from the start of the answer).
- For each \(t'\), ALTI top-**3** non-trivial **sources** \(s'\).
- → up to **9** candidate train edges, each matched to the **selected** test edge.

“Trivial” ≈ chat template / whitespace / lone punctuation — skipped as sources and targets.

---

## Related tools

| Tool | Role |
|------|------|
| `run_batch_experiments.sh` | Batch all-tokens runs + model-tagged outputs |
| `tools/compact_to_chat_jsonl.py` | Compact graphsignal → chat JSONL (**keeps `attention_edges`**) |
| `tools/merge_lora_adapter.py` | Optional merge for inference-only; **prefer adapters for attribution** |
| `.cache/saliency_train_bank/` | LoRA train gradient sketches (per model / bank loss) |
