# Data and Model Guide

## 1. Scope

This document explains ordinary data preparation, serialization, model selection, tokenizer coupling, paths, and reproducibility. Method-specific dependency annotation is covered in [Curation](curation.md).

The most important rule is to keep three concerns separate:

- **task meaning:** what code is visible and what span must be completed;
- **model presentation:** how that task is serialized into ChatML/FIM tokens;
- **training storage:** how token IDs and labels are stored for efficient loading.

## 2. Data lifecycle

A robust pipeline should be one-way:

```text
raw source data (read-only)
  -> filtered canonical records
  -> split and deduplicated canonical records
  -> ChatML/FIM rendering
  -> tokenizer-specific compact records
  -> training/evaluation outputs
```

Never treat a generated compact file as the only copy of the task. It cannot be safely moved to another tokenizer without reconstructing the original text.

Recommended external storage follows the environment-variable convention:

```text
EIF_DATA_DIR/
  raw/          original downloads; never edited in place
  interim/      caches, rejected-row logs, partially processed records
  processed/    canonical, ChatML, and compact datasets
EIF_MODEL_DIR/  base models and immutable model artifacts
EIF_RUN_DIR/    long-running logs and checkpoints
HF_HOME/        Hugging Face cache
```

These variables are conventions, not magic. A script must explicitly read them or receive the resulting path as an argument.

## 3. Canonical representation

A canonical record should remain independent of any chat template or tokenizer. The Go single-statement builder uses the following core structure:

```json
{
  "uid": "stable-example-id",
  "source_dataset": "codesearchnet",
  "split": "train",
  "language": "go",
  "task_type": "go_single_statement_completion",
  "prefix": "visible code before the hole",
  "target": "missing statement",
  "suffix": "visible code after the hole",
  "full_code": "prefix + target + suffix",
  "target_kind": "assignment",
  "metadata": {}
}
```

A fundamental invariant is

$$
\texttt{full\_code}
=\texttt{prefix}+\texttt{target}+\texttt{suffix}.
$$

Use a stable ID derived from provenance and source span, not from row order alone. Preserve repository/path metadata for split auditing and leakage checks.

## 4. Candidate extraction and cleaning

The Go pipeline in `scripts/go_singleline_fim_exp/go_single_pipeline.py` focuses on complete, single-statement holes. It:

- locates a function or method body;
- rejects test/generated/noisy code and functions containing comments;
- proposes assignment, nontrivial return, and call-expression statements;
- rejects block headers, braces, very short/long candidates, and low-information statements;
- constructs prefix/target/suffix by character offsets;
- deduplicates normalized code;
- samples target kinds toward configured ratios.

General cleaning helpers in `src/data/code_clean.py` strip C-like comments while respecting string, character, and Go raw-string literals. `normalize_for_match` removes comments, empty lines, and trailing whitespace for fair exact/edit comparison. It should be applied identically to predictions and references.

Cleaning changes the population being studied. Every filter should therefore produce counts and rejection reasons rather than silently dropping rows.

## 5. Splits and leakage control

Split before producing correlated variants. All versions of one base function—different holes, cue variants, canonical renamings, or prompts—must stay in the same split.

At minimum, check:

- repository-level separation when repository metadata exists;
- exact duplicate hashes of `full_code`;
- normalized hashes after whitespace/literal normalization;
- near clones when evaluation credibility depends on it;
- duplicated targets paired with almost identical contexts;
- contamination from benchmark tests or generated solutions.

The controlled experiment design requires repository-first splitting and groups all variants by `base_id`.

## 6. ChatML/FIM representation

This project often presents FIM as an instruction-style chat:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
... prefix [MASK] suffix ...<|im_end|>
<|im_start|>assistant
target<|im_end|>
```

This differs from native sentinel-token FIM such as `<fim_prefix> ... <fim_suffix> ... <fim_middle>`, but the underlying task is the same: prefix and suffix are visible, target is hidden.

The model input contains all turns. Labels are `-100` over system/user/context tokens and equal the token IDs over the assistant response. With the causal-LM shift, logits at position $t-1$ predict the labeled token at position $t$.

The constants in `src/train/dataset.py` must agree with the postprocessing code. A template change after compact data is built invalidates token offsets and annotation indices.

## 7. Compact representation

The training loader accepts JSONL rows in either of two forms.

### 7.1 Preferred compact form

```json
{
  "input_ids": [151644, 8948, 198, 2610],
  "label": [-100, -100, -100, 2610],
  "attention_edges": [
    {"src": 20, "dst": 35, "subtype": "defuse"}
  ]
}
```

The key can be `label` or `labels`. Requirements:

- `input_ids` is nonempty;
- labels have exactly the same length;
- every usable edge satisfies `0 <= src < dst < sequence_length`;
- rows longer than `max_len` are skipped, not partially truncated by this loader.

### 7.2 Legacy annotated form

Rows may instead contain `sft_input`, `qwen_tokens`, and `qwen_annotations`. The loader extracts token IDs and finds the assistant-output boundary by character offsets.

The preferred compact form is easier to validate and faster to load. The legacy form remains useful for inspecting token surfaces and offsets.

## 8. Collation and padding

`AnnotatedSFTDataset` loads records into memory and returns:

- `input_ids: LongTensor[T]`;
- `labels: LongTensor[T]`;
- `annot_pairs: LongTensor[E,2]`;
- optional graph weights.

`DataCollatorForAnnotatedSFT` pads token IDs with the tokenizer's pad ID, pads labels with `-100`, builds a Boolean attention mask, and keeps edge lists ragged. Padding therefore does not contribute to CE.

A consequence is that very large JSONL datasets require enough CPU memory to hold all processed rows.

## 9. Tokenizer coupling

Compact data and graph indices are tokenizer-specific. Postprocessing follows this sequence:

1. maintain character offsets on the reconstructed full code;
2. remap those offsets to the final ChatML sequence;
3. tokenize that exact sequence with the target Qwen tokenizer and request offset mappings;
4. map each simple code token to all overlapping BPE tokens;
5. expand a simple-token edge into the BPE cross product;
6. drop self-edges and deduplicate pairs.

If an identifier becomes multiple subwords, one conceptual edge may become multiple BPE edges. Changing the tokenizer or chat template requires rebuilding compact data.

Despite the function name `get_qwen3_tokenizer`, the hard-coded fallback model is currently `Qwen/Qwen2.5-Coder-1.5B-Instruct`. Treat names and implementation as separate evidence and verify the actual tokenizer artifact used for a run.

## 10. Model configuration

The low-level trainer loads `AutoModelForCausalLM` in bfloat16 and optionally attaches LoRA adapters. Important settings include:

- `model_name_or_path`: immutable base checkpoint;
- `max_len`: dataset and tokenizer sequence limit;
- `use_peft`, LoRA rank, alpha, dropout, and target modules;
- batch size and gradient accumulation;
- learning rate, maximum steps, seed, and output directory;
- attention backend: eager for saliency objectives, SDPA otherwise;
- optional gradient checkpointing, multi-process launch, FSDP, or `device_map`.

A LoRA checkpoint is not a standalone base model unless it has been merged. Evaluation must load the same base model and then the adapter.

The paper reports Qwen2.5-Instruct models at 0.5B/1.5B/7B, LoRA rank 16, alpha 32, dropout 0.05, one epoch, learning rate $2\times10^{-4}$, effective batch size 8, maximum length 2048, and bf16. Current YAML files may use different values; report the executed configuration, not the paper defaults by assumption.

## 11. YAML configuration behavior

A YAML file does nothing by itself. A Python entry point must load it and map keys to arguments.

Two configuration styles coexist:

- `configs/train/default.yaml` is a flat example and is not consumed by `src/train/run_train.py` as written;
- `run_train.py` expects an experiment YAML with top-level `name`, `model`, `data`, `train`, and optional `eval`.

Likewise, `configs/annotate/default.yaml` contains useful values but the legacy `src/annotate/main.py` is mostly hard-coded and does not load it.

Environment placeholders such as `${ANNOTATE_MODEL}` are not expanded by `yaml.safe_load`. Expansion requires explicit application code or a tool such as Hydra/OmegaConf. Do not assume the displayed value becomes an environment variable automatically.

## 12. Current path and registry caveats

`src/data/registry.py` intends `configs/datasets.yaml` to be the single dataset source of truth and resolves relative paths from the repository root. That registry file is missing in this checkout, so `get_dataset`, `run_train.py`, and `run_eval.py` cannot currently resolve named datasets.

Several checked-in defaults contain machine-specific absolute paths. They are examples from one workspace, not portable project contracts. Prefer:

1. a registry with repository-relative metadata and environment-root expansion;
2. CLI overrides for a particular run;
3. environment variables for machine-level roots and secrets.

## 13. Validation checklist

Before training, validate a small deterministic sample:

- canonical reconstruction equality;
- parseability before and after hole insertion;
- split and duplicate audits;
- ChatML role boundaries;
- equal lengths for IDs and labels;
- at least one supervised token;
- decoded supervised span equals the target;
- edge bounds and source-before-target direction;
- tokenizer/model identity;
- sample count, length distribution, and target-kind distribution;
- one collated batch and one forward pass.

Save the executed command, configuration, seed, data fingerprint, model revision, package versions, GPU type, and output path with every run.
