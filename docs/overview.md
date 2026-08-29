# Project Overview

## 1. Research question

This repository studies a failure mode of code-model fine-tuning: a model can predict the right token for the wrong reason. When fine-tuned on a relatively small or project-specific corpus, it may learn local token co-occurrences, formatting habits, comments, naming conventions, or repository-specific API patterns instead of transferable program dependencies.

The paper calls the proposed framework **CausalCoder**. Its central idea is to augment ordinary next-token supervision with token-level evidence about program dependencies:

- identify which visible source tokens can legitimately support a target token;
- avoid supervising targets whose required evidence is absent;
- encourage the model's internal saliency to favor supported sources over unsupported context;
- test both completion quality and whether the model still relies on spurious cues.

This is a data-curation project in a broad sense, but the main method does not merely delete samples. It augments labels *inside* each retained sample.

## 2. The complete research logic

The project can be read as five connected questions:

1. **Does the problem exist?** Use data attribution and feature attribution to find first-error tokens whose salient context is structurally unsupported.
2. **What should the model have used?** Build a conservative token-dependency graph from program analysis plus a code-oriented annotation model.
3. **How can that knowledge affect training?** Convert graph edges and target eligibility into auxiliary supervision alongside next-token prediction.
4. **Did internal behavior change?** Measure saliency alignment and apply interventions to supported and unsupported tokens.
5. **Did useful behavior improve?** Compare pass@k, CodeBLEU, exact/edit similarity, and robustness against baselines and ablations.

The intended end-to-end flow is:

```text
raw code
  -> canonical FIM example: prefix + target + suffix
  -> model-facing ChatML/FIM example
  -> neuro-symbolic token dependency annotation
  -> tokenizer-aligned compact training record
  -> LoRA/full fine-tuning with CE and optional graph-guided objectives
  -> generation and execution-based evaluation
  -> attribution, saliency-alignment, and counterfactual diagnostics
```

## 3. Three layers that should not be confused

### 3.1 Research claim

The paper describes the final scientific story: spurious token correlations are diagnosed with gradient-based data attribution and ALTI feature attribution; CausalCoder then uses selective NTP and contrastive last-layer saliency supervision.

### 3.2 Current implementation

The source tree is a broader research workbench. In addition to the paper's main route, it contains:

- several saliency losses;
- counterfactual shortcut masking;
- graph-edge prediction;
- graph-conditioned attention bias;
- transitive edge augmentation;
- teacher-NLL token selection;
- attribution retrieval and intervention diagnostics.

These are alternatives, ablations, or intermediate ideas. Their presence does not mean every one belongs to the final paper method.

### 3.3 Experiment record

The manuscript reports finished empirical results. A result in the PDF is not automatically reproduced by the current checkout. Reproduction requires matching the exact data revision, model checkpoint, tokenizer, configuration, seed, and evaluation harness.

## 4. Main code areas

| Area | Primary responsibility |
| --- | --- |
| `src/annotate/` | Build symbolic and neural token-dependency edges; remap source spans to model tokens. |
| `src/attribution/` | Compute per-token ALTI saliency, training-sample influence, and train/test correlation matching. |
| `src/data/` | Resolve dataset specifications, normalize code, and augment dependency edges. |
| `src/train/` | Load annotated compact data, compute CE/graph-guided objectives, and train Qwen models with Hugging Face/PEFT. |
| `src/eval/` | Launch generation evaluation and run saliency/counterfactual diagnostics. |
| `scripts/go_singleline_fim_exp/` | Build and evaluate the concrete Go single-statement FIM task used during development. |
| `configs/` | Store declarative defaults for data, annotation, baselines, and training. |

Read the focused documents next:

- [Data and Model Guide](data_model.md)
- [Attribution Guide](attribution.md)
- [Curation Guide](curation.md)
- [Experiment Guide](experiments.md)
- [Controlled Spurious-Correlation Design](FIM_spurious_correlation_experiment_design.md)

## 5. Core data objects

The repository uses three conceptual representations:

1. **Canonical data** preserves task meaning with fields such as `prefix`, `target`, `suffix`, `full_code`, language, split, and provenance.
2. **ChatML/FIM data** renders the canonical task into a prompt and assistant completion for generation or SFT.
3. **Compact annotated data** stores `input_ids`, `label`/`labels`, and `attention_edges` for efficient training.

The canonical record is the source of truth. ChatML is a presentation layer. Compact data is model- and tokenizer-dependent.

## 6. What the graph means

An annotation edge is conceptually

$$
e=(s,t,r),
$$

where source position $s$ provides evidence for target position $t$, and $r$ is a relation type such as def-use, call-argument, return-value, type binding, data flow, or API semantics.

The code normalizes usable training edges into causal token order $s<t$. A direct edge has weight 1. Optional graph augmentation can add reachable multi-hop pairs with a decayed weight. The training dataset exposes these as `annot_pairs` and, when enabled, `annot_weights` or per-node weights.

## 7. Training logic

Ordinary causal-language-model training minimizes next-token cross entropy only on the assistant completion. Context labels are `-100` and therefore ignored.

The paper's conceptual objective is

$$
\mathcal L
=\mathcal L_{\mathrm{selective\text{-}NTP}}
+\lambda\mathcal L_{\mathrm{saliency}}.
$$

Selection decides **which target tokens are safe to learn**. Saliency supervision decides **which evidence should support an eligible target**.

The current trainer generalizes this into several `loss_mode` choices:

| Mode | Meaning |
| --- | --- |
| `ce_only` | Standard next-token cross entropy. |
| `ce_saliency` | CE plus a chosen saliency-alignment loss. |
| `saliency_only` | Isolate the saliency objective for debugging. |
| `ce_shortcut_mask` | Hide unsupported context and train for robust prediction. |
| `ce_shortcut_mask_saliency` | Combine masking and saliency alignment. |
| `ce_edge_pred` | Add an auxiliary graph-edge prediction head. |
| `ce_attn_bias` | Add a learned positive attention-logit bias on graph edges during training. |

Only saliency modes require eager attention probabilities; other modes can use memory-efficient SDPA.

## 8. Evaluation logic

No single metric proves the full claim.

- **Functional quality:** pass@1/pass@k with executable tests.
- **Surface quality:** CodeBLEU, normalized exact match, or edit similarity when tests are unavailable.
- **Internal alignment:** annotation-edge recall, precision, and mAP among salient sources.
- **Spurious reliance:** SCR@10 at the first generation error.
- **Causal robustness:** prediction degradation when unsupported context is masked versus when supported evidence is changed.
- **Statistical reliability:** paired examples, multiple seeds, standard deviation/confidence intervals, and failure counts.

Teacher-forced saliency alignment is a mechanism diagnostic, not a substitute for free-generation correctness.

## 9. Important implementation boundaries in this checkout

Several high-level entry points reference files that are not present in the current working tree:

- `src/train/run_train.py` and `src/data/registry.py` expect `configs/datasets.yaml`;
- `src/eval/run_eval.py` expects `scripts/benchmark/benchmark_eval.py`;
- `src/attribution/NIF.py` imports `src.sft.inference`;
- `scripts/data_process/build_fim_annotation_data.py` imports `src.data_process.pipeline`;
- baseline scripts import source modules under `src/baseline/`, but only bytecode caches are present.

Therefore, these paths document intended architecture but are not all directly runnable in this checkout. The lower-level trainer and the Go-specific data-building code are substantially more self-contained. Do not “repair” these gaps by guessing paths; recover the matching branch or artifact first.

## 10. Recommended learning order

1. Read `src/train/dataset.py` to understand one batch.
2. Read `src/annotate/utils.py` and `postprocessing.py` to see how source spans become BPE indices.
3. Read `src/annotate/neural_annot.py` for graph construction.
4. Read `src/train/loss.py`, then `src/train/train.py`, to see how graph supervision becomes gradients.
5. Read `src/attribution/saliency.py` for diagnostic ALTI rollout.
6. Read `src/attribution/intervention_experiment.py` for the expensive retrieval/matching pipeline.
7. Read `src/eval/diagnostics/` and the experiment document to connect internal metrics with behavior.

For any costly run, first validate one sample or one batch, record peak memory and throughput, and estimate the full cost before scaling.
