# Attribution Guide

## 1. What “attribution” means here

The repository uses two different kinds of attribution.

- **Data attribution:** which training examples most influence a test prediction?
- **Feature attribution:** which earlier tokens most influence a particular target token?

The paper combines them to diagnose spurious correlations. Feature attribution identifies a suspicious test-time source-to-target relation; data attribution then retrieves training examples that may have taught a similar behavior.

Attribution is evidence about model behavior, not proof of causality. The intervention code exists because a convincing claim also requires changing the alleged cause and measuring the prediction response.

## 2. Causal-LM indexing

If token $x_t$ is the target, its probability is produced by the hidden state at position $t-1$:

$$
p_\theta(x_t\mid x_{<t})
=\operatorname{softmax}(W_{\mathrm{LM}}h_{t-1}).
$$

Accordingly, `compute_alti_saliency_vector(model, batch, target_idx_in_seq=t)` runs the model on `input_ids[:, :t]` and explains the final query row `t-1`. Confusing target position $t$ with query position $t-1$ produces an off-by-one explanation.

## 3. Feature attribution with ALTI

### 3.1 Why raw attention is insufficient

An attention probability says how strongly a query selects a key, but ignores:

- the value vector carried by that key;
- the output projection;
- the residual path;
- mixing across layers.

ALTI uses these components to estimate token-to-token contribution.

### 3.2 Single-layer contribution

For attention head $h$, source $j$, and query $i$, a simplified contribution vector is

$$
T_i(x_j)
\propto
\sum_h W_O^{(h)}
\left(A_{i,j}^{(h)}W_V^{(h)}\tilde x_j\right),
$$

with the query's residual contribution added on the diagonal. The paper converts this vector into a normalized scalar by comparing the reconstructed output $y_i$ with and without each source contribution:

$$
d_{i,j}=\lVert y_i-T_i(x_j)\rVert_1,
\qquad
C_{i,j}
=
\frac{\max(0,\lVert y_i\rVert_1-d_{i,j})}
{\sum_k\max(0,\lVert y_i\rVert_1-d_{i,k})}.
$$

The diagnostic implementation in `src/attribution/saliency.py` builds this row-stochastic matrix for every layer. It supports Qwen grouped-query attention by inferring query/KV head layout and repeating KV heads correctly.

### 3.3 Cross-layer rollout

Let $C^{(l)}$ be the contribution matrix at layer $l$. Full diagnostic ALTI computes

$$
C_{\mathrm{roll}}
=C^{(L)}C^{(L-1)}\cdots C^{(1)}.
$$

The target saliency vector is the target query row of $C_{\mathrm{roll}}$. `compute_alti_saliency_vector` implements this forward-only rollout for batch size one.

The training loss uses a cheaper differentiable surrogate: the selected decoder layer, usually the last layer, and the $\ell_2$ norm of each transformed contribution. Full rollout is used for diagnosis; last-layer contribution is used for scalable gradient-based training.

## 4. Data attribution

### 4.1 Gradient similarity

The active data-attribution implementation represents a sample by the gradient of its language-model loss:

$$
g(z)=\nabla_\theta\mathcal L(z;\theta).
$$

A training sample $z_i$ and query $z_q$ receive cosine similarity

$$
\operatorname{DA}_\theta(z_i,z_q)
=
\frac{g(z_i)^\top g(z_q)}
{\lVert g(z_i)\rVert_2\lVert g(z_q)\rVert_2}.
$$

`NewInferenceFunction.influence_gradient_single` computes this exactly over the parameters selected by `param_filter_fn`. It is TracIn-like in spirit but uses a single current checkpoint rather than accumulating checkpoint-wise dot products.

### 4.2 Fast coarse screening

Full gradients for every training example are expensive. `saliency.py` therefore implements analytic LM-head CE gradients:

$$
\frac{\partial\mathcal L}{\partial W_{\mathrm{LM}}}
=
\frac{1}{N}\sum_t
\left(p_t-\operatorname{onehot}(y_t)\right)h_t^\top.
$$

The code computes them from hidden states without backpropagating through the transformer. It can also project them into deterministic TensorSketch/CountSketch features and cache train-side sketches. These approximations are for candidate retrieval; expensive fine matching is applied only to a small pool.

## 5. Train/test correlation matching

The main experimental flow is in `src/attribution/intervention_experiment.py`.

### Stage 0: optional global pool

In all-token mode, compute one response-level CE feature and retrieve a coarse pool of training samples. This avoids scanning the complete training set separately for every output token.

### Stage 1: test-side feature attribution

For a selected generated target token:

1. compute its ALTI saliency vector;
2. filter special, whitespace, punctuation, and other trivial sources;
3. retain the top source tokens;
4. form test pairs `source -> target`.

### Stage 1b: parameter-space pair features

For each retained pair, compute

$$
f_\theta(s\rightarrow t)
=
\nabla_\theta\operatorname{ALTI}_\theta(s\rightarrow t),
$$

usually over Q/K projections in the last attention layer. This asks which parameter changes would strengthen or weaken that specific relation.

### Stage 2: sample-level reranking

Compute token- or response-specific CE gradient similarity between the test target and the coarse train pool. Keep the highest-scoring training samples.

### Stage 3: fine pair matching

Within each retained training sample:

1. choose nontrivial response targets;
2. compute their salient source tokens;
3. compute ALTI-gradient features for train pairs;
4. rank train/test pairs by cosine similarity.

The final record is not merely “training sample 17 was influential.” It is a proposed mechanistic match:

```text
train_source -> train_target
        resembles
test_source  -> generated_target
```

## 6. Single-token and all-token modes

**Single-token mode** is the best debugging path. It analyzes one manually selected output position and produces a relatively small report.

**All-token mode** analyzes nontrivial response tokens up to a configured limit. It reuses a global coarse pool and cached train details, but it is still expensive because ALTI-gradient matching involves eager attentions and parameter gradients.

Before an all-token run, measure one token's:

- peak GPU memory;
- number of surviving source pairs;
- coarse-screen time;
- fine-match time;
- cache hit rate.

Then estimate the complete runtime. The code retries ALTI-gradient OOMs with smaller chunks and can skip overlong prefixes, but a skipped pair is missing evidence and must be reported.

## 7. Intervention and faithfulness

A saliency ranking may be plausible yet behaviorally irrelevant. Useful validation includes:

- mask or delete a high-saliency supported source;
- mask a matched unsupported source;
- flip a controlled spurious cue while preserving program semantics;
- apply a consistent alpha-renaming to source and target identifiers;
- compare the change in gold log probability or accuracy.

The desired ordering is

$$
\Delta_{\mathrm{supported}}
>
\Delta_{\mathrm{unsupported}},
\qquad
\Delta_{\mathrm{supported}}
>
\Delta_{\mathrm{cue}}.
$$

`src/eval/diagnostics/cf_robustness.py` implements a related behavioral test: it hides non-annotated context with a deterministic mask and compares clean versus masked target accuracy/NLL across models.

## 8. Saliency alignment metrics

For a target with annotated source set $P_t$, rank causal context tokens by model saliency and compute:

- hit@k: whether at least one annotated source appears in the top k;
- precision@k: annotated sources among the first k;
- recall@k: fraction of annotated sources recovered;
- average precision/mAP: quality of the full supported-source ranking.

“Honest” diagnostics exclude special tokens and an initial attention-sink prefix from the candidate negative set. Always report the candidate policy because removing easy sink tokens can substantially change mAP.

Alignment alone is not the final outcome: a model can optimize an attribution metric without becoming more correct or robust.

## 9. Source map

| File | Role |
| --- | --- |
| `src/attribution/saliency.py` | CE gradients, analytic LM-head gradients, sketches, forward ALTI rollout, and differentiable ALTI pair gradients. |
| `src/attribution/NIF.py` | Dataset/model scaffolding, generation, loss-gradient sample influence, and older experimental paths. |
| `src/attribution/intervention_experiment.py` | Multi-stage coarse-to-fine train/test correlation matching. |
| `src/attribution/process_data.py` | Convert message records into ChatML IDs/labels and filter trivial Go tokens. |
| `src/attribution/auto_annotate.py` | Ask an annotation model to mark relevant spans with `<ATTN>` tags. |
| `src/eval/diagnostics/` | Counterfactual masking and saliency-alignment comparisons. |

## 10. Current caveats

- `NIF.py` imports `src.sft.inference`, which is missing in this checkout.
- `load_model_and_tokenizer` and the main functions contain legacy hard-coded checkpoint/data paths and selected sample/token constants.
- `influence_overfit_single` intentionally raises because its old gradient-saliency path was removed.
- Full ALTI currently assumes a Qwen-style decoder and batch size one.
- Eager attention materializes quadratic attention tensors and is much more memory-intensive than SDPA.
- Several scripts mutate module-level constants from CLI arguments; save the final report metadata, not just the command line.

Treat these files as research code requiring path/config reconciliation before execution.

## 11. A safe learning exercise

A minimal conceptual trace is:

1. construct one already-tokenized ChatML sample;
2. locate the first supervised target position;
3. run `compute_alti_saliency_vector`;
4. decode and inspect the top five nontrivial sources;
5. verify the target/query off-by-one convention;
6. compare with annotation parents;
7. mask one source and observe the target log-probability change.

This isolates feature attribution before adding train-set retrieval or ALTI gradients.
