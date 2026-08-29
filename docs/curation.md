# Curation Guide

## 1. Curation in this project

Conventional data curation asks which samples or target tokens should be kept. CausalCoder adds a third option: keep the sample but augment its supervision with evidence about **why** a target is predictable.

For a token sequence $x_{1:T}$, an annotated record carries a graph

$$
E=\{(s,t,r)\},
$$

where source token $s$ supports target token $t$, and $r$ describes the program relation. Training can then select inferable targets, align saliency to graph sources, mask unsupported context, or learn graph structure as an auxiliary task.

The annotation is best understood as a conservative program-dependency approximation, not a fully causal graph in Pearl's interventionist sense.

## 2. End-to-end curation pipeline

```text
canonical FIM code
  -> reconstruct the complete program
  -> simple code tokenization with character offsets
  -> deterministic tree-sitter edges
  -> LLM-generated semantic/data-flow/API edges
  -> restrict, validate, orient, and deduplicate edges
  -> remap complete-code offsets into the final ChatML sequence
  -> retokenize with the training tokenizer
  -> expand simple-token edges to BPE-token edges
  -> save compact input_ids, labels, and attention_edges
  -> optionally augment graph edges or select target tokens
  -> train with CE plus a graph-guided objective
```

The complete program is analyzed because dependencies may cross the masked span. The final graph is remapped to what the model actually sees.

## 3. Annotation data structures

`src/annotate/utils.py` defines:

- `SubwordToken(surface, token_id, char_start, char_end)`;
- `TokenCorrelation(token_i, token_j, source, subtype, token_i_idx, token_j_idx)`;
- `AnnotationResult` for merged symbolic/neural results.

During initial annotation, “subword” is often a simple regex token with `token_id=-1`. Actual Qwen BPE token IDs are assigned during postprocessing.

Comments and Python docstrings are excluded from annotation tokens. C preprocessor directives are retained. This reduces obvious lexical leakage but also means comment semantics cannot become graph evidence.

## 4. Deterministic structural annotation

`SyntacticCheckerTool` in `src/annotate/neural_annot.py` parses supported languages with tree-sitter and maps syntax-node byte spans back to token indices.

Its structural relations include:

| Relation | Intended meaning |
| --- | --- |
| `bracket` | Matching delimiters or structural pairs. |
| `defuse` | A definition/binding and a later use. |
| `call` | A callee and its argument tokens. |
| `return` | A return construct and returned expression. |
| `type` | A type token and the declaration/value it constrains. |

The implementation includes Go-specific handling for selector/member chains, parameter/result types, receiver methods, short declarations, and type assertions.

These edges form the high-precision backbone. Tree-sitter gives syntax, not complete interprocedural data flow, alias analysis, dynamic dispatch, or repository-scale resolution.

## 5. Neural semantic annotation

`AnnotatorAgent` is a tool-calling agent using an OpenAI-compatible client. Its protocol is:

1. call `get_structural_edges`;
2. do not duplicate returned structural edges;
3. add missing `dataflow`, `semantic`, and `api` relations;
4. submit all extra pairs through `emit_correlations`.

The additional relation types are:

- `dataflow`: producer-to-consumer relationships not covered by same-binding def-use;
- `semantic`: paired control-flow/grammar concepts such as `if/else`, `try/catch`, or `async/await`;
- `api`: library usage relations such as acquire/release or call-result use.

The agent may call `search_api_docs`, but external search introduces network availability, nondeterminism, and provenance concerns. Cache results and record the annotation model/version.

The agent returns all pairs with `source="Neural"`, including the tree-sitter seeds. The `subtype` distinguishes their origin in the relation taxonomy.

## 6. Target-region and direction rules

For the MCEval-style path, `main.py` reconstructs masked spans and passes only the valid incomplete-code block to tree-sitter. Global character offsets map this slice back to the full instruction token list.

The current call restricts retained pairs so both endpoints lie within the incomplete-code region. This is safer than letting instruction prose create edges, but it can exclude useful cross-boundary evidence depending on how the region is defined.

For causal training, edges must point from available evidence to a later query token. FIM complicates this because source-code order and ChatML order differ: suffix code is visible in the user prompt even though it follows the hole in the original program. `normalize_fim_annotation_edge_direction` can orient a context-to-completion edge according to model-visible order.

Always inspect edge direction after final serialization, not only in original code order.

## 7. Remapping to Qwen BPE tokens

`postprocessing.py` and `postprocessing_safim.py`:

1. locate target spans in reconstructed code;
2. map character offsets from complete code into user/assistant ChatML positions;
3. tokenize the exact final sequence with offsets;
4. map each simple token to overlapping BPE tokens;
5. expand each conceptual edge to every source/target BPE pair;
6. drop self-edges and duplicate pairs.

This stage is fragile because a one-character offset error can attach a semantically valid edge to the wrong model token. Visualization is not cosmetic; it is a required data-quality check.

Useful invariants:

- decoded tokens cover the intended text;
- completion tokens lie in the supervised region;
- edge endpoints decode to the expected source/target surfaces;
- every endpoint is within bounds;
- context-to-completion edges point forward in model sequence order;
- the percentage of dropped unmapped edges is recorded.

## 8. Optional graph augmentation

`src/data/edge_augment.py` densifies direct edges by shortest-path reachability.

For a path of $h$ edges, the generated pair receives

$$
w_h=\delta^{h-1},
$$

where $\delta\in(0,1]$ is the decay. Direct edges keep weight 1. The directed mode follows edge direction; the undirected mode connects token pairs in the same component.

This is label augmentation, not additional program analysis. A transitive relation can be useful, but it is weaker evidence than a direct relation and may amplify annotation errors. Bound `max_hops`, inspect graph-density growth with `augment_stats.py`, and compare direct-only versus augmented ablations.

## 9. Target selection

The paper's selective NTP masks targets explicitly marked uninferable while preserving grammar-intrinsic tokens.

The current dataset implements a related but not identical mechanism: when `token_select` is enabled, a completion token with precomputed `comp_teacher_nll > threshold` receives label `-100`, except optionally retained special tokens.

This is teacher-NLL gating, not direct use of the annotator's “uninferable” decision. Document it as an implementation approximation or baseline unless the data field is proven to encode the paper's exact criterion.

## 10. Graph-guided training objectives

### 10.1 CE baseline

$$
\mathcal L_{\mathrm{CE}}
=-\frac{1}{|T_{\mathrm{sup}}|}
\sum_{t\in T_{\mathrm{sup}}}
\log p_\theta(x_t\mid x_{<t}).
$$

### 10.2 Saliency alignment

`loss.py` computes last-layer ALTI-like contribution magnitudes

$$
c_{t,s}=\lVert T_t(x_s)\rVert_2.
$$

Annotated sources are positives; earlier non-annotated causal sources are negatives. The implementation supports:

- `softmax`: multi-positive softmax NLL;
- `softmax_margin`: log-saliency NLL with a negative floor;
- `margin_bce`: decoupled positive/negative penalties;
- `ranknet`: pairwise logistic ranking;
- `contrastive`: pairwise hinge/triplet ranking.

The trainer combines it as

$$
\mathcal L
=\mathcal L_{\mathrm{CE}}+\lambda\mathcal L_{\mathrm{sal}}.
$$

Only rows with at least one valid positive and negative contribute. Special tokens and an attention-sink prefix can be excluded from negatives.

### 10.3 Counterfactual shortcut masking

`ce_shortcut_mask` randomly hides non-annotated context keys while protecting graph sources, special tokens, an optional prefix, and a local recency window.

Variant A optimizes CE on the masked input. Variant B uses clean CE plus

$$
\beta\,
D_{\mathrm{KL}}\!
\left(p_{\mathrm{clean}}\,\Vert\,p_{\mathrm{masked}}\right)
$$

only where the clean model is already correct. Per-target and graph-weight-aware masking are also implemented.

### 10.4 Edge prediction

`ce_edge_pred` learns source and target projections and predicts graph edges with positive/negative BCE. The auxiliary head is training-only. It encourages hidden states to encode dependencies without directly forcing an attribution metric.

### 10.5 Graph attention bias

`ce_attn_bias` adds a learned scalar bonus to attention logits on graph edges during training. The graph is absent at inference, so any benefit must be internalized in model weights.

These alternatives answer different hypotheses and should not be reported interchangeably as “CausalCoder.”

## 11. Training integration

`AnnotatedSFTDataset` converts edges to `annot_pairs`; optional augmentation adds `annot_weights` and per-node relevance weights. The collator pads sequences but keeps edge lists ragged.

`AnnotatedSFTTrainer.compute_loss` dispatches by `loss_mode`. Saliency modes request attentions and hidden states from an eager-attention model. Masking, edge prediction, and attention bias can use SDPA.

LoRA is enabled by default in `train.py`. Saliency is differentiable through attention/value/output paths and LoRA parameters. Auxiliary head/gate parameters live on the Trainer and are not part of the saved adapter, a detail that matters if training is resumed.

## 12. Annotation quality assurance

Audit a stratified sample by language, relation type, sequence length, and prompt/completion crossing. Measure at least edge precision; recall is harder because the full dependency set is unknown.

Check common errors:

- duplicate surfaces mapped to the wrong occurrence;
- byte versus character offsets;
- suffix evidence oriented backward after FIM serialization;
- simple-token-to-BPE expansion producing excessive edges;
- member access and qualified types being under-annotated;
- dynamic or cross-file dependencies hallucinated by the LLM;
- unsupported tokens mislabeled as negatives merely because the analyzer missed an edge;
- annotation model nondeterminism across retries.

The paper reports a manual validation of its structural-support rules, but that result should not be transferred to a different data revision or annotation prompt without re-audit.

## 13. Cost and reproducibility

Annotation can be API-bound rather than GPU-bound. Use:

- one-sample structural-only smoke tests;
- a small cached pilot with `num_workers=1`;
- request timeout, bounded retries, and exponential backoff;
- resumable row-level caches;
- periodic flushes to a new output path;
- recorded model name, base URL class, prompt version, code revision, and error counts.

Do not place API keys in YAML or logs. `src/annotate/main.py` currently sets localhost proxy defaults and hard-codes input/output behavior in its main block; reconcile these before a formal run.

## 14. Current implementation caveats

- `configs/annotate/default.yaml` is not automatically loaded by the legacy annotation entry points.
- The tokenizer helper name and actual fallback checkpoint disagree.
- `postprocessing.py` executes its conversion loop at import time rather than only under a main guard.
- Some annotation scripts write back into their input JSONL during periodic flushes; preserve an immutable raw copy.
- The paper's explicit uninferable-token annotation is not clearly represented by the current agent schema.
- The neural prompt encourages many edges, while the paper emphasizes conservative high-confidence labels; density and precision must be measured.
- Repository-scale and cross-file dependencies are outside the current local tree-sitter/agent context.

## 15. Practical learning path

Start with one short function:

1. print simple tokens with indices and offsets;
2. run structural-only annotation;
3. render the edge graph;
4. add neural edges and compare the difference;
5. map to Qwen BPE and decode every endpoint;
6. load the compact row through `AnnotatedSFTDataset`;
7. compute CE-only and saliency loss on one batch;
8. perturb one edge and verify the diagnostic changes as expected.

Only after this chain is correct should annotation concurrency or multi-GPU training be increased.
