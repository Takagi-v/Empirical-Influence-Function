# Experiment Guide

## 1. What must be demonstrated

The project makes a chain of claims, and each needs different evidence:

1. fine-tuning creates or preserves shortcut reliance;
2. program-dependency annotations identify better evidence;
3. graph-guided training changes internal reliance;
4. that change improves robustness and completion quality;
5. improvements are not caused only by extra compute, regularization, or data filtering.

A strong experiment therefore combines behavior, mechanism, intervention, and fairness controls.

## 2. The paper's empirical diagnosis

The manuscript's diagnosis uses CodeSearchNet functions from Python, Go, Java, and JavaScript, with Qwen2.5-Instruct/Coder models at 0.5B, 1.5B, and 7B.

For each held-out generation it:

1. finds the first nontrivial lexical error;
2. computes ALTI over context sources for that target;
3. takes the top-k salient sources;
4. classifies each source as structurally supported or unsupported;
5. uses gradient-based data attribution to inspect influential training samples.

The spurious-correlation rate is

$$
\operatorname{SCR@K}
=
\frac{
\#\{\text{top-K sources without structural support}\}
}{
K
}.
$$

The paper reports 676 analyzable first-error cases from 1,200 generations and a mean SCR@10 of 72.8%. Treat these as manuscript results, not values recalculated by the current checkout.

The reported recurring patterns are training-set co-occurrence, comment/string leakage, lexical similarity, scope confusion, and prompt-format shortcuts.

## 3. Main paper evaluation

### 3.1 Research questions

- **RQ1:** Does CausalCoder improve pass@k and CodeBLEU?
- **RQ2:** Does it reduce SCR@10 on the model's own free generations?
- **RQ3:** What do selective NTP and contrastive saliency each contribute?

### 3.2 Data and benchmarks

The paper describes 10,000 FIM training functions per language from CodeSearchNet, 40,000 total. Evaluation uses FIM variants of HumanEval-X and McEval in the same four languages.

Execution-based benchmarks are primary because a completion may differ textually from the reference while remaining correct. CodeBLEU is supplementary surface fidelity.

### 3.3 Paper baselines

| Method | Curation level | What changes |
| --- | --- | --- |
| NTP | none | Standard next-token CE on all supervised completion tokens. |
| Token Cleaning | token | Masks low-quality target tokens according to reference/base-model loss signals. |
| LESS | sample | Selects influential training samples by gradient similarity. |
| CausalCoder | within-token evidence | Selects inferable targets and aligns saliency to program-supported sources. |

The comparison is designed to test whether changing *which evidence supports a token* adds value beyond selecting samples or target tokens.

### 3.4 Paper results

The manuscript reports that CausalCoder:

- improves average pass@1 by 2.6 points over NTP on both HumanEval-X and McEval;
- improves average pass@5 by roughly 3 points;
- reduces average SCR@10 from 77.3% to 56.8%, a 20.5-point absolute reduction;
- gains most at smaller model scales;
- obtains complementary effects from selective NTP and contrastive saliency.

These are reported findings from `reports/paper.pdf`. Reproduction should quote the exact paper table and use the matching artifacts rather than infer results from current configuration files.

## 4. Component ablations

The paper's key ablation is:

| Configuration | Main interpretation |
| --- | --- |
| NTP | Reference behavior. |
| + selective NTP | Tests removal of memorization pressure on uninferable targets. |
| + contrastive saliency | Tests direct evidence realignment. |
| both | Tests complementarity. |

Expected separation:

- selective NTP should improve accuracy and modestly reduce SCR;
- contrastive saliency should sharply reduce SCR, possibly with a smaller accuracy gain;
- both should provide the best accuracy/robustness balance.

The current trainer also permits research ablations not necessarily in the paper:

- direct versus multi-hop/undirected graph augmentation;
- saliency loss family and layer;
- negative sampling and special/sink exclusion;
- shortcut masking and invariance KL;
- edge prediction;
- graph attention bias;
- teacher-NLL token selection;
- LoRA versus full fine-tuning.

Change one causal factor per ablation whenever possible.

## 5. Controlled spurious-correlation experiment

The detailed design is in [FIM_spurious_correlation_experiment_design.md](FIM_spurious_correlation_experiment_design.md).

Its minimal task uses a variable-binding/def-use FIM problem. A semantic program relation $G$ determines the correct target $Y$, while a behavior-preserving cue $Z$ is correlated with $Y$ only in training:

$$
P_{\mathrm{train}}(Z=Y)=\rho,
\qquad
Y\bigl(do(Z=0)\bigr)=Y\bigl(do(Z=1)\bigr).
$$

Test variants hold the program and target fixed while making the cue aligned, conflicting, or absent.

The essential training groups are:

| Group | Data | Objective |
| --- | --- | --- |
| A | no cue | NTP |
| B | random cue, $\rho=0.5$ | NTP |
| C | biased cue, $\rho=0.95$ | NTP |
| D | biased cue | NTP + relation loss |
| E | biased cue | shuffled relation labels |
| F | counterfactually balanced cue | NTP |

The shuffled-label control is crucial: it distinguishes useful program relations from generic extra regularization.

## 6. Metrics

### 6.1 Functional and surface behavior

- pass@1 and pass@k;
- exact target accuracy for controlled single-token/identifier tasks;
- CodeBLEU;
- normalized exact match and edit similarity when executable tests do not exist;
- parse/compile rate and judge timeout/failure counts.

### 6.2 Distribution-shift behavior

$$
\operatorname{SCGap}
=
\operatorname{Acc}_{\mathrm{aligned}}
-
\operatorname{Acc}_{\mathrm{conflicting}}.
$$

Also report neutral accuracy, worst-group accuracy, paired cue-flip rate, and the change in gold-versus-distractor log odds.

### 6.3 Internal mechanism

- positive-source saliency mass;
- cue and hard-negative saliency mass;
- annotation precision/recall/mAP at k;
- SCR@10 on free generations;
- layer-, distance-, relation-, and prefix/suffix-side breakdowns.

### 6.4 Causal faithfulness

Compare the gold log-probability change after interventions on supported sources, matched negatives, and cues. A useful model should be more sensitive to genuine evidence than to behavior-preserving cue changes.

## 7. Fair-comparison checklist

All methods in one table should share:

- identical base checkpoint and tokenizer;
- identical canonical examples and split;
- equal training-token budget or explicitly reported wall-clock budget;
- identical optimizer, LR schedule, effective batch size, epoch/step count, and seeds;
- identical LoRA target modules or full-fine-tuning policy;
- identical decoding temperature, top-p, sample count, and maximum tokens;
- the same execution judge, timeout, and postprocessing;
- checkpoint selection without using the final test set;
- at least three seeds for headline comparisons.

Saliency training uses eager attention while CE can use SDPA, so wall-clock and memory are inherently different. Report both task quality and resource cost.

## 8. Current evaluation code

`src/eval/run_eval.py` is intended to:

1. resolve datasets from `configs/datasets.yaml`;
2. evaluate a trained experiment checkpoint or ad-hoc model;
3. route test-bearing datasets to pass@k plus CodeBLEU;
4. route datasets without tests to CodeBLEU only;
5. write per-dataset JSON and `summary.json`.

In this checkout, the dataset registry and `scripts/benchmark/benchmark_eval.py` are absent, so the unified entry point cannot run without recovering the matching files.

Go-specific scripts provide more concrete alternatives:

- `evaluate_go_single_predictions.py`: reconstruct code and execute Go tests;
- `oracle_eval_go_single.py`: insert gold targets to validate the judge/data pipeline;
- `evaluate_codesearchnet_go_internal.py`: measure in-domain exact/edit/CodeBLEU and teacher-forced saliency alignment.

Always run the oracle before trusting a zero or unexpectedly low model score.

## 9. Diagnostic experiments in src/eval

### 9.1 Counterfactual shortcut removal

`cf_robustness.py` uses the same deterministic unsupported-token mask for every model and reports clean/masked gold-token accuracy and NLL. Smaller degradation means greater robustness to removing putative shortcuts.

### 9.2 Honest saliency alignment

`honest_map_diagnostic.py` and `honest_map_diag_cfmask.py` compare last versus middle layers and raw versus sink/special-excluded candidate sets.

The former contains hard-coded legacy root/model/checkpoint paths; the latter uses the registry. They are diagnostic research scripts, not portable benchmarks as currently written.

### 9.3 Interpretation warning

High saliency mAP after directly optimizing saliency can be a Goodhart effect. The strongest evidence is a joint pattern:

```text
higher alignment
+ lower cue/unsupported sensitivity
+ higher supported-token intervention effect
+ improved conflicting/worst-group behavior
+ maintained clean performance
```

## 10. Baseline implementation status

The repository contains YAML and CLI wrappers for Token Cleaning, XTF, CLEAR-style filtering/correction, and LLM-assisted code cleaning. Their Python source modules under `src/baseline/` are missing in this checkout; only compiled bytecode caches remain. The wrappers therefore describe intended interfaces but are not auditable or safely reproducible from source here.

Configured transformations are conceptually:

- Token Cleaning: retain a global top ratio of scored target tokens;
- XTF: mask tokens using RI/PCP/TR-style scores and thresholds;
- CLEAR: drop low-confidence samples and optionally replace responses;
- LLM code cleaning: apply externally generated cleaned responses;
- IBFT: invoked by a shell training script and should be audited separately.

Do not report these baselines until the exact source revision and scoring artifacts are restored.

## 11. Statistical reporting

Use paired analysis for counterfactual variants from the same base program. Bootstrap by repository or base program, not by correlated variant row. Report:

- sample and judged counts;
- seed-level values, mean, and standard deviation;
- confidence intervals for paired effects;
- skipped, timed-out, unparsable, and OOM cases;
- predeclared primary metrics;
- multiple-comparison correction for large ablation grids.

A failed or skipped sample must not silently disappear from the denominator.

## 12. Run discipline

Before a full experiment:

1. validate one raw-to-canonical example;
2. run oracle evaluation on a few examples;
3. load one compact row and one batch;
4. run one forward/backward step;
5. record peak GPU memory and step time;
6. estimate total annotation/training/evaluation time;
7. choose batch size, accumulation, number of GPUs, and concurrency from measured bottlenecks;
8. run a small end-to-end seed;
9. only then launch the full grid.

Save the resolved configuration, command, environment, git revision, data fingerprint, model revision, random seed, hardware, start/end time, and output directory.
