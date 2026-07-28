# Spurious Correlation Measurement Runbook

这个文档用于测一个 fine-tuned code model 在测试失败样本上的 spurious correlation 比例。主流程只包含两步：先用 ALTI 生成 feature attribution 文件，再用 tree-sitter 程序结构分析这些 high-saliency source -> target correlations 是否 structurally supported。

## 1. 相关脚本

主脚本：

- `src.feature_attribution_batch_evaluation`: 批量生成每个 test sample 的 ALTI feature attribution JSON。测新模型时先跑这个。
- `src.spurious_correlation_analysis`: 读取上一步的 feature JSON，定位第一个 lexical mismatch，并计算 spurious correlation 指标。
- `src.multilang_split`: 如果本地没有 `data/multilang_splits/*_{train,test}.jsonl`，先用它从原始多语言 SFT 数据生成 deterministic train/test splits。
- `src.multimodel_multilang_spurious_report`: 可选。把多个 model/language 的 `summary.json` 汇总成论文表格和图。

不要混淆：

- `src.intervention_experiment.py` 是 case study 用的训练数据归因和 correlation matching 脚本，不是批量统计 spurious rate 的主入口。
- `src.feature_attribution_evaluation` 是单样本入口；批量测模型时推荐用 `src.feature_attribution_batch_evaluation`，它会复用一次加载好的模型。

## 2. 输入和输出

输入：

- fine-tuned model path，例如某个 LoRA merge 后模型目录或 HuggingFace 模型目录。
- test JSONL，例如 `data/multilang_splits/go_test.jsonl`。
- train JSONL，例如 `data/multilang_splits/go_train.jsonl`。这只用于估计六类 spurious pattern 里的 training-set co-occurrence 类别；spurious rate 本身主要由 feature attribution 和 structural support 决定。

注意：`data/` 是本仓库的 ignored large-data directory，`data/multilang_splits/` 不随代码 commit 提交。如果本地没有这些 split 文件，需要按第 4 节先生成；如果原始数据也不存在，需要先准备第 4 节列出的原始 JSONL 文件。

输出：

- ALTI feature files: `<ATTR_DIR>/feature/*_feature.json`
- batch status: `<ATTR_DIR>/reports/batch_status.tsv`
- structural analysis: `<SPURIOUS_DIR>/summary.json`
- per-sample details: `<SPURIOUS_DIR>/per_sample.jsonl`
- case candidates: `<SPURIOUS_DIR>/case_candidates.tsv`

论文主结果目前使用：

- spurious rate: `summary.json` 里的 `metrics.loose_mapped_spurious@50`
- 六类 breakdown: `summary.json` 里的 `metrics.loose_mapped_category_proportions@50`

## 3. 环境检查

在 repo 根目录运行：

```bash
cd /home/yilu/Repo/Empirical-Influence-Function

python -c "import torch, transformers, accelerate; print('model deps ok')"
python -c "import tree_sitter, tree_sitter_go, tree_sitter_java, tree_sitter_javascript, tree_sitter_python; print('tree-sitter deps ok')"
```

如果只测某一种语言，至少要安装对应的 `tree_sitter_<language>` 包。

## 4. 准备多语言 train/test splits

如果 `data/multilang_splits/` 不存在，先运行：

```bash
cd /home/yilu/Repo/Empirical-Influence-Function

python -m src.multilang_split \
  --output-dir data/multilang_splits \
  --test-size 1000 \
  --seed 42
```

该命令默认读取以下原始数据：

- `data/go/go_single_train_v2_chatml.jsonl`
- `data/java/java_single_train_codesearchnet_20000_chatml.jsonl`
- `data/javascript/javascript_single_train_codesearchnet_20000_chatml.jsonl`
- `data/python/python_single_train_codesearchnet_20000_chatml.jsonl`

生成后应能看到：

```text
data/multilang_splits/go_train.jsonl
data/multilang_splits/go_test.jsonl
data/multilang_splits/java_train.jsonl
data/multilang_splits/java_test.jsonl
data/multilang_splits/javascript_train.jsonl
data/multilang_splits/javascript_test.jsonl
data/multilang_splits/python_train.jsonl
data/multilang_splits/python_test.jsonl
```

## 5. 单个模型、单个语言的标准命令

下面以 Go 为例。把 `MODEL_PATH` 换成要测的模型路径。

```bash
cd /home/yilu/Repo/Empirical-Influence-Function

MODEL_NAME=my-model
MODEL_PATH=/path/to/fine-tuned-model
LANG=go

TEST_DATA=data/multilang_splits/${LANG}_test.jsonl
TRAIN_DATA=data/multilang_splits/${LANG}_train.jsonl
ATTR_DIR=runs/spurious_eval/attribution/${MODEL_NAME}/${LANG}
SPURIOUS_DIR=spurious_correlation_results/spurious_eval/${MODEL_NAME}/${LANG}

CUDA_VISIBLE_DEVICES=0 python -m src.feature_attribution_batch_evaluation \
  --model-path "${MODEL_PATH}" \
  --test-data "${TEST_DATA}" \
  --start-idx 0 \
  --end-idx 99 \
  --output-dir "${ATTR_DIR}" \
  --feature-evaluation-mode saliency_only \
  --feature-ranking-mode alti \
  --feature-k-values 5,10,20,50 \
  --max-feature-sources 50 \
  --top-k-prompt-tokens 50 \
  --max-output-tokens 20 \
  --generation-limit 128 \
  --feature-max-prefix-len 2048 \
  --attn-implementation eager

python -m src.spurious_correlation_analysis \
  --feature-dir "${ATTR_DIR}/feature" \
  --output-dir "${SPURIOUS_DIR}" \
  --language "${LANG}" \
  --train-data "${TRAIN_DATA}" \
  --k-values 5,10,20,50 \
  --source-ranking alti_saliency
```

说明：

- `--feature-evaluation-mode saliency_only` 表示只算纯 ALTI saliency，不跑 expensive masking intervention。
- `--feature-ranking-mode alti` 表示使用 raw ALTI 排序，是论文主结果。
- `--max-feature-sources 50` 必须保留，否则 `Spurious@50` 没有足够的 source correlations。
- `--feature-max-prefix-len 2048` 会跳过太长 prefix 的 target，避免 ALTI rollout 显存爆掉。跳过数量会体现在 `summary.json` 的 `status_counts` 里。
- `--attn-implementation eager` 很重要；ALTI 需要显式 attention probabilities。

## 6. 查看结果

不用依赖 `jq`，直接用 Python 读主指标：

```bash
python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("spurious_correlation_results/spurious_eval/my-model/go/summary.json").read_text())
metrics = summary["metrics"]

print("input_count:", summary["input_count"])
print("ok_count:", summary["ok_count"])
print("status_counts:", summary["status_counts"])
print("loose_mapped_spurious@50:", metrics["loose_mapped_spurious@50"])
print("category breakdown:", metrics["loose_mapped_category_proportions@50"])
PY
```

主结果写成百分比时乘以 100。例如 `0.7391` 就是 `73.91%`。

## 7. 四语言批量模板

如果同一个模型要测 Go、Java、JavaScript、Python，可以用：

```bash
cd /home/yilu/Repo/Empirical-Influence-Function

MODEL_NAME=my-model
MODEL_PATH=/path/to/fine-tuned-model

for LANG in go java javascript python; do
  TEST_DATA=data/multilang_splits/${LANG}_test.jsonl
  TRAIN_DATA=data/multilang_splits/${LANG}_train.jsonl
  ATTR_DIR=runs/spurious_eval/attribution/${MODEL_NAME}/${LANG}
  SPURIOUS_DIR=spurious_correlation_results/spurious_eval/${MODEL_NAME}/${LANG}

  CUDA_VISIBLE_DEVICES=0 python -m src.feature_attribution_batch_evaluation \
    --model-path "${MODEL_PATH}" \
    --test-data "${TEST_DATA}" \
    --start-idx 0 \
    --end-idx 99 \
    --output-dir "${ATTR_DIR}" \
    --feature-evaluation-mode saliency_only \
    --feature-ranking-mode alti \
    --feature-k-values 5,10,20,50 \
    --max-feature-sources 50 \
    --top-k-prompt-tokens 50 \
    --max-output-tokens 20 \
    --generation-limit 128 \
    --feature-max-prefix-len 2048 \
    --attn-implementation eager

  python -m src.spurious_correlation_analysis \
    --feature-dir "${ATTR_DIR}/feature" \
    --output-dir "${SPURIOUS_DIR}" \
    --language "${LANG}" \
    --train-data "${TRAIN_DATA}" \
    --k-values 5,10,20,50 \
    --source-ranking alti_saliency
done
```

## 8. 多模型论文图表汇总

如果输出目录组织成下面形式：

```text
spurious_correlation_results/spurious_eval/
  model-a/go/summary.json
  model-a/java/summary.json
  model-b/go/summary.json
  ...
```

则可以生成汇总 CSV、LaTeX 表和单栏图：

```bash
python -m src.multimodel_multilang_spurious_report \
  --input-dir spurious_correlation_results/spurious_eval \
  --output-dir spurious_correlation_results/paper_draft/figures \
  --rate-key loose_mapped_spurious@50 \
  --category-key loose_mapped_category_proportions@50
```

生成的主要文件：

- `spurious_correlation_results/paper_draft/figures/multimodel_multilang_spurious_breakdown.csv`
- `spurious_correlation_results/paper_draft/figures/table_multimodel_multilang_spurious_breakdown.tex`
- `spurious_correlation_results/paper_draft/figures/figure_multimodel_multilang_spurious_breakdown.pdf`
- `spurious_correlation_results/paper_draft/figures/figure_multimodel_multilang_spurious_breakdown.tex`

## 9. 常见问题

1. `ok_count` 小于 100 是正常的。只有生成失败、能定位第一个 lexical mismatch、且该 target 的 ALTI attribution 可用的样本会进入主统计。
2. 如果 `uncovered_target` 很多，通常是 `--max-output-tokens` 太小，或者 first mismatch 出现在没有计算 ALTI 的 target 位置。可以适当提高 `--max-output-tokens`，但会增加计算量。
3. 如果显存爆掉，先降低 `--feature-max-prefix-len`，例如 1536；或者减少一次跑的 index 范围。
4. 如果只想快速试跑，先用 `--start-idx 0 --end-idx 4`，确认 feature JSON 和 `summary.json` 都能生成后再跑 100 个样本。
5. 论文主结果不要用 `signed` 或 `signed_clip`；当前主结果统一使用 `--feature-ranking-mode alti` 和 `--source-ranking alti_saliency`。
6. 如果报 `No module named src.feature_attribution_batch_evaluation`，说明代码 commit 漏掉了 `src/feature_attribution_batch_evaluation.py`；如果报 `data/multilang_splits/... does not exist`，按第 4 节先生成 split。
