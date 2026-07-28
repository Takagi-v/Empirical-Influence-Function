# 虚假相关性人工标注说明

这个文档给人工标注者使用。标注目标是验证：我们用程序结构判定 ALTI 找到的高显著性 `source token -> target token` 是否为虚假相关性的做法，和人工判断是否一致。

这里的“虚假相关性”指：模型生成某个 target token 时高度依赖了某个 source token，但从程序语义、数据依赖、控制依赖或局部代码结构来看，这个 source token 不应该是生成该 target token 的合理依据。

标注者请拉取 `feat/qwen3-moe-alti-correlation` 分支，打开 `spurious_correlation_results/human_annotation/human_annotation_top10.tsv` 进行标注，并结合 `spurious_correlation_results/human_annotation/full_code/` 中的完整代码判断。之前基于旧 signed_clip 排名的标注表不能直接继续标，本轮只使用纯 ALTI saliency top-10。已经标过的旧 TSV 可以交给实验负责人，用脚本自动迁移其中仍然对应同一条 correlation 的标签。

## 1. 标注任务

对每个生成失败的样本，我们取模型生成代码中的第一个错误 token 作为 target token。ALTI 会给出模型生成这个 target token 时最依赖的 top-10 source tokens。本版本使用纯 ALTI saliency 排名，不使用 signed / signed_clip 等新方法。这里的 top-10 按 `alti_saliency` 降序排列。人工标注者需要逐行判断每个 `source token -> target token` 是否是虚假相关性。

标注完成后，我们会把人工标签和自动方法的标签进行比较，计算准确率、召回率、F1 和集合重合度。

## 2. 数据位置

原始 feature attribution 结果在：

```bash
attribution_results_feature_alti_saliency_full100/feature/
```

注意：不要使用旧目录 `attribution_results_feature_loo_overlap_signed_clip_full100/feature/` 做本轮人工标注。旧目录里的 `method_top` 是 signed_clip 排名，不是纯 ALTI saliency 排名。

程序结构分析结果在：

```bash
spurious_correlation_results/
```

给人工标注者使用的盲标表是：

```bash
spurious_correlation_results/human_annotation/human_annotation_top10.tsv
```

每个样本的完整代码上下文在：

```bash
spurious_correlation_results/human_annotation/full_code/
```

自动方法的标签在：

```bash
spurious_correlation_results/human_annotation/human_annotation_top10_auto_key.tsv
```

注意：`human_annotation_top10_auto_key.tsv` 不要给标注者看。它只用于标注结束后和人工标签做比较。

## 3. 环境检查

在 repo 根目录运行：

```bash
cd /home/yilu/Repo/Empirical-Influence-Function
python -c "import tree_sitter, tree_sitter_go; print('tree-sitter ok')"
```

如果输出 `tree-sitter ok`，说明 tree-sitter 环境正常。

## 4. 重新生成程序结构分析结果

如果 `spurious_correlation_results/per_sample.jsonl` 已经存在，可以跳过这一步。

```bash
python -m src.spurious_correlation_analysis \
  --feature-dir attribution_results_feature_alti_saliency_full100/feature \
  --output-dir spurious_correlation_results \
  --k-values 5,10,20,50 \
  --source-ranking alti_saliency
```

这一步会生成：

- `spurious_correlation_results/summary.json`
- `spurious_correlation_results/per_sample.jsonl`
- `spurious_correlation_results/case_candidates.tsv`

## 5. 生成标注表

当前已经生成好的正式标注表包含 30 个样本。每个样本标纯 ALTI top-10 source tokens，一共 300 个 `source -> target` 相关性。

为了减少旧标注工作的浪费，当前表保留了旧 signed_clip 标注表中仍可被纯 ALTI 结构分析覆盖的 25 个样本，并用固定随机种子补了 5 个新样本。旧表中有 5 个样本在纯 ALTI 分析下是 `uncovered_target`，不能用于本轮自动评估。

当前表对应的样本编号是：

```text
test0, test6, test8, test16, test18, test21, test27, test28, test34, test38,
test39, test41, test49, test51, test53, test55, test57, test58, test65, test66,
test68, test69, test70, test79, test84, test86, test90, test91, test92, test95
```

复现当前正式表的命令是：

```bash
python -m src.export_spurious_annotation_sheet \
  --top-k 10 \
  --max-cases 30 \
  --selection test_index \
  --case-indices 0,6,8,16,18,21,27,28,34,38,39,41,49,51,53,55,57,58,65,66,68,69,70,79,84,86,90,91,92,95
```

输出文件：

```bash
spurious_correlation_results/human_annotation/human_annotation_top10.tsv
spurious_correlation_results/human_annotation/human_annotation_top10_auto_key.tsv
spurious_correlation_results/human_annotation/full_code/
```

如果只想先做小规模试标，可以标 5 个样本，共 50 个相关性：

```bash
python -m src.export_spurious_annotation_sheet --top-k 10 --max-cases 5
```

如果时间足够，可以标当前所有可覆盖的生成失败样本。当前是 42 个样本，共 420 个相关性：

```bash
python -m src.export_spurious_annotation_sheet --top-k 10 --max-cases 42
```

如果只想标指定样本，例如 `test80`, `test89`, `test90`：

```bash
python -m src.export_spurious_annotation_sheet \
  --top-k 10 \
  --case-indices 80,89,90 \
  --max-cases 99
```

如果完全重新开始一轮随机标注，可以用下面的命令生成一份新的随机 30-case 表：

```bash
python -m src.export_spurious_annotation_sheet \
  --top-k 10 \
  --max-cases 30 \
  --selection random \
  --seed 42
```

不要用自动虚假相关性分数挑选样本，否则会让评估偏向自动方法。`--selection high_spurious` 只适合找论文里的定性 case，不适合用于人工准确性评估。

## 6. 迁移旧标注

如果标注者已经在旧 signed_clip 表中标了一部分，不要手工复制标签。把旧的已标 TSV 保存成一个单独文件，例如：

```bash
spurious_correlation_results/human_annotation/old_signed_clip_labeled.tsv
```

然后运行：

```bash
python -m src.transfer_spurious_annotation_labels \
  --old-annotation spurious_correlation_results/human_annotation/old_signed_clip_labeled.tsv \
  --new-annotation spurious_correlation_results/human_annotation/human_annotation_top10.tsv \
  --output spurious_correlation_results/human_annotation/human_annotation_top10_transferred.tsv
```

这个脚本只会在下面字段完全一致时迁移标签：

- `test_sample_index`
- `target_token_index`
- `target_token`
- `source_token_index`
- `source_token`

也就是说，它只迁移同一个 target 上同一个 source token 的标签。旧表中不再对应纯 ALTI top-10 的行不会迁移，需要重新标注。当前 300 行新表和旧表最多有 75 行可以直接迁移；实际能迁移多少取决于旧表已经标了多少行。

如果运行了迁移脚本，后续标注者应继续填写 `human_annotation_top10_transferred.tsv` 中仍为空的 `human_label` 行。最后评估时，把填完的 transferred TSV 作为 `--annotation` 输入即可。

## 7. 标注表字段

人工只需要打开：

```bash
spurious_correlation_results/human_annotation/human_annotation_top10.tsv
```

每一行是一个 `source token -> target token` 相关性。

重要字段如下：

- `item_id`: 每个相关性的唯一 ID。
- `test_sample_index`: eval 样本编号。
- `target_token`: 模型生成时被解释的 tokenizer token。
- `generated_first_error`: 模型生成代码中的第一个错误 lexical token。
- `reference_token`: reference code 中对应的正确 lexical token。
- `source_rank`: source token 在 ALTI top-10 中的排名，按 `alti_saliency` 降序排列。
- `source_method_rank`: 该 source token 在原始 feature 文件 `method_top` 中的排名，仅用于记录，不是标注排序依据。
- `source_ranking`: 当前标注表使用的排序方式，正式表应为 `alti_saliency`。
- `source_token`: ALTI 认为重要的 source token。
- `source_context`: source token 周围上下文，`[SOURCE:...]` 标出 source。
- `target_context`: target token 周围上下文，`[TARGET:...]` 标出 target。
- `generated_completion_preview`: 模型生成的代码片段。
- `reference_completion_preview`: reference 代码片段。
- `full_generated_code_path`: 当前样本的完整 generated code 文件路径。
- `full_reference_code_path`: 当前样本的完整 reference code 文件路径。
- `full_generated_completion_path`: 模型生成的完整补全内容。
- `full_reference_completion_path`: reference 的完整补全内容。
- `full_prompt_path`: 原始 prompt 文本。
- `human_label`: 标注者要填写的主标签。
- `human_confidence`: 标注者填写的置信度。
- `human_rationale`: 可选，简短说明标注原因。

标注时不要只看 `generated_completion_preview` 和 `reference_completion_preview`。这两个字段只是快速预览。正式判断应以 `full_generated_code_path` 和 `full_reference_code_path` 指向的完整代码上下文为准。

`full_code/` 中每个样本目录包含：

- `generated.go`: 将模型生成内容填入 `<MID>` 后得到的完整代码上下文。
- `reference.go`: 将 reference 内容填入 `<MID>` 后得到的完整代码上下文。
- `generated_completion.txt`: 模型生成的完整补全部分。
- `reference_completion.txt`: reference 的完整补全部分。
- `prompt.txt`: 原始 prompt，包括任务描述、代码片段和带 `<MID>` 的目标函数。

## 8. 标注标签

`human_label` 只填写三种标签：

```text
S = 虚假相关性
N = 非虚假相关性，也就是合理依赖
U = 不确定
```

### 标为 `S`

如果 source token 对 target token 的生成没有合理的程序语义依赖，就标 `S`。

典型情况：

- source 是 package path、repository path、自然语言 prompt、注释或无关文本。
- source 是通用模板 token，例如 `func`, `var`, `if`, `return`, `:=` 或标点符号，但 target 是具体变量名、函数名、字段名或类型名。
- source 和 target 只是表面字符串相似，但从代码逻辑看不应该决定 target。
- source 来自无关作用域、无关变量、无关函数或无关字段。
- source 是导致错误变量名、错误函数名或错误类型名的局部 shortcut。

### 标为 `N`

如果 source token 对 target token 的生成有合理程序依赖，就标 `N`。

典型情况：

- source 和 target 是同一个变量、字段、函数、类型或 selector chain 的一部分。
- source 是 target 所在表达式、赋值语句、函数调用或参数传递中的必要成分。
- source 是控制语句、guard condition 或循环变量，并且合理影响 target 所在语句。
- source 是 receiver、argument、field access、index expression 或 type annotation 中和 target 有直接关系的 token。
- source 是 reference 或 generated code 中应该被复制、一致使用或继续使用的 identifier。

### 标为 `U`

如果上下文不足或无法可靠判断，就标 `U`。

典型情况：

- tokenizer fragment 太碎，无法确定它对应哪个代码实体。
- source 或 target 的上下文不够，无法判断作用域或语义关系。
- source 看起来既可能是合理依赖，也可能是 shortcut。
- 需要完整文件或更多上下文才能判断。

主实验计算准确率、召回率和 F1 时会先去掉 `U`。论文附录可以报告 `U` 的比例。

## 9. 置信度

`human_confidence` 填 1、2、3：

```text
1 = 低置信度
2 = 中等置信度
3 = 高置信度
```

如果 `human_label` 是 `U`，置信度可以留空或填 1。

## 10. 标注流程

建议按下面顺序标注：

1. 先看 `generated_first_error` 和 `reference_token`，理解模型第一个错在哪里。
2. 打开 `full_generated_code_path` 和 `full_reference_code_path`，从完整代码上下文理解错误代码和正确代码的差异。
3. 回到 TSV，看当前行的 `source_token`、`source_context` 和 `target_context`，定位这个 source 和 target。
4. 结合完整 generated/reference 代码，判断这个 source token 是否应该影响 target token 的生成。
5. 在 `human_label` 填 `S`、`N` 或 `U`。
6. 在 `human_confidence` 填 1、2、3。
7. 如果不是显然情况，在 `human_rationale` 写一句很短的原因。

标注者不需要知道 tree-sitter oracle 的判断，也不要打开 `human_annotation_top10_auto_key.tsv`。

## 11. 推荐标注规模

只有一个标注者时，推荐分三档：

- 试标：5 个样本 × top-10 = 50 个相关性。用于确认标注规则是否清楚。
- 主实验：30 个样本 × top-10 = 300 个相关性。建议作为论文主实验。
- 完整标注：42 个样本 × top-10 = 420 个相关性。时间足够时使用。

当前最推荐的是主实验设置，也就是已经生成好的 300 行。

## 12. 后续评估

标注完成后，把填好的 TSV 和自动 key 按 `item_id` 合并：

- 人工正例：`human_label == S`
- 自动正例：`auto_strict_label == spurious` 或 `auto_loose_label == spurious`
- 去掉 `human_label == U` 的行

报告下面几个指标：

- 准确率：自动标为虚假相关性的相关性中，人工也标为虚假相关性的比例。
- 召回率：人工标为虚假相关性的相关性中，自动也找出来的比例。
- F1：准确率和召回率的调和平均。
- 集合重合度：自动虚假相关性集合和人工虚假相关性集合的交并比。

主文建议报告 strict oracle 和 loose oracle 两组结果。strict oracle 更保守地定义程序支持关系，loose oracle 允许 same statement、control dependency 等更宽松的支持关系。

可以直接运行下面的脚本计算指标：

```bash
python -m src.evaluate_spurious_annotation \
  --annotation spurious_correlation_results/human_annotation/human_annotation_top10.tsv \
  --key spurious_correlation_results/human_annotation/human_annotation_top10_auto_key.tsv \
  --output-json spurious_correlation_results/human_annotation/human_annotation_eval.json
```

如果把盲标表发给外部标注者，建议只发 `human_annotation_top10.tsv` 和本说明文档，不要发 `human_annotation_top10_auto_key.tsv`。标注者填完后，把填好的 TSV 发回，再由实验负责人用本地 auto key 计算指标。
