# 数据处理脚本交接

**处理入口**

```
bash scripts/data_process/run_huawei_batch_annotation.sh \
  --output-root data/huawei_data/processed_full_clean \
  --model-path "$MODEL_PATH" \
  --annotate-model "$ANNOTATE_MODEL" \
  --workers 8 \
  "$TRAIN_DATA_1" "$TRAIN_DATA_2" "$TRAIN_DATA_3" "$TRAIN_DATA_4"
```

**输入**

输入是一个或多个 `.jsonl` 文件，每行一个 JSON object。推荐 raw 格式如下：

```
{
  "prompt": "This is a go programming task...\n```go\n<PRE> func f() {\n\tif x {\n<SUF>\t}\n}\n<MID>\n```\n### Response:",
  "response": "\t\treturn nil\n",
  "task_id": "Coding-CC-L2-Go_xxx"
}
```

字段要求：


| 字段       | 类型   | 必需 | 含义                                             |
| ---------- | ------ | ---- | ------------------------------------------------ |
| `prompt`   | string | 是   | 包含 FIM 上下文，必须有`<PRE>`、`<SUF>`、`<MID>` |
| `response` | string | 是   | `<MID>`处的真实补全代码                          |
| `task_id`  | string | 建议 | 样本唯一 ID，用于追踪、去重、排查问题            |

也兼容已整理过的格式：

```
{
  "prefix": "func f() {\n",
  "target": "\treturn nil\n",
  "suffix": "}\n",
  "task_id": "sample_001",
  "language": "Go"
}
```

**输出**

每个输入文件会生成一个独立输出目录。比如输入文件名是：

```
cloud_core_test_25.JunJunly_GoOnly_length_filter.jsonl
```

输出目录大致是：

```
data/huawei_data/processed_full_clean/cloud_core_test_25_JunJunly_GoOnly_length_filter/
```

里面主要有：


| 文件                           | 格式  | 用途                                                 |
| ------------------------------ | ----- | ---------------------------------------------------- |
| `<run>_canonical.jsonl`        | JSONL | 统一后的 FIM 样本，含`prefix/target/suffix/messages` |
| `<run>_chatml.jsonl`           | JSONL | ChatML 训练/标注输入格式                             |
| `<run>_compact.jsonl`          | JSONL | 最终训练用紧凑格式，含 token id 和标注边             |
| `<run>_report.json`            | JSON  | 处理统计：accepted/rejected/annotated 等             |
| `<run>_annotation_cache.jsonl` | JSONL | 标注缓存，断点续跑用                                 |
| `<run>_rejects.jsonl`          | JSONL | 被过滤样本及原因，如有                               |
| `<run>_failures.json`          | JSON  | 标注失败样本及错误，如有                             |

**输出示例 1：canonical/chatml**

```
{
  "uid": "fim_go_abcd1234",
  "raw_id": "Coding-CC-L2-Go_xxx",
  "source_dataset": "raw_fim",
  "split": "train",
  "language": "Go",
  "task_type": "go_fim_completion",
  "prefix": "func f() {\n\tif x {\n",
  "target": "\t\treturn nil\n",
  "suffix": "\t}\n}\n",
  "full_code": "func f() {\n\tif x {\n\t\treturn nil\n\t}\n}\n",
  "messages": [
    {
      "role": "system",
      "content": "You are a Go code completion assistant."
    },
    {
      "role": "user",
      "content": "Fill the missing part in the Go code. Return only the missing Go code, without Markdown fences or explanation.\n\n* Incomplete Code:\nfunc f() {\n\tif x {\n[MASK]\t}\n}\n"
    },
    {
      "role": "assistant",
      "content": "\t\treturn nil\n"
    }
  ],
  "only_last_turn_loss": true
}
```

**输出示例 2：compact**

```
{
  "input_ids": [151644, 8948, 198, 2610, 525, 264],
  "label": [-100, -100, -100, -100, 470, 2735],
  "length": 512,
  "attention_edges": [
    {
      "src": 120,
      "dst": 188,
      "subtype": "defuse"
    },
    {
      "src": 145,
      "dst": 190,
      "subtype": "call"
    }
  ],
  "uid": "fim_go_abcd1234",
  "language": "Go",
  "raw_id": "Coding-CC-L2-Go_xxx"
}
```

`compact` 字段含义：


| 字段                  | 含义                                                                   |
| --------------------- | ---------------------------------------------------------------------- |
| `input_ids`           | tokenizer 后的完整 ChatML token 序列                                   |
| `label`               | 训练标签；`-100`表示 prompt 部分不算 loss，非`-100`是 completion token |
| `attention_edges`     | token-level 标注边，`src -> dst`表示 src token 是预测 dst token 的线索 |
| `length`              | token 序列长度                                                         |
| `uid/raw_id/language` | 样本追踪信息                                                           |

**输出示例 3：report**

```
{
  "input_path": "/home/model_project/Open_CC_SFT_Eval/train/cloud_core_test_25.JunJunly_GoOnly_length_filter.jsonl",
  "run_name": "cloud_core_test_25_JunJunly_GoOnly_length_filter",
  "seen": 199923,
  "accepted": 180000,
  "rejected": 19923,
  "annotated": 180000,
  "skip_annotation": false,
  "use_llm": true,
  "num_workers": 8,
  "reject_reasons": {
    "target_too_many_nonempty_lines": 10000,
    "target_too_many_rough_tokens": 5000,
    "missing_prefix_suffix_or_target": 4923
  },
  "language_counts": {
    "Go": 180000
  }
}
```

先验收可以跑小样本：

```
bash scripts/data_process/run_huawei_batch_annotation.sh \
  --output-root data/huawei_data/processed_30_clean \
  --model-path "$MODEL_PATH" \
  --annotate-model "$ANNOTATE_MODEL" \
  --workers 2 \
  --max-accepted-rows 30 \
  "$TRAIN_DATA_1"
```
