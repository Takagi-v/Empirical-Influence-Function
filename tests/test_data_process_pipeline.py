from __future__ import annotations

import json
from pathlib import Path

from src.data_process.adapters import adapt_row
from src.data_process.pipeline import PipelineConfig, run_pipeline


def test_adapt_huawei_pre_suf_mid_row_to_canonical_go():
    row = {
        "prompt": "This is a go programming task.\n```go\n<PRE> func f() {\n    x := 1\n<SUF>     println(x)\n}\n\n<MID>\n```",
        "response": "    x += 1\n",
        "task_id": "task-1",
    }

    sample, reason = adapt_row(row, 0, language="auto")

    assert reason is None
    assert sample is not None
    assert sample["language"] == "Go"
    assert sample["prefix"].endswith("x := 1\n")
    assert sample["target"] == "    x += 1\n"
    assert sample["suffix"].lstrip().startswith("println(x)")
    assert sample["full_code"] == sample["prefix"] + sample["target"] + sample["suffix"]
    assert sample["messages"][1]["content"].count("[MASK]") == 1


def test_adapt_canonical_java_row_preserves_language_and_renders_chatml():
    row = {
        "uid": "java-1",
        "language": "java",
        "prefix": "class A { void f(){ int x = 1; ",
        "target": "x += 1;",
        "suffix": " } }",
    }

    sample, reason = adapt_row(row, 0, language="auto")

    assert reason is None
    assert sample is not None
    assert sample["uid"] == "java-1"
    assert sample["language"] == "Java"
    assert sample["messages"][0]["content"] == "You are a Java code completion assistant."
    assert "missing Java code" in sample["messages"][1]["content"]
    assert sample["messages"][2]["content"] == "x += 1;"


def test_pipeline_skip_annotation_writes_canonical_chatml_and_report(tmp_path: Path):
    input_path = tmp_path / "raw.jsonl"
    input_path.write_text(json.dumps({
        "prompt": "This is a java programming task. <PRE> class A { void f(){ int x = 1; <SUF> } } <MID>",
        "response": "x += 1;",
        "task_id": "java-task",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    result = run_pipeline(PipelineConfig(
        input_path=input_path,
        output_dir=out_dir,
        model_path="unused",
        annotate_model="unused",
        skip_annotation=True,
    ))

    assert result.canonical_path.exists()
    assert result.chatml_path.exists()
    assert result.report_path.exists()
    canonical = [json.loads(line) for line in result.canonical_path.read_text(encoding="utf-8").splitlines()]
    chatml = [json.loads(line) for line in result.chatml_path.read_text(encoding="utf-8").splitlines()]
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert canonical[0]["language"] == "Java"
    assert chatml[0]["messages"][1]["content"].count("[MASK]") == 1
    assert report["accepted"] == 1
    assert report["annotated"] == 0
