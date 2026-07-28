from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from glob import glob
from pathlib import Path

from src.spurious_correlation_analysis import _build_code_context


def _load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _feature_path(feature_dir: str, test_sample_index: int) -> str | None:
    direct = os.path.join(feature_dir, f"test{test_sample_index}_feature.json")
    if os.path.exists(direct):
        return direct
    matches = sorted(glob(os.path.join(feature_dir, f"*{test_sample_index}*_feature.json")))
    return matches[0] if matches else None


def _token_context(tokens: list[str], index: int | None, marker: str, window: int) -> str:
    if index is None or index < 0 or index >= len(tokens):
        return ""
    start = max(0, index - window)
    end = min(len(tokens), index + window + 1)
    pieces = []
    for pos in range(start, end):
        token = tokens[pos].replace("\n", "\\n").replace("\t", "\\t")
        if pos == index:
            pieces.append(f"[{marker}:{token}]")
        else:
            pieces.append(token)
    return " ".join(pieces)


def _single_line(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n").replace("\t", "\\t")


def _completion_context(tokens: list[str], prompt_len: int, window: int = 80) -> str:
    if prompt_len < 0 or prompt_len >= len(tokens):
        return ""
    text = "".join(tokens[prompt_len : prompt_len + window])
    return _single_line(text)


def _load_feature_payload(feature_dir: str, test_sample_index: int) -> dict:
    path = _feature_path(feature_dir, test_sample_index)
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_tokens(feature_dir: str, test_sample_index: int) -> list[str]:
    payload = _load_feature_payload(feature_dir, test_sample_index)
    baseline = payload.get("test_sample_baseline", {})
    return baseline.get("generated_full_tokens", []) or []


def _tokens_to_text(tokens: list[str], start_idx: int = 0, stop_at_im_end: bool = False) -> str:
    parts: list[str] = []
    for token in tokens[start_idx:]:
        if stop_at_im_end and token == "<|im_end|>":
            break
        parts.append(token)
    return "".join(parts)


def _relative_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _reconstructed_code(prompt_text: str, completion_text: str) -> str:
    context = _build_code_context(prompt_text, completion_text)
    if context is not None:
        return context.code
    return (
        "/* Reconstructed prompt/code context fallback. */\n\n"
        + prompt_text
        + "\n\n/* Completion */\n"
        + completion_text
    )


def _write_case_code_files(feature_dir: str, output_dir: str, test_sample_index: int) -> dict[str, str]:
    payload = _load_feature_payload(feature_dir, test_sample_index)
    baseline = payload.get("test_sample_baseline", {})
    prompt_len = int(baseline.get("prompt_len", 0))
    generated_tokens = baseline.get("generated_full_tokens", []) or []
    reference_tokens = baseline.get("ground_truth_full_tokens", []) or []

    prompt_text = "".join(generated_tokens[:prompt_len])
    generated_completion = _tokens_to_text(generated_tokens, prompt_len, stop_at_im_end=True)
    reference_completion = _tokens_to_text(reference_tokens, prompt_len, stop_at_im_end=True)

    case_dir = Path(output_dir) / "full_code" / f"test{test_sample_index}"
    case_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "full_prompt_path": case_dir / "prompt.txt",
        "full_generated_completion_path": case_dir / "generated_completion.txt",
        "full_reference_completion_path": case_dir / "reference_completion.txt",
        "full_generated_code_path": case_dir / "generated.go",
        "full_reference_code_path": case_dir / "reference.go",
    }
    paths["full_prompt_path"].write_text(prompt_text, encoding="utf-8")
    paths["full_generated_completion_path"].write_text(generated_completion, encoding="utf-8")
    paths["full_reference_completion_path"].write_text(reference_completion, encoding="utf-8")
    paths["full_generated_code_path"].write_text(
        _reconstructed_code(prompt_text, generated_completion),
        encoding="utf-8",
    )
    paths["full_reference_code_path"].write_text(
        _reconstructed_code(prompt_text, reference_completion),
        encoding="utf-8",
    )

    return {name: _relative_path(path) for name, path in paths.items()}


def _select_records(
    records: list[dict],
    max_cases: int | None,
    case_indices: set[int] | None,
    selection: str,
    seed: int,
) -> list[dict]:
    ok_records = [record for record in records if record.get("status") == "ok"]
    if case_indices is not None:
        ok_records = [
            record for record in ok_records
            if int(record.get("test_sample_index", -1)) in case_indices
        ]
    if selection == "random":
        ok_records = sorted(ok_records, key=lambda record: int(record.get("test_sample_index", -1)))
        rng = random.Random(seed)
        rng.shuffle(ok_records)
    elif selection == "high_spurious":
        ok_records = sorted(
            ok_records,
            key=lambda record: (
                float(record.get("metrics", {}).get("strict_spurious@10", -1)),
                float(record.get("metrics", {}).get("strict_spurious_mass@10", -1)),
                int(record.get("test_sample_index", -1)),
            ),
            reverse=True,
        )
    elif selection == "test_index":
        ok_records = sorted(ok_records, key=lambda record: int(record.get("test_sample_index", -1)))
    else:
        raise ValueError(f"Unknown selection strategy: {selection}")
    if max_cases is not None:
        ok_records = ok_records[: max(0, int(max_cases))]
    return ok_records


def _auto_label(source_row: dict, mode: str) -> str:
    if not bool(source_row.get("mapped")):
        return "spurious"
    supported_key = f"{mode}_supported"
    return "supported" if bool(source_row.get(supported_key)) else "spurious"


def export_annotation_sheets(
    analysis_jsonl: str,
    feature_dir: str,
    output_dir: str,
    top_k: int,
    max_cases: int | None,
    case_indices: set[int] | None,
    context_window: int,
    selection: str,
    seed: int,
) -> tuple[str, str]:
    records = _select_records(_load_jsonl(analysis_jsonl), max_cases, case_indices, selection, seed)
    os.makedirs(output_dir, exist_ok=True)
    full_code_dir = Path(output_dir) / "full_code"
    if full_code_dir.exists():
        shutil.rmtree(full_code_dir)

    annotation_path = os.path.join(output_dir, f"human_annotation_top{top_k}.tsv")
    key_path = os.path.join(output_dir, f"human_annotation_top{top_k}_auto_key.tsv")

    annotation_fields = [
        "item_id",
        "case_order",
        "source_order",
        "test_sample_index",
        "task_id",
        "target_token_index",
        "target_token",
        "generated_first_error",
        "reference_token",
        "source_rank",
        "source_method_rank",
        "source_ranking",
        "source_token_index",
        "source_token",
        "alti_saliency",
        "source_context",
        "target_context",
        "generated_completion_preview",
        "reference_completion_preview",
        "full_generated_code_path",
        "full_reference_code_path",
        "full_generated_completion_path",
        "full_reference_completion_path",
        "full_prompt_path",
        "human_label",
        "human_confidence",
        "human_rationale",
    ]
    key_fields = [
        "item_id",
        "test_sample_index",
        "target_token_index",
        "source_rank",
        "auto_strict_label",
        "auto_loose_label",
        "mapped",
        "oracle_reason",
        "source_node_type",
        "target_node_type",
        "source_node_text",
        "target_node_text",
    ]

    with open(annotation_path, "w", encoding="utf-8", newline="") as anno_handle, open(
        key_path, "w", encoding="utf-8", newline=""
    ) as key_handle:
        annotation_writer = csv.DictWriter(anno_handle, fieldnames=annotation_fields, delimiter="\t")
        key_writer = csv.DictWriter(key_handle, fieldnames=key_fields, delimiter="\t")
        annotation_writer.writeheader()
        key_writer.writeheader()

        for case_order, record in enumerate(records, start=1):
            test_index = int(record["test_sample_index"])
            tokens = _load_tokens(feature_dir, test_index)
            prompt_len = int(record.get("prompt_len", 0))
            target_index = int(record.get("target_token_index", -1))
            generated_preview = _single_line(
                record.get("generated_completion_preview") or _completion_context(tokens, prompt_len)
            )
            reference_preview = _single_line(record.get("reference_completion_preview") or "")
            target_context = _token_context(tokens, target_index, "TARGET", context_window)
            code_paths = _write_case_code_files(feature_dir, output_dir, test_index)

            for source_order, source_row in enumerate(record.get("top_sources", [])[:top_k], start=1):
                source_index = int(source_row.get("source_token_index", -1))
                item_id = f"test{test_index}_t{target_index}_s{source_order:02d}"
                annotation_writer.writerow({
                    "item_id": item_id,
                    "case_order": case_order,
                    "source_order": source_order,
                    "test_sample_index": test_index,
                    "task_id": record.get("task_id", ""),
                    "target_token_index": target_index,
                    "target_token": _single_line(record.get("target_token", "")),
                    "generated_first_error": _single_line(record.get("generated_lex", "")),
                    "reference_token": _single_line(record.get("reference_lex", "")),
                    "source_rank": source_row.get("rank", source_order),
                    "source_method_rank": source_row.get("method_rank", ""),
                    "source_ranking": record.get("source_ranking", ""),
                    "source_token_index": source_index,
                    "source_token": _single_line(source_row.get("source_token", "")),
                    "alti_saliency": source_row.get("alti_saliency", ""),
                    "source_context": _token_context(tokens, source_index, "SOURCE", context_window),
                    "target_context": target_context,
                    "generated_completion_preview": generated_preview,
                    "reference_completion_preview": reference_preview,
                    "full_generated_code_path": code_paths.get("full_generated_code_path", ""),
                    "full_reference_code_path": code_paths.get("full_reference_code_path", ""),
                    "full_generated_completion_path": code_paths.get("full_generated_completion_path", ""),
                    "full_reference_completion_path": code_paths.get("full_reference_completion_path", ""),
                    "full_prompt_path": code_paths.get("full_prompt_path", ""),
                    "human_label": "",
                    "human_confidence": "",
                    "human_rationale": "",
                })
                key_writer.writerow({
                    "item_id": item_id,
                    "test_sample_index": test_index,
                    "target_token_index": target_index,
                    "source_rank": source_row.get("rank", source_order),
                    "auto_strict_label": _auto_label(source_row, "strict"),
                    "auto_loose_label": _auto_label(source_row, "loose"),
                    "mapped": source_row.get("mapped", ""),
                    "oracle_reason": _single_line(source_row.get("reason", "")),
                    "source_node_type": _single_line(source_row.get("source_node_type", "")),
                    "target_node_type": _single_line(source_row.get("target_node_type", "")),
                    "source_node_text": _single_line(source_row.get("source_node_text", "")),
                    "target_node_text": _single_line(source_row.get("target_node_text", "")),
                })

    return annotation_path, key_path


def parse_case_indices(raw: str) -> set[int] | None:
    raw = raw.strip()
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a blind human-annotation TSV from spurious-correlation analysis results."
    )
    parser.add_argument("--analysis-jsonl", default="spurious_correlation_results/per_sample.jsonl")
    parser.add_argument("--feature-dir", default="attribution_results_feature_alti_saliency_full100/feature")
    parser.add_argument("--output-dir", default="spurious_correlation_results/human_annotation")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--case-indices", default="")
    parser.add_argument("--context-window", type=int, default=8)
    parser.add_argument(
        "--selection",
        choices=["random", "test_index", "high_spurious"],
        default="random",
        help="Case selection strategy. Use random for human-study evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    annotation_path, key_path = export_annotation_sheets(
        analysis_jsonl=args.analysis_jsonl,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        top_k=max(1, int(args.top_k)),
        max_cases=args.max_cases,
        case_indices=parse_case_indices(args.case_indices),
        context_window=max(0, int(args.context_window)),
        selection=args.selection,
        seed=int(args.seed),
    )
    print(f"Wrote annotation sheet: {Path(annotation_path).resolve()}")
    print(f"Wrote automatic key:    {Path(key_path).resolve()}")


if __name__ == "__main__":
    main()
