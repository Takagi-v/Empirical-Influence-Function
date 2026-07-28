from __future__ import annotations

import argparse
import os
import random
import traceback
from functools import partial

import torch
from accelerate import Accelerator
from transformers import DataCollatorForSeq2Seq, set_seed

from src.NIF import CustomCollator
from src.attribution_batch_report import build_report, _write_json as write_report_json, _write_markdown
from src.attribution_evaluation import (
    DEFAULT_FEATURE_EFFECT_THRESHOLDS,
    DEFAULT_FEATURE_SALIENCY_MASS_THRESHOLDS,
    _decode_token,
    _tensor_batch_to_device,
    _valid_target_positions,
    _write_json,
    build_test_batch,
    evaluate_feature_attribution,
    parse_float_tuple,
    parse_int_tuple,
    parse_saliency_mass_thresholds,
)
from src.intervention_experiment import SEED, load_model_and_tokenizer, load_samples
from src.process_data import process_func_chatml


def _parse_indices(raw: str | None, start_idx: int, end_idx: int | None, total: int) -> list[int]:
    if raw:
        chunks = [x for x in raw.replace(",", " ").split() if x]
        indices = [int(x) for x in chunks]
    else:
        stop = total - 1 if end_idx is None else int(end_idx)
        indices = list(range(int(start_idx), stop + 1))
    for idx in indices:
        if idx < 0 or idx >= total:
            raise ValueError(f"test index out of range: {idx} (valid: 0..{total - 1})")
    return indices


def _record_status(path: str, idx: int, task_id: str, status: str, output_file: str, log_file: str = "") -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{idx}\t{task_id}\tfeature\t{status}\t0\t{output_file}\t{log_file}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature attribution evaluator that reuses one loaded model across many test samples."
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--test-data", type=str, default="sft_test.jsonl")
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=20)
    parser.add_argument("--generation-limit", type=int, default=128)
    parser.add_argument("--use-ground-truth-response", action="store_true")
    parser.add_argument("--max-gpu-memory", type=str, default=None)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--top-k-prompt-tokens", type=int, default=20)
    parser.add_argument("--feature-k-values", type=str, default="5,10,20")
    parser.add_argument("--feature-saliency-mass-thresholds", type=str, default="")
    parser.add_argument(
        "--feature-perturb-mode",
        choices=["replace", "random_replace", "delete", "zero_attention"],
        default="replace",
    )
    parser.add_argument("--feature-random-trials", type=int, default=5)
    parser.add_argument("--feature-perturb-batch-size", type=int, default=1)
    parser.add_argument("--replacement-token-id", type=int, default=None)
    parser.add_argument("--max-feature-sources", type=int, default=None)
    parser.add_argument(
        "--feature-evaluation-mode",
        choices=["effectiveness", "full", "saliency_only"],
        default="effectiveness",
    )
    parser.add_argument("--feature-max-prefix-len", type=int, default=2048)
    parser.add_argument("--feature-effect-thresholds", type=str, default="0.1,0.2,0.5")
    parser.add_argument("--feature-effect-metric", choices=["logprob_drop", "prob_drop"], default="logprob_drop")
    parser.add_argument(
        "--feature-ranking-mode",
        choices=["alti", "signed", "signed_clip"],
        default="alti",
    )
    parser.add_argument(
        "--feature-direction-mode",
        choices=["hidden", "alti_last"],
        default="hidden",
    )
    parser.add_argument(
        "--feature-direction-score",
        choices=["cosine", "projection"],
        default="cosine",
    )
    parser.add_argument("--feature-source-unit", choices=["token", "span"], default="token")
    parser.add_argument("--feature-span-score", choices=["sum", "max", "mean"], default="sum")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise RuntimeError("Feature batch evaluator expects a single Accelerator process.")

    output_dir = os.path.abspath(args.output_dir)
    feature_dir = os.path.join(output_dir, "feature")
    report_dir = os.path.join(output_dir, "reports")
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    status_path = os.path.join(report_dir, "batch_status.tsv")
    with open(status_path, "w", encoding="utf-8") as f:
        f.write("sample_index\ttask_id\tstage\tstatus\texit_code\toutput_file\tlog_file\n")

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        attn_implementation=args.attn_implementation,
        max_gpu_memory=args.max_gpu_memory,
    )
    convert_to_chatml = partial(process_func_chatml, tokenizer=tokenizer)
    test_samples = load_samples(args.test_data)
    test_indices = _parse_indices(args.indices, args.start_idx, args.end_idx, len(test_samples))

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    replacement_token_id = args.replacement_token_id
    if replacement_token_id is None:
        replacement_token_id = tokenizer.pad_token_id
    if replacement_token_id is None:
        replacement_token_id = tokenizer.eos_token_id
    if replacement_token_id is None:
        raise RuntimeError("No replacement token id available; pass --replacement-token-id.")

    feature_k_values = parse_int_tuple(args.feature_k_values)
    feature_effect_thresholds = (
        parse_float_tuple(args.feature_effect_thresholds)
        or DEFAULT_FEATURE_EFFECT_THRESHOLDS
    )
    feature_saliency_mass_thresholds = (
        parse_saliency_mass_thresholds(args.feature_saliency_mass_thresholds)
        or DEFAULT_FEATURE_SALIENCY_MASS_THRESHOLDS
    )

    done = skipped = failed = 0
    for test_index in test_indices:
        test_sample = test_samples[int(test_index)]
        task_id = test_sample.get("task_id") or f"test{test_index}"
        output_file = os.path.join(feature_dir, f"{task_id}_feature.json")
        print("\n-----------------------------------------------------------------", flush=True)
        print(f"Sample {test_index}: {task_id}", flush=True)
        print("-----------------------------------------------------------------", flush=True)
        if os.path.exists(output_file):
            print(f"[feature-batch] SKIP existing {output_file}", flush=True)
            skipped += 1
            _record_status(status_path, int(test_index), task_id, "skipped", output_file)
            continue

        try:
            print("[feature-batch] building test batch...", flush=True)
            test_batch, prompt_len, test_meta = build_test_batch(
                model,
                tokenizer,
                test_sample,
                convert_to_chatml,
                base_collator,
                accelerator,
                use_generated_response=not args.use_ground_truth_response,
                generation_limit=args.generation_limit,
            )
            test_batch = _tensor_batch_to_device(test_batch, accelerator.device)
            target_positions = _valid_target_positions(
                tokenizer,
                test_batch["input_ids"][0],
                prompt_len,
                int(args.max_output_tokens),
            )
            if not target_positions:
                raise RuntimeError("No target positions selected for evaluation.")

            feature_rows = evaluate_feature_attribution(
                model,
                tokenizer,
                test_batch,
                target_positions,
                device=accelerator.device,
                top_k_prompt_tokens=max(1, int(args.top_k_prompt_tokens)),
                k_values=feature_k_values,
                perturb_mode=args.feature_perturb_mode,
                replacement_token_id=int(replacement_token_id),
                max_feature_sources=args.max_feature_sources,
                evaluation_mode=args.feature_evaluation_mode,
                effect_thresholds=feature_effect_thresholds,
                effect_metric=args.feature_effect_metric,
                random_trials=max(1, int(args.feature_random_trials)),
                vocab_size=len(tokenizer),
                random_excluded_token_ids=set(
                    int(x) for x in (getattr(tokenizer, "all_special_ids", []) or [])
                ),
                group_only=False,
                perturb_batch_size=max(1, int(args.feature_perturb_batch_size)),
                ranking_mode=args.feature_ranking_mode,
                direction_mode=args.feature_direction_mode,
                direction_score=args.feature_direction_score,
                source_unit=args.feature_source_unit,
                span_score=args.feature_span_score,
                saliency_mass_thresholds=feature_saliency_mass_thresholds,
                max_prefix_len=(
                    None if args.feature_max_prefix_len <= 0 else int(args.feature_max_prefix_len)
                ),
            )
            report = {
                "experiment_meta": {
                    "test_sample_index": int(test_index),
                    "task_id": task_id,
                    "target_token_indices": [int(x) for x in target_positions],
                    "target_tokens": [
                        _decode_token(tokenizer, int(test_batch["input_ids"][0, t].item()))
                        for t in target_positions
                    ],
                    "prompt_len": int(prompt_len),
                    "train_size": None,
                    "stage": "feature_attribution",
                    "is_checkpoint": True,
                    "config": {
                        "feature_k_values": feature_k_values,
                        "feature_saliency_mass_thresholds": feature_saliency_mass_thresholds,
                        "feature_perturb_mode": args.feature_perturb_mode,
                        "feature_random_trials": max(1, int(args.feature_random_trials)),
                        "feature_perturb_batch_size": max(1, int(args.feature_perturb_batch_size)),
                        "feature_group_only": False,
                        "max_feature_sources": args.max_feature_sources,
                        "feature_evaluation_mode": args.feature_evaluation_mode,
                        "feature_effect_thresholds": feature_effect_thresholds,
                        "feature_effect_metric": args.feature_effect_metric,
                        "feature_ranking_mode": args.feature_ranking_mode,
                        "feature_direction_mode": args.feature_direction_mode,
                        "feature_direction_score": args.feature_direction_score,
                        "feature_source_unit": args.feature_source_unit,
                        "feature_span_score": args.feature_span_score,
                        "feature_max_prefix_len": (
                            None if args.feature_max_prefix_len <= 0 else int(args.feature_max_prefix_len)
                        ),
                        "max_output_tokens": int(args.max_output_tokens),
                        "generation_limit": int(args.generation_limit),
                    },
                },
                "test_sample_baseline": test_meta,
                "feature_attribution": feature_rows,
            }
            _write_json(output_file, report)
            print(f"[feature-batch] wrote {output_file}", flush=True)
            done += 1
            _record_status(status_path, int(test_index), task_id, "done", output_file)
        except ValueError as exc:
            if "No labeled response tokens" not in str(exc):
                failed += 1
                _record_status(status_path, int(test_index), task_id, "failed", output_file)
                print(f"[feature-batch] FAIL {task_id}: {exc}", flush=True)
                traceback.print_exc()
                continue
            skip_reason = "skipped_no_labeled_response_after_truncation"
            skipped_report = {
                "status": "skipped",
                "skip_reason": skip_reason,
                "experiment_meta": {
                    "test_sample_index": int(test_index),
                    "task_id": task_id,
                    "target_token_indices": [],
                    "target_tokens": [],
                    "prompt_len": None,
                    "train_size": None,
                    "stage": "feature_attribution",
                    "is_checkpoint": True,
                    "config": {
                        "feature_evaluation_mode": args.feature_evaluation_mode,
                        "feature_max_prefix_len": (
                            None if args.feature_max_prefix_len <= 0 else int(args.feature_max_prefix_len)
                        ),
                        "max_output_tokens": int(args.max_output_tokens),
                        "generation_limit": int(args.generation_limit),
                    },
                },
                "test_sample_baseline": {
                    "status": "skipped",
                    "skip_reason": skip_reason,
                    "error": str(exc),
                    "response_source": "unavailable",
                },
                "feature_attribution": [],
            }
            _write_json(output_file, skipped_report)
            print(f"[feature-batch] skipped {task_id}: {skip_reason}", flush=True)
            skipped += 1
            _record_status(status_path, int(test_index), task_id, "skipped", output_file)
        except Exception as exc:
            failed += 1
            _record_status(status_path, int(test_index), task_id, "failed", output_file)
            print(f"[feature-batch] FAIL {task_id}: {exc}", flush=True)
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    report_json = os.path.join(report_dir, "attribution_batch_report.json")
    report_md = os.path.join(report_dir, "attribution_batch_report.md")
    report = build_report(output_dir, status_path)
    write_report_json(report_json, report)
    _write_markdown(report_md, report)
    print("\n=================================================================", flush=True)
    print("  Feature batch finished", flush=True)
    print(f"  Done   : {done}", flush=True)
    print(f"  Skipped: {skipped}", flush=True)
    print(f"  Failed : {failed}", flush=True)
    print(f"  Results: {output_dir}", flush=True)
    print(f"  Report : {report_json}", flush=True)
    print("=================================================================", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
