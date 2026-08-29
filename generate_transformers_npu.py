#!/usr/bin/env python3
"""Generate AI4Go predictions with Transformers on multiple Ascend NPUs.

This is an inference-only launcher. It uses Accelerate big-model dispatch to put
whole decoder layers on multiple NPUs, so it must be started with plain Python,
not torchrun and not FSDP. For higher throughput, use generate_local.py in a
separate, version-compatible vLLM-Ascend environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from generate_local import (
    CHAT_STOP,
    Sample,
    batches,
    expand_inputs,
    load_completed_ids,
    load_samples,
    remove_leading_thinking,
    render_model_prompt,
)


LOGGER = logging.getLogger("ai4go.transformers_npu")


def load_model(args: argparse.Namespace):
    try:
        import torch
        import torch_npu  # noqa: F401 - registers the NPU backend
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(f"Missing NPU inference dependency: {exc}") from exc

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is False")
    device_count = torch.npu.device_count()
    if device_count < args.npu_count:
        raise RuntimeError(f"Requested {args.npu_count} NPUs, but torch sees {device_count}")

    max_memory = {index: args.max_memory_per_npu for index in range(args.npu_count)}
    LOGGER.info("Loading tokenizer from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info(
        "Loading model with device_map=%s, max_memory=%s",
        args.device_map,
        max_memory,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=args.device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    if args.adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required when --adapter is set") from exc
        LOGGER.info("Loading LoRA adapter from %s", args.adapter)
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)

    model.eval()
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        LOGGER.info("Resolved device map: %s", device_map)
        offloaded = {str(device) for device in device_map.values()} & {"cpu", "disk"}
        if offloaded and not args.allow_offload:
            raise RuntimeError(
                f"Model map contains offload targets {sorted(offloaded)}. "
                "Reduce --max-memory-per-npu only if intentional, or pass --allow-offload."
            )
    input_device = model.get_input_embeddings().weight.device
    if input_device.type != "npu":
        raise RuntimeError(f"Input embedding is on {input_device}, expected an NPU")
    return torch, tokenizer, model, input_device


def context_lengths(tokenizer, samples: Sequence[Sample], args: argparse.Namespace) -> None:
    available = args.max_model_len - args.max_new_tokens
    too_long: list[tuple[str, int]] = []
    for sample in samples:
        model_prompt = render_model_prompt(tokenizer, sample, enable_thinking=args.thinking)
        count = len(tokenizer.encode(model_prompt, add_special_tokens=False))
        if count > available:
            too_long.append((sample.task_id, count))
    if too_long:
        preview = ", ".join(f"{task_id}={count}" for task_id, count in too_long[:5])
        raise ValueError(
            f"{len(too_long)} prompts exceed the {available}-token input budget. "
            f"First cases: {preview}. Inputs are not truncated silently."
        )


def generate_batch(torch, tokenizer, model, input_device, batch: Sequence[Sample], args: argparse.Namespace):
    encoded = tokenizer(
        [render_model_prompt(tokenizer, sample, enable_thinking=args.thinking) for sample in batch],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]
    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    im_end_id = tokenizer.convert_tokens_to_ids(CHAT_STOP)
    eos_ids = [tokenizer.eos_token_id]
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "top_p": args.top_p,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_ids,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)

    rows = []
    for index, sample in enumerate(batch):
        generated = output_ids[index, prompt_width:]
        generated_list = generated.tolist()
        text = tokenizer.decode(generated, skip_special_tokens=True)
        prediction, removed_thinking = remove_leading_thinking(text)
        finish_reason = "stop" if generated_list and generated_list[-1] in eos_ids else "length"
        row = {
                "task_id": sample.task_id,
                "prompt": sample.prompt,
                "label": sample.label,
                "predict": prediction,
                "source_file": sample.source_file,
                "source_line": sample.source_line,
                "finish_reason": finish_reason,
                "prompt_tokens": int(prompt_lengths[index]),
                "generated_tokens": len(generated_list),
                "thinking_enabled": args.thinking,
            }
        if removed_thinking:
            row["predict_raw"] = text
            row["thinking_removed"] = True
        rows.append(row)
    return rows


def generate_file(torch, tokenizer, model, input_device, input_path: Path, output_path: Path, args) -> None:
    samples = load_samples(input_path, args.limit)
    completed = load_completed_ids(output_path) if args.resume else set()
    pending = [sample for sample in samples if sample.task_id not in completed]
    LOGGER.info(
        "%s: total=%d completed=%d pending=%d",
        input_path.name,
        len(samples),
        len(samples) - len(pending),
        len(pending),
    )
    if not pending:
        return
    context_lengths(tokenizer, pending, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        done = len(samples) - len(pending)
        for batch in batches(pending, args.batch_size):
            for row in generate_batch(torch, tokenizer, model, input_device, batch, args):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done += len(batch)
            LOGGER.info("%s: %d/%d", input_path.name, done, len(samples))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI4Go evaluation generation on Ascend NPUs")
    parser.add_argument("--input", nargs="+", required=True, help="Files, glob(s), or processed-parts directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", help="Local Qwen base-model directory (required unless --dry-run)")
    parser.add_argument("--adapter", help="Optional PEFT LoRA adapter directory")
    parser.add_argument("--npu-count", type=int, default=8)
    parser.add_argument("--device-map", choices=("auto", "balanced", "balanced_low_0", "sequential"), default="balanced_low_0")
    parser.add_argument("--max-memory-per-npu", default="56GiB")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-offload", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Qwen thinking mode (default: disabled for code-completion evaluation)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run and not args.model:
        parser.error("--model is required unless --dry-run is used")
    if args.batch_size <= 0 or args.npu_count <= 0:
        parser.error("--batch-size and --npu-count must be positive")
    if args.max_model_len <= args.max_new_tokens:
        parser.error("--max-model-len must be greater than --max-new-tokens")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    inputs = expand_inputs(args.input)
    output_dir = Path(args.output_dir).expanduser().resolve()
    total = 0
    for input_path in inputs:
        count = len(load_samples(input_path, args.limit))
        total += count
        LOGGER.info("Validated %s: %d samples", input_path, count)
    if args.dry_run:
        LOGGER.info("Dry run complete: %d files, %d samples", len(inputs), total)
        for input_path in inputs:
            print(f"{input_path} -> {output_dir / (input_path.stem + '.predictions.jsonl')}")
        return 0

    torch, tokenizer, model, input_device = load_model(args)
    for input_path in inputs:
        output_path = output_dir / f"{input_path.stem}.predictions.jsonl"
        if not args.resume and output_path.exists():
            output_path.unlink()
        generate_file(torch, tokenizer, model, input_device, input_path, output_path, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; completed batches are preserved")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.error("%s", exc)
        if "--verbose" in sys.argv:
            raise
        raise SystemExit(1)
