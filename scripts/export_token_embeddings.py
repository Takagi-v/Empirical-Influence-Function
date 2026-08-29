#!/usr/bin/env python3
"""One script: export embedding (.pt) + decoded text (.txt) per sample.

Loads base CausalLM + optional LoRA adapter, runs a forward, and for every
sample writes BOTH:

  out_dir/embeddings/000000.pt   # hidden [T,H], input_ids, labels, surfaces, text
  out_dir/embeddings/000000.txt  # same sequence decoded to text (tokenizer only)

Modes
-----
``--mode train`` (default)
    Compact annotated JSONL with ``input_ids`` (+ optional ``label``/``labels``).

``--mode test``
    Eval predictions JSONL from ``generate_transformers_npu.py``::
    ``{prompt, label, predict, task_id, ...}``.

    Matches eval: wrap ``prompt`` with ChatML (``apply_chat_template`` +
    ``add_generation_prompt``), then append tokenized ``predict`` (+ EOS).
    Labels ignore the ChatML prefix; completion span = predict. Gold ``label``
    is stored in the ``.pt`` for analysis but is NOT fed into the forward.

Adapter is used only for ``hidden``; ``.txt`` decode uses the base tokenizer.

Example (train / compact)::

  CUDA_VISIBLE_DEVICES=1 python scripts/export_token_embeddings.py \\
    --mode train \\
    --model_name_or_path /mnt/md124/jiaxin/models/Qwen3-8B \\
    --adapter_path ./outputs/.../checkpoint-500 \\
    --data_path .../smoke_train_data_oversample_llm_mid_edges.jsonl \\
    --output_dir ./outputs/emb_train \\
    --max_len 3072

Example (test / ChatML prompt + model predict)::

  CUDA_VISIBLE_DEVICES=1 python scripts/export_token_embeddings.py \\
    --mode test \\
    --model_name_or_path /mnt/md124/jiaxin/models/Qwen3-8B \\
    --adapter_path .../checkpoint-500 \\
    --data_path /mnt/md124/jiaxin/test.jsonl \\
    --output_dir /mnt/md124/jiaxin/outputs/test_embedding \\
    --max_len 3072
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("export_token_embeddings")

IGNORE_INDEX = -100


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_adapter_dir(path: Path) -> Path:
    """Accept output_dir or a checkpoint-* subdir that contains adapter_config.json."""
    if (path / "adapter_config.json").is_file():
        return path
    ckpts = sorted(path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    for c in reversed(ckpts):
        if (c / "adapter_config.json").is_file():
            logger.info("Using adapter checkpoint: %s", c)
            return c
    if any(path.glob("adapter_*.safetensors")) or (path / "adapter_model.bin").is_file():
        return path
    raise FileNotFoundError(
        f"No LoRA adapter found under {path} (expected adapter_config.json "
        f"or checkpoint-*/adapter_config.json)"
    )


def build_model(
    model_name_or_path: str,
    adapter_path: str | None,
    dtype: torch.dtype,
    device: str,
):
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    if adapter_path:
        from peft import PeftModel

        adapter_dir = resolve_adapter_dir(Path(adapter_path))
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        logger.info("Loaded LoRA adapter from %s", adapter_dir)
    else:
        logger.info("No adapter: exporting base-model embeddings")

    model.eval()
    model.to(device)
    return tok, model


def resolve_eos_id(tok) -> int:
    eos_id = tok.eos_token_id
    if eos_id is None:
        eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(eos_id, int) or eos_id < 0:
        raise RuntimeError("tokenizer has no usable eos_token_id")
    return int(eos_id)


def render_chatml_prompt(tok, prompt: str, *, enable_thinking: bool = False) -> str:
    """Same idea as eval ``render_model_prompt``: ChatML + assistant generation head.

    Prefer ``tokenizer.apply_chat_template`` (Qwen3); fall back to a manual
    ``<|im_start|>…`` wrapper if the template API is unavailable.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Older transformers: no enable_thinking kwarg
            try:
                return tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        except Exception:
            pass
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def row_to_ids_labels_train(row: dict[str, Any]) -> tuple[list[int], list[int]] | None:
    """Compact annotated row → (input_ids, labels)."""
    ids = row.get("input_ids")
    if not ids:
        return None
    ids = [int(x) for x in ids]
    labels = row.get("label") or row.get("labels")
    labels = [int(x) for x in labels] if labels is not None else [IGNORE_INDEX] * len(ids)
    if len(labels) != len(ids):
        return None
    return ids, labels


def row_to_ids_labels_test(
    row: dict[str, Any],
    tok,
    *,
    max_len: int,
    enable_thinking: bool = False,
    append_eos: bool = True,
) -> tuple[list[int], list[int], dict[str, Any]] | None:
    """Eval row → ChatML(prompt) + predict (+ EOS).

    Sequence matches generation: model saw ChatML prompt, then produced ``predict``.
    Gold ``label`` is returned in ``meta`` only (not in the forward string).
    """
    prompt = row.get("prompt", "") or ""
    predict = row.get("predict", "")
    if predict is None:
        predict = ""
    if not prompt:
        return None
    if not str(predict):
        logger.warning("empty predict; will embed ChatML prompt only")

    chatml = render_chatml_prompt(tok, prompt, enable_thinking=enable_thinking)
    eos_id = resolve_eos_id(tok)
    prompt_ids = tok(chatml, add_special_tokens=False).input_ids
    pred_ids = tok(str(predict), add_special_tokens=False).input_ids if str(predict) else []

    n_eos = 1 if (append_eos and pred_ids) else 0
    if pred_ids:
        budget = max_len - len(pred_ids) - n_eos
        if budget <= 0:
            keep = max_len - n_eos
            if keep <= 0:
                return None
            pred_ids = pred_ids[-keep:]
            prompt_ids = []
        elif len(prompt_ids) > budget:
            prompt_ids = prompt_ids[-budget:]
        n_prompt = len(prompt_ids)
        input_ids = prompt_ids + pred_ids + ([eos_id] if n_eos else [])
        labels = (
            [IGNORE_INDEX] * n_prompt
            + list(pred_ids)
            + ([eos_id] if n_eos else [])
        )
    else:
        if len(prompt_ids) > max_len:
            prompt_ids = prompt_ids[-max_len:]
        input_ids = prompt_ids
        labels = [IGNORE_INDEX] * len(input_ids)
        n_prompt = len(input_ids)

    meta = {
        "chatml_prompt": chatml,
        "n_prompt_tokens": n_prompt,
        "n_predict_tokens": len(pred_ids),
        "predict": str(predict),
        "label": row.get("label", ""),
        "prompt_raw": prompt,
    }
    return input_ids, labels, meta


@torch.inference_mode()
def embed_one(
    model,
    input_ids: list[int],
    *,
    device: str,
    max_len: int,
    layer: int,
) -> torch.Tensor:
    """Return float32 CPU tensor [T, H] for the chosen hidden layer."""
    ids = input_ids[:max_len]
    t = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(input_ids=t, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[layer][0].detach().float().cpu()
    return h


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--adapter_path", default="", help="LoRA output_dir or checkpoint-* (empty = base only)")
    p.add_argument(
        "--data_path",
        required=True,
        help="train: compact jsonl with input_ids; test: {prompt,label,predict,...} predictions jsonl",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--mode",
        choices=("train", "test"),
        default="train",
        help="train=pre-encoded input_ids; test=ChatML(prompt)+predict (eval predictions)",
    )
    p.add_argument("--max_len", type=int, default=3072)
    p.add_argument(
        "--thinking",
        action="store_true",
        help="[test] enable_thinking=True in apply_chat_template (default off, matches eval)",
    )
    p.add_argument(
        "--no_append_eos",
        action="store_true",
        help="[test] do not append EOS after predict",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Hidden-state index: -1 last layer (contextual), 0 token embedding table output",
    )
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_samples", type=int, default=0, help="0 = all")
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    out_dir = Path(args.output_dir)
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    rows = load_jsonl(Path(args.data_path))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    logger.info("Loaded %d rows from %s (mode=%s)", len(rows), args.data_path, args.mode)

    tok, model = build_model(
        args.model_name_or_path,
        args.adapter_path or None,
        dtype=dtype,
        device=args.device,
    )

    n_ok = 0
    n_skip = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        for i, row in enumerate(rows):
            extra: dict[str, Any] = {}
            if args.mode == "train":
                built = row_to_ids_labels_train(row)
                if built is None:
                    logger.warning("skip row %d: need input_ids (+ matching labels)", i)
                    n_skip += 1
                    continue
                ids, labels = built
            else:
                built = row_to_ids_labels_test(
                    row,
                    tok,
                    max_len=args.max_len,
                    enable_thinking=args.thinking,
                    append_eos=not args.no_append_eos,
                )
                if built is None:
                    logger.warning("skip row %d: need non-empty prompt (+ predict preferred)", i)
                    n_skip += 1
                    continue
                ids, labels, extra = built

            hidden = embed_one(
                model, ids, device=args.device, max_len=args.max_len, layer=args.layer
            )
            t = hidden.shape[0]
            ids_t = ids[:t]
            labels_t = labels[:t]
            n_prompt = int(extra.get("n_prompt_tokens", 0)) if extra else 0
            if extra and n_prompt > t:
                n_prompt = sum(1 for x in labels_t if x == IGNORE_INDEX)
                extra["n_prompt_tokens"] = n_prompt
                extra["n_predict_tokens"] = t - n_prompt

            surfaces = [tok.decode([tid], skip_special_tokens=False) for tid in ids_t]
            text = tok.decode(ids_t, skip_special_tokens=False)

            stem = f"{i:06d}"
            pt_path = emb_dir / f"{stem}.pt"
            payload: dict[str, Any] = {
                "hidden": hidden,
                "input_ids": torch.tensor(ids_t, dtype=torch.long),
                "labels": torch.tensor(labels_t, dtype=torch.long),
                "token_surfaces": surfaces,
                "text": text,
                "layer": args.layer,
                "mode": args.mode,
                "adapter_path": args.adapter_path or "",
                "model_name_or_path": args.model_name_or_path,
            }
            if extra:
                payload.update(
                    {
                        "n_prompt_tokens": extra.get("n_prompt_tokens"),
                        "n_predict_tokens": extra.get("n_predict_tokens"),
                        "predict": extra.get("predict", ""),
                        "label": extra.get("label", ""),
                        "chatml_prompt": extra.get("chatml_prompt", ""),
                    }
                )
            torch.save(payload, pt_path)
            (emb_dir / f"{stem}.txt").write_text(text, encoding="utf-8")

            meta: dict[str, Any] = {
                "index": i,
                "path": str(pt_path.relative_to(out_dir)),
                "txt_path": f"embeddings/{stem}.txt",
                "length": t,
                "hidden_dim": int(hidden.shape[-1]),
                "n_completion": sum(1 for x in labels_t if x != IGNORE_INDEX),
                "task_id": row.get("task_id") or row.get("raw_id") or row.get("uid"),
                "layer": args.layer,
                "mode": args.mode,
            }
            if extra:
                meta["n_prompt_tokens"] = extra.get("n_prompt_tokens")
                meta["n_predict_tokens"] = extra.get("n_predict_tokens")
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            n_ok += 1
            if (i + 1) % 10 == 0 or i + 1 == len(rows):
                logger.info("embedded %d/%d (ok=%d skip=%d)", i + 1, len(rows), n_ok, n_skip)

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "model_name_or_path": args.model_name_or_path,
                "adapter_path": args.adapter_path or None,
                "data_path": args.data_path,
                "max_len": args.max_len,
                "layer": args.layer,
                "dtype": args.dtype,
                "thinking": bool(args.thinking) if args.mode == "test" else None,
                "n_samples": n_ok,
                "n_skipped": n_skip,
                "note": (
                    "train: compact input_ids (often gold completion). "
                    "test: ChatML(prompt)+predict(+EOS); labels mark predict span; "
                    "gold label stored in .pt but not in forward. "
                    "Use n_prompt_tokens to split hidden into context vs predict."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Done: %d samples (skipped %d) → %s", n_ok, n_skip, out_dir)


if __name__ == "__main__":
    main()
