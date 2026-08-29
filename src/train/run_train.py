"""Unified training launcher — config-driven, single entry point.

    python src/train/run_train.py --config configs/experiments/<exp>.yaml

Reads the experiment YAML, resolves dataset paths from the registry, computes
max_steps from epochs, builds the argument list for the underlying trainer
(src/train/train.py), and runs it with uniform output paths
(outputs/<name>/checkpoints + outputs/<name>/train/train.log).

This replaces the dozens of copy-pasted train_*.sh scripts. Add a new experiment
= add a YAML, not a shell script.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve()
SRC = THIS.parents[1]
REPO = THIS.parents[2]
sys.path.insert(0, str(SRC))

from common import ExpPaths, compute_max_steps, load_experiment   # noqa: E402
from data.registry import get_dataset, default_model              # noqa: E402

TRAIN_PY = SRC / "train" / "train.py"


def build_argv(cfg: dict) -> tuple[list[str], ExpPaths]:
    name = cfg["name"]
    paths = ExpPaths.for_experiment(name).mkdirs()
    t = cfg.get("train", {})
    data = cfg.get("data", {})
    model = cfg.get("model", default_model())

    # TEMP DEBUG: data.raw_path + raw_prompt_response bypasses the compact registry.
    raw_path = data.get("raw_path")
    use_raw = bool(data.get("raw_prompt_response") or raw_path)
    if use_raw:
        if not raw_path:
            raise ValueError("data.raw_prompt_response requires data.raw_path to a {prompt,response} JSONL")
        train_path = Path(raw_path)
        if not train_path.is_absolute():
            train_path = (REPO / train_path).resolve()
        n = int(data.get("raw_max_samples") or 0) or sum(1 for _ in open(train_path, encoding="utf-8"))
        train_ds_path = str(train_path)
        train_ds = None
    else:
        train_ds = get_dataset(data["train"])
        if train_ds.format != "compact":
            raise ValueError(f"train dataset {train_ds.name} must be compact, got {train_ds.format}")
        n = train_ds.n or sum(1 for _ in open(train_ds.path))
        train_ds_path = str(train_ds.path)

    bs = int(t.get("batch_size", 1))
    ga = int(t.get("grad_accum", 8))
    epochs = int(t.get("epochs", 3))
    max_steps = compute_max_steps(n, epochs, bs, ga)
    # Save once per epoch by default.
    save_steps = int(t.get("save_steps", max(1, max_steps // epochs)))

    lora = t.get("lora", {}) or {}
    # full_finetune: true (or lora: false) -> train all params, no LoRA adapter.
    full_ft = bool(t.get("full_finetune", False)) or (t.get("lora", "unset") is False)
    # Multi-GPU launch (config-gated). train.nproc_per_node>1 -> torchrun (data-parallel;
    # pair with train.fsdp to shard a big model across ranks). Orthogonally,
    # train.device_map='auto' shards ONE model across all visible GPUs (naive MP, single
    # process). Default (nproc=1, no device_map) = the original single-GPU path.
    nproc = int(t.get("nproc_per_node", 1))
    launcher = ["torchrun", "--standalone", f"--nproc_per_node={nproc}"] if nproc > 1 else [sys.executable]
    argv = [
        *launcher, str(TRAIN_PY),
        "--model_name_or_path", str(model),
        "--data_path", train_ds_path,
        "--output_dir", str(paths.checkpoints),
        "--use_peft", "False" if full_ft else "True",
        "--loss_mode", str(t.get("loss_mode", "ce_only")),
        "--per_device_train_batch_size", str(bs),
        "--gradient_accumulation_steps", str(ga),
        "--max_steps", str(max_steps),
        "--learning_rate", str(t.get("lr", 2e-4)),
        "--max_len", str(t.get("max_len", 8192)),
        "--logging_steps", "1",
        "--save_strategy", "steps",
        "--save_steps", str(save_steps),
        "--save_total_limit", str(t.get("save_total_limit", 4)),
        "--report_to", "none",
    ]
    if not full_ft:
        argv += [
            "--lora_r", str(lora.get("r", 32)),
            "--lora_alpha", str(lora.get("alpha", 64)),
            "--lora_dropout", str(lora.get("dropout", 0.05)),
        ]
        tm = lora.get("target_modules")
        if tm:
            argv += ["--lora_target_modules", ",".join(tm) if isinstance(tm, list) else str(tm)]
    if t.get("gradient_checkpointing"):
        argv += ["--gradient_checkpointing", "True"]
    # Large-model sharding (see nproc/launcher above): device_map='auto' (naive MP) OR
    # fsdp with nproc_per_node>1 (data-parallel + sharded). They are mutually exclusive.
    if t.get("device_map"):
        argv += ["--device_map", str(t["device_map"])]
    if nproc > 1 and t.get("fsdp"):
        argv += ["--fsdp", str(t["fsdp"])]
        # Newer transformers dropped --fsdp_transformer_layer_cls_to_wrap;
        # wrap class + activation_checkpointing live in --fsdp_config.
        fsdp_cfg = dict(t.get("fsdp_config") or {})
        if t.get("fsdp_wrap") and "transformer_layer_cls_to_wrap" not in fsdp_cfg:
            wrap = t["fsdp_wrap"]
            fsdp_cfg["transformer_layer_cls_to_wrap"] = (
                wrap if isinstance(wrap, list) else [wrap]
            )
        if t.get("gradient_checkpointing") and "activation_checkpointing" not in fsdp_cfg:
            fsdp_cfg["activation_checkpointing"] = True
        if fsdp_cfg:
            import json as _json
            argv += ["--fsdp_config", _json.dumps(fsdp_cfg)]

    # Optional seed override (for repeat runs to test seed variance). Default
    # omitted -> HF TrainingArguments default (42).
    if t.get("seed") is not None:
        argv += ["--seed", str(int(t["seed"]))]

    # ---- edge-label augmentation (config-gated, default OFF) ------------------
    aug = data.get("edge_augment", {}) or {}
    if aug.get("enabled"):
        argv += [
            "--edge_augment", "True",
            "--edge_augment_decay", str(aug.get("decay", 0.5)),
            "--edge_augment_max_hops", str(aug.get("max_hops", 0)),
            "--edge_augment_mode", str(aug.get("mode", "directed")),
        ]
        if aug.get("node_weight"):
            argv += ["--edge_augment_node_weight", "True"]

    # ---- teacher-gated informative-token selection (config-gated, default OFF)
    sel = data.get("token_select", {}) or {}
    if sel.get("enabled"):
        argv += [
            "--token_select", "True",
            "--token_select_threshold", str(sel.get("threshold", 2.0)),
            "--token_select_keep_special", "True" if sel.get("keep_special", True) else "False",
        ]

    # ---- annotation-gated skip with structural protection (default OFF)
    ask = data.get("annot_skip", {}) or {}
    if ask.get("enabled"):
        argv += [
            "--annot_skip", "True",
            "--annot_skip_keep_first", str(ask.get("keep_first", 2)),
            "--annot_skip_keep_special", "True" if ask.get("keep_special", True) else "False",
        ]

    # ---- trailing EOS supervision (default ON; set supervise_eos: false to disable)
    if "supervise_eos" in data:
        argv += ["--supervise_eos", "True" if data.get("supervise_eos") else "False"]
    elif "supervise_eos" in t:
        argv += ["--supervise_eos", "True" if t.get("supervise_eos") else "False"]

    if t.get("eos_loss_weight") is not None:
        argv += ["--eos_loss_weight", str(t.get("eos_loss_weight"))]

    if use_raw:
        argv += ["--raw_prompt_response", "True"]
        if data.get("raw_max_samples"):
            argv += ["--raw_max_samples", str(int(data["raw_max_samples"]))]

    valid_name = data.get("valid")
    if use_raw:
        valid_name = None  # raw debug path has no registry valid set
    if valid_name:
        valid_ds = get_dataset(valid_name)
        argv += [
            "--eval_data_path", str(valid_ds.resolve("compact")),
            "--eval_max_samples", str(t.get("eval_max_samples", 500)),
            "--eval_strategy", "steps",
            "--eval_steps", str(t.get("eval_steps", save_steps)),
            "--per_device_eval_batch_size", str(t.get("eval_batch_size", 2)),
        ]
        if t.get("eval_codebleu_samples"):
            argv += ["--eval_codebleu_samples", str(t["eval_codebleu_samples"])]

    # ---- loss-specific knobs --------------------------------------------------
    mode = t.get("loss_mode", "ce_only")
    if mode in ("ce_shortcut_mask", "ce_shortcut_mask_saliency"):
        cf = t.get("cfmask", {}) or {}
        argv += [
            "--cfmask_rate", str(cf.get("rate", 0.3)),
            "--cfmask_recency_window", str(cf.get("recency_window", 8)),
            "--cfmask_protect_prefix", str(cf.get("protect_prefix", 16)),
            "--cfmask_exclude_special", str(cf.get("exclude_special", True)),
            "--cfmask_invariance_beta", str(cf.get("invariance_beta", 0.0)),
        ]
        if "max_k" in cf:
            argv += ["--cfmask_max_k", str(cf["max_k"])]
        if "min_k" in cf:
            argv += ["--cfmask_min_k", str(cf["min_k"])]
        if cf.get("weight_aware"):
            argv += [
                "--cfmask_weight_aware", "True",
                "--cfmask_p_max", str(cf.get("p_max", 0.9)),
                "--cfmask_p_gamma", str(cf.get("p_gamma", 1.0)),
            ]
        if cf.get("per_target"):
            argv += ["--cfmask_per_target", "True"]
    if mode in ("ce_saliency", "saliency_only", "ce_shortcut_mask_saliency"):
        sal = t.get("saliency", {}) or {}
        argv += [
            "--saliency_loss_type", str(sal.get("loss_type", "contrastive")),
            "--saliency_lambda", str(sal.get("lambda", 1.5)),
            "--saliency_alpha", str(sal.get("alpha", 1.0)),
            "--saliency_margin_plus", str(sal.get("margin_plus", 2.0)),
            "--saliency_layer", str(sal.get("layer", -1)),
            "--saliency_agg", str(sal.get("agg", "last")),
        ]
        if sal.get("neg_sample_k"):
            argv += ["--saliency_neg_sample_k", str(sal["neg_sample_k"])]
        if sal.get("exclude_sink_prefix"):
            argv += ["--saliency_exclude_sink_prefix", str(sal["exclude_sink_prefix"])]
        if sal.get("exclude_special_tokens"):
            argv += ["--saliency_exclude_special_tokens", "True"]
        if sal.get("detail_log_steps"):
            argv += [
                "--saliency_detail_log_steps", str(sal["detail_log_steps"]),
                "--saliency_detail_log_path", str(paths.train / "saliency_detail.jsonl"),
                "--saliency_detail_top_k", str(sal.get("detail_top_k", 10)),
            ]
    if mode == "ce_edge_pred":
        ep = t.get("edge", {}) or {}
        argv += [
            "--edge_lambda", str(ep.get("lambda", 0.5)),
            "--edge_proj_dim", str(ep.get("proj_dim", 256)),
            "--edge_neg_weight", str(ep.get("neg_weight", 1.0)),
            "--edge_temperature", str(ep.get("temperature", 1.0)),
            "--edge_layer", str(ep.get("layer", -1)),
        ]
        if ep.get("neg_sample_k"):
            argv += ["--edge_neg_sample_k", str(ep["neg_sample_k"])]
    if mode == "ce_attn_bias":
        ab = t.get("attn_bias", {}) or {}
        argv += ["--attn_bias_init", str(ab.get("init", 1.0))]
    return argv, paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified config-driven training launcher.")
    ap.add_argument("--config", required=True, help="experiment YAML")
    ap.add_argument("--dry_run", action="store_true", help="print resolved command and exit")
    args = ap.parse_args()

    cfg = load_experiment(args.config)
    argv, paths = build_argv(cfg)
    log_path = paths.train / "train.log"

    print(f"[run_train] experiment: {cfg['name']}")
    print(f"[run_train] checkpoints -> {paths.checkpoints}")
    print(f"[run_train] log         -> {log_path}")
    print("[run_train] command:\n  " + " ".join(argv) + "\n")
    if args.dry_run:
        return

    # Persist a self-documenting meta.json (intention + full recipe) next to the
    # run, so months later we can recall WHAT this run was for and HOW it was
    # configured. `description:` is a free-text field in the experiment YAML.
    t = cfg.get("train", {}) or {}
    meta = {
        "name": cfg["name"],
        "description": cfg.get("description", ""),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": cfg.get("model", default_model()),
        "train_data": (cfg.get("data", {}) or {}).get("train"),
        "loss_mode": t.get("loss_mode", "ce_only"),
        "epochs": t.get("epochs"),
        "lr": t.get("lr"),
        "seed": t.get("seed"),
        "config": cfg,
    }
    (paths.root / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")
    _desc = cfg.get("description")
    print(f"[run_train] meta       -> {paths.root / 'meta.json'}"
          + (f"  ({_desc[:60]})" if _desc else "  (no description set)"))

    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(argv, stdout=logf, stderr=subprocess.STDOUT, cwd=str(REPO))
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\nexit={proc.returncode}\n")
    print(f"[run_train] done, exit={proc.returncode}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
