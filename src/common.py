"""Shared helpers: experiment-config loading + uniform output paths.

Output layout (uniform across every experiment):
    outputs/<name>/
        checkpoints/      LoRA adapter + checkpoint-* (training writes here)
        train/            train.log, train_curves.json
        eval/             <dataset>.json (per dataset) + summary.json
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPO_ROOT / "outputs"


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment(path: str | Path) -> dict:
    cfg = load_yaml(path)
    if "name" not in cfg:
        raise ValueError(f"experiment config {path} missing required 'name'")
    return cfg


@dataclass
class ExpPaths:
    name: str
    root: Path
    checkpoints: Path
    train: Path
    eval: Path

    @classmethod
    def for_experiment(cls, name: str) -> "ExpPaths":
        root = OUTPUTS_ROOT / name
        p = cls(
            name=name,
            root=root,
            checkpoints=root / "checkpoints",
            train=root / "train",
            eval=root / "eval",
        )
        return p

    def mkdirs(self) -> "ExpPaths":
        for d in (self.checkpoints, self.train, self.eval):
            d.mkdir(parents=True, exist_ok=True)
        return self


def compute_max_steps(n_samples: int, epochs: int, batch_size: int, grad_accum: int) -> int:
    steps_per_epoch = math.ceil(n_samples / (batch_size * grad_accum))
    return steps_per_epoch * epochs
