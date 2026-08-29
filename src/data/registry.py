"""Dataset registry — single source of truth for dataset paths/specs.

Load `configs/datasets.yaml` once and resolve dataset specs by name. All paths
are resolved to absolute (relative to the repo root). Nothing else in the
codebase should hardcode dataset paths — add new datasets to the YAML instead.

Usage:
    from data.registry import get_dataset, list_datasets, REPO_ROOT
    ds = get_dataset("csn_test")          # -> DatasetSpec
    ds.path                               # absolute path (generation format)
    ds.has_tests                          # bool
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import functools

import yaml

# repo root = two levels up from this file (src/data/registry.py -> repo)
REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "configs" / "datasets.yaml"


@dataclass
class DatasetSpec:
    name: str
    language: str
    role: str                       # train | valid | test
    format: str                     # compact | chatml
    has_tests: bool
    path: Path                      # primary path (resolved absolute)
    source_dataset: Optional[str] = None   # judge-routing key
    compact_path: Optional[Path] = None    # teacher-forcing representation
    chatml_path: Optional[Path] = None     # generation representation
    n: Optional[int] = None
    raw: dict = field(default_factory=dict)

    def resolve(self, kind: str) -> Path:
        """Return the path for a requested physical format.

        kind="compact": prefer compact_path, else path if format==compact.
        kind="chatml" : prefer chatml_path, else path if format==chatml.
        Raises if the requested format is unavailable for this dataset.
        """
        if kind == "compact":
            if self.format == "compact":
                return self.path
            if self.compact_path is not None:
                return self.compact_path
            raise ValueError(f"dataset {self.name!r} has no compact representation")
        if kind == "chatml":
            if self.format == "chatml":
                return self.path
            if self.chatml_path is not None:
                return self.chatml_path
            raise ValueError(f"dataset {self.name!r} has no chatml representation")
        raise ValueError(f"unknown format kind {kind!r} (expected compact|chatml)")


def _abs(p) -> Optional[Path]:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


@functools.lru_cache(maxsize=1)
def _load_raw(registry_path: str = str(REGISTRY_PATH)) -> dict:
    with open(registry_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def _load_specs() -> dict[str, DatasetSpec]:
    raw = _load_raw()
    defaults = raw.get("defaults", {}) or {}
    specs: dict[str, DatasetSpec] = {}
    for name, d in (raw.get("datasets", {}) or {}).items():
        specs[name] = DatasetSpec(
            name=name,
            language=d.get("language", defaults.get("language", "go")),
            role=d["role"],
            format=d["format"],
            has_tests=bool(d.get("has_tests", False)),
            path=_abs(d["path"]),
            source_dataset=d.get("source_dataset"),
            compact_path=_abs(d.get("compact_path")),
            chatml_path=_abs(d.get("chatml_path")),
            n=d.get("n"),
            raw=d,
        )
    return specs


def get_dataset(name: str, *, check_exists: bool = True) -> DatasetSpec:
    specs = _load_specs()
    if name not in specs:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(specs)}")
    spec = specs[name]
    if check_exists and not spec.path.exists():
        raise FileNotFoundError(f"dataset {name!r} path missing: {spec.path}")
    return spec


def list_datasets(role: str | None = None, has_tests: bool | None = None,
                  language: str | None = None) -> list[DatasetSpec]:
    out = []
    for s in _load_specs().values():
        if role is not None and s.role != role:
            continue
        if has_tests is not None and s.has_tests != has_tests:
            continue
        if language is not None and s.language != language:
            continue
        out.append(s)
    return out


def default_model() -> str:
    return (_load_raw().get("defaults", {}) or {}).get("model", "models/Qwen2.5-Coder-7B-Instruct")


if __name__ == "__main__":
    # Self-check: print the registry and validate every path exists.
    print(f"repo root: {REPO_ROOT}")
    print(f"registry : {REGISTRY_PATH}\n")
    ok = True
    for s in _load_specs().values():
        exists = s.path.exists()
        ok &= exists
        flags = f"role={s.role} fmt={s.format} tests={s.has_tests}"
        print(f"[{'OK ' if exists else 'MISS'}] {s.name:16s} {flags:38s} {s.path}")
        for k in ("compact_path", "chatml_path"):
            p = getattr(s, k)
            if p is not None:
                e = p.exists(); ok &= e
                print(f"        {'OK ' if e else 'MISS'} {k}: {p}")
    print("\nALL PATHS OK" if ok else "\nSOME PATHS MISSING")
