#!/usr/bin/env python3
"""Go-single wrapper for the shared multi-target overfit experiment."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "scripts/saliency_exp/overfit_multi_target.py"), run_name="__main__")
