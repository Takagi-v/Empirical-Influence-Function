"""Unit tests for paper §B ALTI L1 normalization + saliency_agg switch."""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "train"))

from loss import (  # noqa: E402
    canonical_saliency_agg,
    normalize_alti_importance_l1,
)


def test_canonical_saliency_agg():
    assert canonical_saliency_agg("last") == "last"
    assert canonical_saliency_agg("rollout") == "rollout"
    assert canonical_saliency_agg("alti_b") == "rollout"
    assert canonical_saliency_agg(None) == "last"


def test_normalize_alti_importance_l1_matches_paper_formula():
    # One query, three sources, D=2. Construct T so y = sum_j T_j is known.
    # T0=(1,0), T1=(0,2), T2=(0,0) → y=(1,2), ||y||_1=3
    # d0=||y-T0||_1=||(0,2)||_1=2 → score0=max(0,3-2)=1
    # d1=||y-T1||_1=||(1,0)||_1=1 → score1=max(0,3-1)=2
    # d2=||y-T2||_1=||(1,2)||_1=3 → score2=max(0,3-3)=0
    # C = [1,2,0] / 3
    source = torch.tensor([
        [[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]],
    ])  # [Q=1, S=3, D=2]
    C = normalize_alti_importance_l1(source)
    expected = torch.tensor([[1.0 / 3.0, 2.0 / 3.0, 0.0]])
    assert torch.allclose(C, expected, atol=1e-6)
    assert torch.allclose(C.sum(dim=-1), torch.ones(1), atol=1e-6)


def test_normalize_is_differentiable():
    source = torch.randn(2, 5, 8, requires_grad=True)
    C = normalize_alti_importance_l1(source)
    C.sum().backward()
    assert source.grad is not None
    assert source.grad.shape == source.shape


def test_rollout_left_multiply_order():
    # Sanity: C2 @ C1 applied left-to-right matches successive matmul used in code.
    C1 = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32)
    C2 = torch.tensor([[0.6, 0.4], [0.1, 0.9]], dtype=torch.float32)
    rollout = C1
    rollout = torch.matmul(C2, rollout)
    expected = C2 @ C1
    assert torch.allclose(rollout, expected)


if __name__ == "__main__":
    test_canonical_saliency_agg()
    test_normalize_alti_importance_l1_matches_paper_formula()
    test_normalize_is_differentiable()
    test_rollout_left_multiply_order()
    print("ok")
