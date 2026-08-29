"""Unit tests for left-truncate + attention_edges realignment."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "train"))

from dataset import AnnotatedSFTDataset, IGNORE_INDEX  # noqa: E402


def _ds() -> AnnotatedSFTDataset:
    """Minimal instance without running __init__ (no data / tokenizer)."""
    obj = AnnotatedSFTDataset.__new__(AnnotatedSFTDataset)
    obj._left_truncated = 0
    obj._edges_dropped_by_trunc = 0
    return obj


def test_no_op_when_short():
    ds = _ds()
    ids = [1, 2, 3, 4]
    labs = [IGNORE_INDEX, IGNORE_INDEX, 3, 4]
    edges = [{"src": 0, "dst": 2}, {"src": 1, "dst": 3}]
    out_ids, out_labs, out_edges = ds._left_truncate_align_edges(ids, labs, edges, max_len=8)
    assert out_ids == ids
    assert out_labs == labs
    assert out_edges == edges
    assert ds._left_truncated == 0


def test_left_truncate_shifts_and_filters_edges():
    ds = _ds()
    # positions: 0 1 2 3 4 5 6 7 ; keep last 4 → drop=4 → new 0..3 were old 4..7
    ids = list(range(10, 18))
    labs = [IGNORE_INDEX] * 5 + [15, 16, 17]
    edges = [
        {"src": 0, "dst": 1},   # both in dropped prefix → gone
        {"src": 1, "dst": 5},   # src out after shift → gone
        {"src": 4, "dst": 6},   # → (0, 2) kept
        {"src": 5, "dst": 7},   # → (1, 3) kept
        {"src": 6, "dst": 7},   # → (2, 3) kept
        {"token_i_idx": 3, "token_j_idx": 7},  # → (-1, 3) invalid → gone
    ]
    out_ids, out_labs, out_edges = ds._left_truncate_align_edges(ids, labs, edges, max_len=4)
    assert out_ids == [14, 15, 16, 17]
    assert out_labs == [IGNORE_INDEX, 15, 16, 17]
    assert [(e["src"], e["dst"]) for e in out_edges if "src" in e] == [(0, 2), (1, 3), (2, 3)]
    assert ds._left_truncated == 1
    assert ds._edges_dropped_by_trunc == 3  # first two + token_i/j form


def test_skip_sample_when_completion_fully_cropped():
    """After truncate, all labels IGNORE → loader should skip (caller check)."""
    ds = _ds()
    ids = list(range(20))
    # completion only in the prefix; left-truncate to last 4 → all IGNORE
    labs = [10, 11, 12, 13] + [IGNORE_INDEX] * 16
    out_ids, out_labs, _ = ds._left_truncate_align_edges(ids, labs, [], max_len=4)
    assert out_ids == [16, 17, 18, 19]
    assert all(l == IGNORE_INDEX for l in out_labs)


def test_source_target_keys_remapped():
    ds = _ds()
    ids = list(range(6))
    labs = [IGNORE_INDEX] * 3 + [3, 4, 5]
    edges = [{"source": 2, "target": 5}]
    _, _, out = ds._left_truncate_align_edges(ids, labs, edges, max_len=4)
    assert len(out) == 1
    assert out[0]["source"] == 0 and out[0]["target"] == 3


if __name__ == "__main__":
    test_no_op_when_short()
    test_left_truncate_shifts_and_filters_edges()
    test_skip_sample_when_completion_fully_cropped()
    test_source_target_keys_remapped()
    print("ok")
