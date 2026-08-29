"""Unit tests for annot-skip protection / classification."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "train"))

from token_klass import (  # noqa: E402
    classify_token_surface,
    is_protected_completion_token,
)


def test_classify_surfaces():
    assert classify_token_surface("  ") == "whitespace"
    assert classify_token_surface("\n") == "whitespace"
    assert classify_token_surface("42") == "number"
    assert classify_token_surface("if") == "keyword"
    assert classify_token_surface("return") == "keyword"
    assert classify_token_surface("nic") == "identifier"
    assert classify_token_surface("err") == "identifier"
    assert classify_token_surface("(") == "punct/op"
    assert classify_token_surface(":=") == "punct/op"


def test_protect_first_two():
    assert is_protected_completion_token(relative_pos=0, surface="nic") is True
    assert is_protected_completion_token(relative_pos=1, surface="Foo") is True
    assert is_protected_completion_token(relative_pos=2, surface="nic") is False


def test_protect_structural_classes():
    assert is_protected_completion_token(relative_pos=5, surface="if") is True
    assert is_protected_completion_token(relative_pos=5, surface="  ") is True
    assert is_protected_completion_token(relative_pos=5, surface=")") is True
    assert is_protected_completion_token(relative_pos=5, surface="42") is False
    assert is_protected_completion_token(relative_pos=5, surface="userID") is False


def test_annot_skip_logic_on_labels():
    """Simulate _apply_annot_skip without a full tokenizer."""
    IGNORE = -100
    # prompt [0,1], completion [2,3,4,5,6] = first, second, keyword, id(unannot), id(annot)
    labels = [IGNORE, IGNORE, 10, 11, 12, 13, 14]
    # relative: 0->pos2, 1->pos3, 2->pos4, 3->pos5, 4->pos6
    annotated_dsts = {6}  # only last completion token annotated
    surfaces = {2: "X", 3: "Y", 4: "if", 5: "nic", 6: "err"}
    keep_first = 2
    out = list(labels)
    excluded = []
    comp = [i for i, lab in enumerate(out) if lab != IGNORE]
    for rel, pos in enumerate(comp):
        if pos in annotated_dsts:
            continue
        if is_protected_completion_token(
            relative_pos=rel, surface=surfaces[pos], keep_first=keep_first,
        ):
            continue
        out[pos] = IGNORE
        excluded.append(pos)
    # pos5 = unannotated identifier at rel=3 → dropped; others kept
    assert excluded == [5]
    assert out == [IGNORE, IGNORE, 10, 11, 12, IGNORE, 14]


if __name__ == "__main__":
    test_classify_surfaces()
    test_protect_first_two()
    test_protect_structural_classes()
    test_annot_skip_logic_on_labels()
    print("ok")
