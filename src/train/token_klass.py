"""Token surface-class heuristics for protected annot-skip.

Replicates the protected-infosel classification used in the new-go study
(keyword / punct / whitespace always kept; only identifier/number may be
dropped), but lives entirely inside Empirical-Influence-Function so remote
training does not depend on scripts/newgo_infosel.
"""
from __future__ import annotations

import re

# Go keywords + common literals treated as structural / highly inferable.
GO_KEYWORDS = frozenset({
    "break", "case", "chan", "const", "continue", "default", "defer", "else",
    "fallthrough", "for", "func", "go", "goto", "if", "import", "interface",
    "map", "package", "range", "return", "select", "struct", "switch", "type",
    "var", "nil", "true", "false", "iota",
})

# Classes that may be dropped when a completion token has no annotation.
DROPPABLE_CLASSES = frozenset({"identifier", "number"})


def classify_token_surface(text: str, *, language: str = "go") -> str:
    """Classify a decoded BPE token surface into a coarse lexical class.

    Returns one of: ``whitespace`` | ``number`` | ``keyword`` | ``identifier``
    | ``punct/op``.
    """
    core = text.strip()
    if core == "":
        return "whitespace"
    if re.fullmatch(r"[0-9][0-9_]*\.?[0-9]*", core):
        return "number"
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", core)
    if words:
        w = words[0]
        rest = re.sub(re.escape(w), "", core, count=1)
        if w in GO_KEYWORDS and re.fullmatch(r"[\s\W]*", rest or ""):
            return "keyword"
        return "identifier"
    return "punct/op"


def is_protected_completion_token(
    *,
    relative_pos: int,
    surface: str,
    keep_first: int = 2,
    language: str = "go",
) -> bool:
    """Whether a completion token must stay in the CE loss under annot-skip.

    Protected if:
      (1) relative position in the completion is < ``keep_first`` (default: 0,1);
      (2) surface class is not identifier/number (keyword / punct / whitespace).
    """
    if relative_pos < max(0, int(keep_first)):
        return True
    return classify_token_surface(surface, language=language) not in DROPPABLE_CLASSES
