"""Shared source-cleaning helpers (comment stripping + blank-line normalization).

Used by dataset builders (e.g. scripts/benchmark/build_new_go_data.py) and the
eval harness (scripts/benchmark/api_eval.py) so both sides apply *identical*
normalization. Keeping a single implementation avoids metric drift between how
inputs are constructed and how predictions/references are scored.
"""
from __future__ import annotations


def remove_c_like_comments(code: str) -> str:
    """Strip // line comments and /* */ block comments (C/Go/Java/...).

    String and char literals are respected. Go raw strings (backticks) are also
    treated as string literals so // inside them is preserved.
    """
    out: list[str] = []
    i = 0
    n = len(code)
    state = "normal"
    quote = ""
    while i < n:
        ch = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if state == "normal":
            if ch in {'"', "'", "`"}:
                quote = ch
                state = "string"
                out.append(ch)
                i += 1
            elif ch == "/" and nxt == "/":
                i += 2
                while i < n and code[i] != "\n":
                    i += 1
                if i < n:
                    out.append("\n")
                    i += 1
            elif ch == "/" and nxt == "*":
                i += 2
                while i + 1 < n and not (code[i] == "*" and code[i + 1] == "/"):
                    if code[i] == "\n":
                        out.append("\n")
                    i += 1
                i += 2 if i + 1 < n else 0
            else:
                out.append(ch)
                i += 1
        elif state == "string":
            out.append(ch)
            # backtick raw strings have no escapes; " and ' do.
            if quote != "`" and ch == "\\" and i + 1 < n:
                out.append(code[i + 1])
                i += 2
            elif ch == quote:
                state = "normal"
                i += 1
            else:
                i += 1
    return "".join(out)


def normalize_blank_lines(code: str) -> str:
    """rstrip each line, collapse runs of blank lines to one, strip ends."""
    lines = [ln.rstrip() for ln in code.splitlines()]
    compact: list[str] = []
    blank = 0
    for ln in lines:
        if ln.strip():
            compact.append(ln)
            blank = 0
        else:
            blank += 1
            if blank <= 1:
                compact.append("")
    return "\n".join(compact).strip("\n")


def strip_comments(code: str, language: str = "go") -> str:
    """Remove comments then normalize blank lines. C-like languages only here."""
    cleaned = remove_c_like_comments(code)
    return normalize_blank_lines(cleaned)


def normalize_for_match(text: str, language: str = "go") -> str:
    """Canonical form for exact-match / edit-similarity scoring.

    Strips comments, drops blank lines entirely, and rstrips each line so that
    cosmetic differences (comments, trailing whitespace, blank lines) do not
    affect EM/ES. Applied identically to predictions and references.
    """
    cleaned = remove_c_like_comments(text)
    lines = [ln.rstrip() for ln in cleaned.splitlines() if ln.strip()]
    return "\n".join(lines).strip()


def edit_similarity(a: str, b: str) -> float:
    """Character-level normalized Levenshtein similarity in [0, 1].

    1.0 == identical, 0.0 == maximally different. Matches the "edit similarity"
    (ES) metric used by RepoBench / CrossCodeEval.
    """
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    dist = prev[lb]
    return 1.0 - dist / max(la, lb)
