import re
import ast
import json
import textwrap
import itertools
from dataclasses import dataclass, field
from typing import Any
# transformers is imported lazily inside get_qwen3_tokenizer() to avoid hard
# dependency at import time — annotation phase does not need it.


@dataclass(frozen=True)
class TokenCorrelation:
    token_i: str
    token_j: str
    source: str  # 'Symbolic' | 'Neural'
    subtype: str = ""  # e.g. 'syntactic' | 'dataflow' | 'type' | 'conceptual' | 'api'
    token_i_idx: int = -1   # index in the original token list (-1 = unknown)
    token_j_idx: int = -1

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.token_i, self.token_j, self.source, self.subtype)

    def as_full_tuple(self) -> tuple[str, str, str, str, int, int]:
        """Full tuple including indices — used for serialization/export."""
        return (self.token_i, self.token_j, self.source, self.subtype,
                self.token_i_idx, self.token_j_idx)

@dataclass(frozen=True)
class SubwordToken:
    surface: str  # decoded string e.g. "fetch", "_user", "Ġget"
    token_id: int  # BPE vocabulary id
    char_start: int  # byte/char offset in original code (inclusive)
    char_end: int  # byte/char offset in original code (exclusive)

    @property
    def clean(self) -> str:
        """Surface string with leading Ġ / whitespace stripped."""
        return self.surface.lstrip("\u0120 \t\n")


_CODE_TOKEN_RE = re.compile(
    r'//[^\n]*|'                 # C/C++/Java/C# single-line comment
    r'/\*[\s\S]*?\*/|'           # C/C++ block comment
    r'#\s*(?:include|define|ifndef|ifdef|endif|pragma|undef|if|else|elif)[^\n]*|'
                                 # C preprocessor directives — keep as one token
    r'#[^\n]*|'                  # Python/shell single-line comment (skip)
    r'"""[\s\S]*?"""|'           # triple double-quoted string
    r"'''[\s\S]*?'''|"           # triple single-quoted string
    r'"(?:[^"\\\n]|\\.)*"|'      # double-quoted string
    r"'(?:[^'\\\n]|\\.)*'|"      # single-quoted string
    r"\*\*=|//=|<<=|>>=|"
    r"==|!=|<=|>=|\*\*|//|->"
    r"|[+\-*/%&|^]=|<<|>>|::|"
    r"[A-Za-z_]\w*|"
    r"0[xX][0-9A-Fa-f]+|"
    r"0[bB][01]+|"
    r"0[oO][0-7]+|"
    r"\d+\.?\d*(?:[eE][+\-]?\d+)?|\.\d+(?:[eE][+\-]?\d+)?|"
    r"\S",
)

# Prefixes that should be dropped during annotation:
#   - Python/shell comments: '#' not followed by include/define/etc.
#   - C/C++ comments: '//' and '/*'
#   - Triple-quoted docstrings
# C preprocessor directives (#include, #define …) are intentionally KEPT.
_SKIP_RE = re.compile(
    r'^(?:'
    r'//|'                       # C++ line comment
    r'/\*|'                      # C block comment
    r'"""'  r'|'                 # Python docstring
    r"'''|"                      # Python docstring
    r'#(?!\s*(?:include|define|ifndef|ifdef|endif|pragma|undef|if|else|elif)\b)'
                                 # Python comment (not a C preprocessor directive)
    r')'
)

def tokenize_code(code: str) -> list[SubwordToken]:
    tokens = []
    for m in _CODE_TOKEN_RE.finditer(code):
        tokens.append(SubwordToken(
            surface=m.group(),
            token_id=-1,
            char_start=m.start(),
            char_end=m.end(),
        ))
    return tokens


def tokenize_code_for_annotation(code: str) -> list[SubwordToken]:
    """Same as tokenize_code but drops comments and docstrings."""
    return [t for t in tokenize_code(code)
            if not _SKIP_RE.match(t.surface)]

# ── BPE tokenizer (postprocessing / export only) ──────────────────────────────

_qwen2_5_tokenizer = None  # module-level cache: load once per process
QWEN2_5_CODER_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def get_qwen3_tokenizer():
    global _qwen2_5_tokenizer
    if _qwen2_5_tokenizer is None:
        from transformers import AutoTokenizer
        _qwen2_5_tokenizer = AutoTokenizer.from_pretrained(
            QWEN2_5_CODER_MODEL,
            trust_remote_code=True,
        )
    return _qwen2_5_tokenizer


def tokenize_subwords_bpe(code: str) -> list[SubwordToken]:
    """
    Tokenize `code` with the Qwen3-Coder BPE tokenizer.
    Call this only for postprocessing / export — **not** during annotation.
    """
    tok = get_qwen3_tokenizer()
    enc = tok(code, add_special_tokens=False, return_offsets_mapping=True)
    ids: list[int] = enc["input_ids"]
    offsets: list[tuple[int, int]] = enc["offset_mapping"]
    return [
        SubwordToken(
            surface=tok.decode([token_id]),
            token_id=token_id,
            char_start=start,
            char_end=end,
        )
        for token_id, (start, end) in zip(ids, offsets)
    ]


# kept for backwards-compat; delegates to simple tokenizer during annotation
def tokenize_subwords(code: str) -> list[SubwordToken]:
    """Alias for `tokenize_code` — use for annotation."""
    return tokenize_code(code)


# ── Postprocessing: map simple-token indices → BPE token indices ──────────────



def normalize_fim_annotation_edge_direction(
    *,
    si_is_completion: bool,
    sj_is_completion: bool,
    qi: int,
    qj: int,
    edge_scope: str,
) -> tuple[int, int] | None:
    """Normalize a simple-token annotation edge after remapping to ChatML tokens.

    In GoSingle-style masked completion, the model sees prefix and suffix in the
    user prompt, then predicts the assistant completion.  For edges crossing the
    masked completion boundary, the non-completion context token should be the
    source and the completion token should be the destination, even if the
    original complete-code order was completion -> suffix.

    For prompt-internal and completion-internal edges, keep normal left-to-right
    ChatML direction.  Returns None for self-edges or directions that would point
    backward in ChatML under the context_response_prompt policy.
    """
    if qi == qj:
        return None

    scope = (edge_scope or "answer_target").strip().lower()
    if scope == "context_response_prompt":
        if si_is_completion ^ sj_is_completion:
            src, dst = (qj, qi) if si_is_completion else (qi, qj)
        else:
            src, dst = (qi, qj) if qi < qj else (qj, qi)
        if src > dst:
            return None
        return int(src), int(dst)

    src, dst = (qi, qj) if qi < qj else (qj, qi)
    return int(src), int(dst)

def map_simple_to_bpe(
    simple_tokens: list[SubwordToken],
    bpe_tokens: list[SubwordToken],
) -> dict[int, list[int]]:
    """
    Build a mapping from simple-token index → list of overlapping BPE token
    indices, using char-offset intersection.

    Two tokens overlap when their char spans share at least one character:
        simple.char_start < bpe.char_end  AND  bpe.char_start < simple.char_end

    Returns a dict where every simple-token index is present; the value is an
    empty list if no BPE token overlaps (rare: e.g. whitespace-only gaps).

    Typical usage:
        simple  = tokenize_code(code)
        bpe     = tokenize_subwords_bpe(code)
        s2b     = map_simple_to_bpe(simple, bpe)
        # then patch TokenCorrelation indices:
        bpe_i   = s2b[corr.token_i_idx]   # list[int]
        bpe_j   = s2b[corr.token_j_idx]   # list[int]
    """
    mapping: dict[int, list[int]] = {i: [] for i in range(len(simple_tokens))}

    # Both lists are sorted by char_start; use a sliding window on bpe_tokens
    # to avoid O(n*m) behaviour.
    bpe_start = 0
    for si, st in enumerate(simple_tokens):
        # advance bpe cursor past tokens that end before this simple token starts
        while bpe_start < len(bpe_tokens) and bpe_tokens[bpe_start].char_end <= st.char_start:
            bpe_start += 1
        bi = bpe_start
        while bi < len(bpe_tokens) and bpe_tokens[bi].char_start < st.char_end:
            mapping[si].append(bi)
            bi += 1

    return mapping


def remap_correlations_to_bpe(
    correlations: list["TokenCorrelation"],
    s2b: dict[int, list[int]],
) -> list["TokenCorrelation"]:
    """
    Return a new list of TokenCorrelation objects whose `token_i_idx` /
    `token_j_idx` are replaced with the *first* BPE token index that overlaps
    the simple token.  Correlations whose simple index has no BPE mapping are
    dropped (logged as warnings).

    Call this after `map_simple_to_bpe` to produce the final export-ready set.
    """
    remapped: list[TokenCorrelation] = []
    for corr in correlations:
        bpe_i_list = s2b.get(corr.token_i_idx, [])
        bpe_j_list = s2b.get(corr.token_j_idx, [])
        if not bpe_i_list or not bpe_j_list:
            import warnings
            warnings.warn(
                f"No BPE mapping for simple token indices "
                f"({corr.token_i_idx}, {corr.token_j_idx}) — skipping."
            )
            continue
        remapped.append(TokenCorrelation(
            token_i=corr.token_i,
            token_j=corr.token_j,
            source=corr.source,
            subtype=corr.subtype,
            token_i_idx=bpe_i_list[0],
            token_j_idx=bpe_j_list[0],
        ))
    return remapped


# ── Identifier recovery (postprocessing) ──────────────────────────────────────

def build_subword_to_identifier_map(
        code: str,
        subwords: list[SubwordToken],
) -> dict[str, str]:
    """
    Map each subword surface string to the identifier it belongs to.

    Strategy: find all Python identifiers in `code` via regex, then for each
    subword check whether its char span falls inside an identifier span.
    Subwords that fall outside any identifier (punctuation, keywords, whitespace)
    map to their own clean surface string.

    Returns: { subword.surface -> identifier_string }
    """
    # All identifier spans in the source
    id_spans: list[tuple[int, int, str]] = [
        (m.start(), m.end(), m.group())
        for m in re.finditer(r"[A-Za-z_]\w*", code)
    ]

    mapping: dict[str, str] = {}
    for sw in subwords:
        parent = sw.clean  # default: map to itself
        for (id_start, id_end, id_str) in id_spans:
            # subword's char span overlaps with this identifier's span
            if sw.char_start >= id_start and sw.char_end <= id_end:
                parent = id_str
                break
        mapping[sw.surface] = parent
    return mapping


def postprocess_to_identifier_graph(
        subword_correlations: list[TokenCorrelation],
        subword_to_id: dict[str, str],
) -> list[TokenCorrelation]:
    # 补充一个 clean surface -> identifier 的映射
    # e.g. "Ġclosest" -> "closest_integer", clean = "closest" -> "closest_integer"
    clean_to_id: dict[str, str] = {}
    for surface, ident in subword_to_id.items():
        clean = surface.lstrip("\u0120 \t\n")
        if clean and clean not in clean_to_id:
            clean_to_id[clean] = ident

    def resolve(tok: str) -> str:
        # try exact surface match first
        if tok in subword_to_id:
            return subword_to_id[tok]
        # try clean match
        if tok in clean_to_id:
            return clean_to_id[tok]
        # fallback: the token itself (may be an identifier already)
        return tok

    seen: set[frozenset] = set()
    results: list[TokenCorrelation] = []

    for corr in subword_correlations:
        id_i = resolve(corr.token_i)
        id_j = resolve(corr.token_j)

        if id_i == id_j:
            continue

        key = frozenset({id_i, id_j})
        if key not in seen:
            seen.add(key)
            results.append(TokenCorrelation(id_i, id_j, corr.source, corr.subtype))

    return results

# ── Convenience: surface strings only (for annotator input) ──────────────────

def get_subword_surfaces(subwords: list[SubwordToken]) -> list[str]:
    """Return unique subword surface strings in order of first appearance."""
    seen, out = set(), []
    for sw in subwords:
        if sw.surface not in seen:
            seen.add(sw.surface)
            out.append(sw.surface)
    return out


def get_token_indices_in_span(
    subwords: list[SubwordToken],
    char_start: int,
    char_end: int,
) -> set[int]:
    """
    Return the set of indices of subwords whose char span overlaps
    [char_start, char_end).

    Used to restrict annotation to the Incomplete Code block only:
    all subwords are still tokenized (preserving global indices), but
    only pairs where both token_i and token_j fall in this set are emitted.
    """
    return {
        i for i, sw in enumerate(subwords)
        if sw.char_start < char_end and sw.char_end > char_start
    }


@dataclass
class AnnotationResult:
    high_confidence: list[TokenCorrelation]  # S & N  — retain directly
    symbolic_only: list[TokenCorrelation]  # S \ N  — retain
    neural_only: list[TokenCorrelation]  # N \ S  — needs counterfactual verification

    def all_retained(self) -> list[TokenCorrelation]:
        return self.high_confidence + self.symbolic_only

    def needs_verification(self) -> list[TokenCorrelation]:
        return self.neural_only


def merge(
        symbolic: list[TokenCorrelation],
        neural:   list[TokenCorrelation],
) -> AnnotationResult:
    def key(c: TokenCorrelation) -> frozenset[str]:
        return frozenset({c.token_i, c.token_j})

    sym_keys = {key(c): c for c in symbolic}
    neu_keys = {key(c): c for c in neural}

    def merge_indices(sym_c: TokenCorrelation, neu_c: TokenCorrelation) -> TokenCorrelation:
        """For S∩N edges, keep symbolic subtype but fill in indices from whichever has them."""
        return TokenCorrelation(
            token_i=sym_c.token_i,
            token_j=sym_c.token_j,
            source="Symbolic+Neural",
            subtype=sym_c.subtype,
            token_i_idx=sym_c.token_i_idx if sym_c.token_i_idx != -1 else neu_c.token_i_idx,
            token_j_idx=sym_c.token_j_idx if sym_c.token_j_idx != -1 else neu_c.token_j_idx,
        )

    return AnnotationResult(
        high_confidence=[
            merge_indices(sym_keys[k], neu_keys[k])
            for k in sym_keys if k in neu_keys
        ],
        symbolic_only=[c for k, c in sym_keys.items() if k not in neu_keys],
        neural_only  =[c for k, c in neu_keys.items() if k not in sym_keys],
    )