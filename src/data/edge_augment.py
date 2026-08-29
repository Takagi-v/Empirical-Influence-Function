"""Transitive-closure label augmentation for graph-signal edges.

The raw graph signal is single-hop: an edge ``a -> b`` marks a *direct*
dependency. This module densifies it by transitive closure — if ``a -> b`` and
``b -> c`` exist, it adds ``a -> c`` — so descendants inherit their ancestors'
labels. Every inherited edge carries a decayed weight, so downstream consumers
can still distinguish first-order edges (weight ``1.0``) from transitively
inherited ones (weight ``decay ** hops``).

Graph-signal edges are a DAG ordered by token position (``src < dst`` for every
edge in the data), so the closure always terminates and min-hop distances are
well defined.

Everything here is a pure function and is config-gated at the call site; when
augmentation is disabled the existing dataset path is untouched.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable


def augment_edges(
    edges: Iterable[dict],
    *,
    decay: float = 0.5,
    max_hops: int = 0,
    n_tokens: int | None = None,
    mode: str = "directed",
) -> list[dict]:
    """Return original + closure edges, each tagged with ``weight`` and ``hops``.

    Args:
        edges: iterable of ``{"src": int, "dst": int, "subtype": str, ...}``.
            Only ``src``/``dst`` are required. Stored direction is ``src -> dst``.
        decay: per-hop multiplier in ``(0, 1]``. A min-hop-``h`` edge gets weight
            ``decay ** (h - 1)``: direct edges (``h=1``) -> ``1.0``, one
            transitive step (``h=2``) -> ``decay``, two steps -> ``decay**2``,
            etc. ``decay=1.0`` makes every inherited edge weight ``1.0`` (full
            inheritance — ancestors' labels pass to descendants unweighted).
        max_hops: cap on path length in edges. ``0`` = unlimited.
        n_tokens: optional sequence length; endpoints outside ``[0, n_tokens)``
            are dropped first (mirrors the dataset bounds check).
        mode: ``"directed"`` (default) follows edge direction — only descendants
            inherit from ancestors (``a->b, b->c => a->c``). ``"undirected"``
            ignores direction and links *every pair of tokens in the same
            connected component*, with ``hops`` = the undirected graph distance
            (so ``a->b, b->c, a->d`` also links ``d`` and ``c``). Inherited edges
            are emitted position-ordered (``src < dst``) regardless of mode.

    Returns:
        list of ``{"src", "dst", "subtype", "weight", "hops"}``. Original edges
        keep their subtype with ``weight=1.0``/``hops=1``; inherited edges get
        ``subtype="transitive"``. When a pair is both a direct edge and reachable
        through the closure, the direct edge wins (weight ``1.0``).
    """
    undirected = (mode == "undirected")
    # Normalize + dedupe direct edges; keep the first subtype seen per (src,dst).
    direct: dict[tuple[int, int], str] = {}
    adj: dict[int, set[int]] = {}
    for e in edges:
        a = int(e["src"])
        b = int(e["dst"])
        if a == b:
            continue
        if n_tokens is not None and not (0 <= a < n_tokens and 0 <= b < n_tokens):
            continue
        key = (a, b)
        if key not in direct:
            direct[key] = e.get("subtype", "edge")
        adj.setdefault(a, set()).add(b)
        if undirected:
            adj.setdefault(b, set()).add(a)

    # Min-hop BFS from every node. BFS visit order gives the minimum number of
    # edges (hops) to each reachable node; with decay <= 1 the min-hop path is
    # also the max-weight path. Pairs are canonicalized to (min, max) so that an
    # undirected connection is emitted once and stays position-ordered.
    best_hops: dict[tuple[int, int], int] = {}
    for s in adj:
        dist = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if max_hops and du >= max_hops:
                continue
            for v in adj.get(u, ()):  # noqa: SIM118 — dict.get default
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
        for t, h in dist.items():
            if h <= 0:
                continue
            k = (s, t) if s < t else (t, s)
            if k not in best_hops or h < best_hops[k]:
                best_hops[k] = h

    out: list[dict] = []
    for (a, b), sub in direct.items():
        out.append({"src": a, "dst": b, "subtype": sub, "weight": 1.0, "hops": 1})
    for (a, c), h in best_hops.items():
        if (a, c) in direct or h <= 1:
            continue
        out.append({
            "src": a,
            "dst": c,
            "subtype": "transitive",
            "weight": float(decay ** (h - 1)),
            "hops": int(h),
        })
    return out


def node_target_weights(
    aug_edges: Iterable[dict],
    target_positions: Iterable[int],
) -> dict[int, float]:
    """Max augmented-edge weight from each non-target node into any target token.

    Used by weight-aware cfmask: a context token that connects to a target
    directly (weight ``1.0``) is "first-order related" and is never masked;
    transitively/component-connected tokens get their decayed weight; tokens with
    no path to any target are absent from the result (treated as weight ``0`` ->
    highest mask probability).

    Checks both endpoints so it works for ``directed`` edges (target is always the
    ``dst``) and ``undirected`` edges (the target may be either endpoint). Edges
    between two targets are ignored.
    """
    tset = {int(t) for t in target_positions}
    w: dict[int, float] = {}
    for e in aug_edges:
        s = int(e["src"])
        d = int(e["dst"])
        ww = float(e.get("weight", 1.0))
        s_t, d_t = s in tset, d in tset
        if d_t and not s_t:
            node = s
        elif s_t and not d_t:
            node = d
        else:
            continue
        if ww > w.get(node, 0.0):
            w[node] = ww
    return w
