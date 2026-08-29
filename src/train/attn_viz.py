from __future__ import annotations

import html as html_lib
import json
import logging
import os
from pathlib import Path

import torch
import transformers
import wandb

logger = logging.getLogger(__name__)


# ── Token cleaning ────────────────────────────────────────────────────────────

_TOK_REPLACEMENTS = [
    ("Ġ", " "),   # BPE space prefix
    ("Ċ", "\n"),  # BPE newline
    ("ĉ", "\t"),  # BPE tab
    ("▁", " "),   # SentencePiece space prefix
]

def _clean_token(tok: str) -> str:
    for special, replacement in _TOK_REPLACEMENTS:
        tok = tok.replace(special, replacement)
    return tok


# ── HTML renderer ─────────────────────────────────────────────────────────────

def _render_attn_html(
    tokens: list[str],
    attn_matrix: list[list[float]],          # [T, T]: current attn, attn_matrix[j][i] = j→i
    annot_pairs: list[tuple[int, int]],      # red  — ground-truth symbolic correlations
    title: str,
    top_k: int = 20,
    init_attn_matrix: list[list[float]] | None = None,   # step-0 full attn matrix
    init_top_pairs: list[tuple[int, int]] | None = None, # step-0 top-K arcs
) -> str:
    T = len(tokens)

    tok_spans = []
    for i, tok in enumerate(tokens):
        clean    = _clean_token(tok)
        stripped = clean.lstrip(" \t\n")
        prefix   = clean[: len(clean) - len(stripped)]
        display  = html_lib.escape(stripped) if stripped else "&#x200b;"
        label    = html_lib.escape(clean, quote=True)
        tok_spans.append(
            f'{prefix}<span class="tok" data-idx="{i}" data-label="{label}">{display}</span>'
        )
    code_html       = "".join(tok_spans)
    attn_json       = json.dumps(attn_matrix)
    annot_json      = json.dumps(annot_pairs)
    init_pairs_json = json.dumps(init_top_pairs or [])
    init_attn_json  = json.dumps(init_attn_matrix or [])  # full matrix for two-col panel
    title_safe      = html_lib.escape(title)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title_safe}</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@600;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:      #f4f3ef;
    --surface: #ffffff;
    --surface2:#f0eff9;
    --border:  #dddbe8;
    --text:    #1a1b2e;
    --dim:     #6b7280;
    --accent:  #4263eb;
    --red:     #ef4444;
    --yellow:  #f59e0b;
    --code-bg: #1e1e2e;
    --code-fg: #cdd6f4;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'JetBrains Mono', monospace;
    height: 100vh; display: flex; flex-direction: column;
    overflow: hidden; background: var(--bg); color: var(--text);
  }}
  header {{
    flex-shrink: 0; display: flex; align-items: center; gap: 16px;
    padding: 10px 20px; background: var(--surface);
    border-bottom: 1px solid var(--border);
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
  }}
  header h1 {{
    font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 800;
    letter-spacing: .08em; text-transform: uppercase; color: var(--accent);
  }}
  .stats {{ font-size: 11px; color: var(--dim); }}
  .layout {{ display: flex; flex: 1; overflow: hidden; }}
  aside {{
    width: 260px; flex-shrink: 0; background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column; overflow-y: auto;
  }}
  .sec {{ padding: 13px 15px; border-bottom: 1px solid var(--border); flex-shrink: 0; }}
  .sec-title {{
    font-size: 9px; font-weight: 700; letter-spacing: .15em;
    text-transform: uppercase; color: var(--dim); margin-bottom: 10px;
  }}
  .topk-row {{ display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--text); }}
  input[type=range] {{ flex: 1; accent-color: var(--accent); }}
  #topk-val {{ min-width: 24px; text-align: right; font-weight: 700; color: var(--accent); }}
  .legend-row {{
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; color: var(--text); margin-bottom: 6px;
  }}
  .dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}
  #info-panel {{ flex: 1; padding: 14px 15px; overflow-y: auto; }}
  .ph {{ font-size: 11px; color: var(--dim); line-height: 1.7; }}
  .info-name {{
    font-size: 15px; font-weight: 700; color: var(--accent);
    margin-bottom: 3px; word-break: break-all;
  }}
  .info-count {{ font-size: 10px; color: var(--dim); margin-bottom: 8px; }}
  /* Two-column comparison table */
  .cmp-table {{ width: 100%; border-collapse: collapse; margin-top: 4px; }}
  .cmp-head th {{
    font-size: 9px; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; padding: 3px 5px 6px;
    text-align: left; border-bottom: 1px solid var(--border);
  }}
  .cmp-head .th-init  {{ color: var(--yellow); }}
  .cmp-head .th-curr  {{ color: var(--accent); }}
  .cmp-row td {{
    font-size: 10px; padding: 3px 5px; vertical-align: top;
    border-bottom: 1px solid rgba(0,0,0,.04);
  }}
  .cmp-row:hover td {{ background: var(--surface2); }}
  .cmp-rank {{ color: var(--dim); font-size: 9px; min-width: 18px; flex-shrink: 0; }}
  .cmp-cell {{
    display: flex; align-items: baseline; gap: 4px;
  }}
  .cmp-tok  {{ flex: 1; font-weight: 600; word-break: break-all; overflow: hidden;
               text-overflow: ellipsis; white-space: nowrap; max-width: 70px; }}
  .cmp-w    {{ color: var(--dim); font-size: 9px; flex-shrink: 0; }}
  .cmp-up   {{ color: #22c55e; font-size: 9px; }}
  .cmp-dn   {{ color: #ef4444; font-size: 9px; }}
  main {{ flex: 1; overflow: auto; background: var(--code-bg); position: relative; }}
  #code-pre {{
    font-family: 'JetBrains Mono', monospace; font-size: 13.5px; line-height: 2;
    padding: 24px 36px; color: var(--code-fg);
    white-space: pre-wrap; word-break: break-all; tab-size: 4; margin: 0; min-height: 100%;
  }}
  .tok {{ border-radius: 3px; cursor: pointer; transition: background .1s, box-shadow .1s; }}
  .tok:hover {{
    background: rgba(66,99,235,.30) !important;
    box-shadow: 0 0 0 1px rgba(66,99,235,.5) !important;
  }}
  .tok.selected {{
    background: rgba(66,99,235,.60) !important;
    box-shadow: 0 0 0 1.5px #4263eb !important;
    color: #fff !important;
  }}
  .tok.annot-src {{ box-shadow: 0 0 0 1.5px var(--red) !important; }}
  #arrow-svg {{
    position: fixed; inset: 0; width: 100vw; height: 100vh;
    pointer-events: none; z-index: 999;
  }}
  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-thumb {{ background: #3a3a52; border-radius: 3px; }}
  aside::-webkit-scrollbar-thumb {{ background: var(--border); }}
</style>
</head>
<body>

<header>
  <h1>{title_safe}</h1>
  <span class="stats" id="stats-label"></span>
</header>

<div class="layout">
  <aside>
    <div class="sec">
      <div class="sec-title">Top-K attention</div>
      <div class="topk-row">
        <span>K =</span>
        <input type="range" id="topk-slider" min="1" max="{min(T, 50)}" value="{min(top_k, T)}">
        <span id="topk-val">{min(top_k, T)}</span>
      </div>
    </div>
    <div class="sec">
      <div class="sec-title">Legend</div>
      <div class="legend-row">
        <div class="dot" style="background:rgba(66,99,235,0.8)"></div>
        <span>Attention weight (query → key)</span>
      </div>
      <div class="legend-row">
        <div class="dot" style="background:var(--red)"></div>
        <span>Ground-truth correlation</span>
      </div>
      <p style="font-size:9px;color:var(--dim);margin-top:8px;line-height:1.5">
        Click token → see top-K attention.<br>
        Click again or background → clear.
      </p>
    </div>
    <div id="info-panel">
      <p class="ph">Click any token to see its top-K attention weights.</p>
    </div>
  </aside>

  <main id="code-area">
    <pre id="code-pre">{code_html}</pre>
  </main>
</div>

<svg id="arrow-svg" xmlns="http://www.w3.org/2000/svg">
  <defs id="arrow-defs"></defs>
</svg>

<script>
const ATTN        = {attn_json};
const INIT_ATTN   = {init_attn_json};  // full step-0 attn matrix (may be empty [])
const ANNOT_PAIRS = {annot_json};
const INIT_PAIRS  = {init_pairs_json};
const T           = ATTN.length;

document.getElementById('stats-label').textContent =
  `${{T}} tokens · ${{ANNOT_PAIRS.length}} annotation pairs`;

// ── SVG helpers ───────────────────────────────────────────────────────────────
const NS       = 'http://www.w3.org/2000/svg';
const arrowSvg = document.getElementById('arrow-svg');
const defs     = document.getElementById('arrow-defs');

function ensureMarker(id, color) {{
  if (defs.querySelector('#' + id)) return;
  const m = document.createElementNS(NS, 'marker');
  m.setAttribute('id', id);
  m.setAttribute('viewBox', '0 -5 10 10');
  m.setAttribute('refX', '9'); m.setAttribute('refY', '0');
  m.setAttribute('markerWidth', '5'); m.setAttribute('markerHeight', '5');
  m.setAttribute('orient', 'auto');
  const p = document.createElementNS(NS, 'path');
  p.setAttribute('d', 'M0,-5L10,0L0,5Z');
  p.setAttribute('fill', color);
  m.appendChild(p); defs.appendChild(m);
}}
ensureMarker('arr-attn',  '#4263eb');
ensureMarker('arr-annot', '#ef4444');
ensureMarker('arr-init',  '#f59e0b');

function appendArrow(x1, y1, x2, y2, cls, markerId, color, opacity, width) {{
  const dx = x2-x1, dy = y2-y1, len = Math.sqrt(dx*dx+dy*dy) || 1;
  const bend = Math.min(len*0.28, 55);
  const cx = (x1+x2)/2 - (dy/len)*bend;
  const cy = (y1+y2)/2 + (dx/len)*bend;
  const p = document.createElementNS(NS, 'path');
  p.classList.add('arrow', cls);
  p.setAttribute('d',              `M${{x1}},${{y1}} Q${{cx}},${{cy}} ${{x2}},${{y2}}`);
  p.setAttribute('stroke',         color);
  p.setAttribute('stroke-width',   String(width));
  p.setAttribute('fill',           'none');
  p.setAttribute('stroke-opacity', String(opacity));
  p.setAttribute('marker-end',     `url(#${{markerId}})`);
  arrowSvg.appendChild(p);
}}

// Pass a class string to remove only that subset, or nothing to remove all.
function clearArrows(cls) {{
  const sel = cls ? `path.arrow.${{cls}}` : 'path.arrow';
  arrowSvg.querySelectorAll(sel).forEach(el => el.remove());
}}

// ── Token helpers ─────────────────────────────────────────────────────────────
function spanByIdx(i) {{
  return document.querySelector(`.tok[data-idx="${{i}}"]`);
}}
function midpoint(el) {{
  const r = el.getBoundingClientRect();
  return [r.left + r.width/2, r.top + r.height/2];
}}

// ── State ─────────────────────────────────────────────────────────────────────
let activeIdx = null;
let topK      = parseInt(document.getElementById('topk-slider').value, 10);

// ── Annotation arcs — redrawn after every clearArrows() ───────────────────────
function drawAnnotArrows() {{
  clearArrows('annot-arrow');
  ANNOT_PAIRS.forEach(([qi, qj]) => {{
    const src = spanByIdx(qi), dst = spanByIdx(qj);
    if (!src || !dst) return;
    const [x1, y1] = midpoint(src);
    const [x2, y2] = midpoint(dst);
    const p = appendArrow(x1, y1, x2, y2, 'annot-arrow', 'arr-annot', '#ef4444', 0.55, 1.4);
  }});
  arrowSvg.querySelectorAll('path.annot-arrow').forEach(el => {{
    el.setAttribute('stroke-dasharray', '4 3');
  }});
}}

// ── Attention arrows for the selected token ───────────────────────────────────
function drawAttnArrows(j) {{
  clearArrows('attn-arrow');
  const row     = ATTN[j];
  const indexed = row.map((w, i) => [i, w]).filter(([i]) => i !== j);
  indexed.sort((a, b) => b[1] - a[1]);
  const pairs = indexed.slice(0, topK);
  const maxW  = pairs.length > 0 ? pairs[0][1] : 1;

  const srcEl = spanByIdx(j);
  if (!srcEl) return;
  const [x1, y1] = midpoint(srcEl);

  pairs.forEach(([i, w]) => {{
    const dstEl = spanByIdx(i);
    if (!dstEl) return;
    const norm    = maxW > 0 ? w / maxW : 0;
    const opacity = 0.15 + 0.75 * norm;
    const strokeW = 0.8  + 2.2  * norm;
    const alpha   = (0.1 + 0.7  * norm).toFixed(3);
    const [x2, y2] = midpoint(dstEl);
    dstEl.style.background = `rgba(66,99,235,${{alpha}})`;
    dstEl.style.boxShadow  = `0 0 0 1px rgba(66,99,235,${{(norm*0.8+0.1).toFixed(3)}})`;
    appendArrow(x1, y1, x2, y2, 'attn-arrow', 'arr-attn', '#4263eb', opacity, strokeW);
  }});
}}

// ── Info panel — two-column comparison ───────────────────────────────────────
function updateInfoPanel(j) {{
  const label = spanByIdx(j)?.dataset.label ?? String(j);
  const safeLabel = label.replace(/&/g, '&amp;').replace(/</g, '&lt;');

  // Current model top-K
  const currRow     = ATTN[j];
  const currIndexed = currRow.map((w, i) => [i, w]).filter(([i]) => i !== j);
  currIndexed.sort((a, b) => b[1] - a[1]);
  const currPairs = currIndexed.slice(0, topK);

  // Init model top-K (may be unavailable)
  const hasInit = INIT_ATTN.length > 0;
  let initPairs = [];
  if (hasInit) {{
    const initRow     = INIT_ATTN[j];
    const initIndexed = initRow.map((w, i) => [i, w]).filter(([i]) => i !== j);
    initIndexed.sort((a, b) => b[1] - a[1]);
    initPairs = initIndexed.slice(0, topK);
  }}

  // Build rank lookup for delta annotation
  const currRankOf = {{}};
  currPairs.forEach(([i], rank) => {{ currRankOf[i] = rank + 1; }});
  const initRankOf = {{}};
  initPairs.forEach(([i], rank) => {{ initRankOf[i] = rank + 1; }});

  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const tokLabel = i => esc(spanByIdx(i)?.dataset.label ?? String(i));

  // Merge both lists for unified row count
  const maxRows = Math.max(currPairs.length, initPairs.length);
  let bodyRows = '';
  for (let r = 0; r < maxRows; r++) {{
    const [ci, cw] = currPairs[r] ?? [null, null];
    const [ii, iw] = initPairs[r] ?? [null, null];

    // Delta: how did this token's rank change from init to current?
    const rankDelta = (ci !== null && initRankOf[ci] !== undefined)
      ? initRankOf[ci] - (r + 1)   // positive = moved up
      : null;
    const deltaHtml = rankDelta === null ? ''
      : rankDelta > 0 ? `<span class="cmp-up">▲${{rankDelta}}</span>`
      : rankDelta < 0 ? `<span class="cmp-dn">▼${{Math.abs(rankDelta)}}</span>`
      : '';

    const initCell = ii !== null
      ? `<div class="cmp-cell">
           <span class="cmp-rank">#${{r+1}}</span>
           <span class="cmp-tok" title="${{tokLabel(ii)}}">${{tokLabel(ii)}}</span>
           <span class="cmp-w">${{(iw*100).toFixed(1)}}%</span>
         </div>`
      : '';
    const currCell = ci !== null
      ? `<div class="cmp-cell">
           <span class="cmp-rank">#${{r+1}}</span>
           <span class="cmp-tok" title="${{tokLabel(ci)}}">${{tokLabel(ci)}}</span>
           <span class="cmp-w">${{(cw*100).toFixed(1)}}%</span>
           ${{deltaHtml}}
         </div>`
      : '';

    bodyRows += `<tr class="cmp-row"><td>${{initCell}}</td><td>${{currCell}}</td></tr>`;
  }}

  const initHeader = hasInit
    ? `<th class="th-init">● Before training</th>`
    : `<th class="th-init" style="color:var(--dim)">● Before (n/a)</th>`;

  document.getElementById('info-panel').innerHTML = `
    <div class="info-name">#${{j}} ${{safeLabel}}</div>
    <div class="info-count">top-${{topK}} attention keys</div>
    <table class="cmp-table">
      <thead><tr class="cmp-head">
        ${{initHeader}}
        <th class="th-curr">● Current</th>
      </tr></thead>
      <tbody>${{bodyRows}}</tbody>
    </table>`;
}}

// ── Clear all state ───────────────────────────────────────────────────────────
function clearAll() {{
  activeIdx = null;
  clearArrows();    // remove every arrow
  document.querySelectorAll('.tok').forEach(s => {{
    s.classList.remove('selected');
    s.style.background = '';
    s.style.boxShadow  = '';
  }});
  ANNOT_PAIRS.forEach(([qi]) => {{
    const el = spanByIdx(qi);
    if (el) el.classList.add('annot-src');
  }});
  document.getElementById('info-panel').innerHTML =
    '<p class="ph">Click any token to see its top-K attention weights.</p>';
  drawAnnotArrows();
  drawInitArrows();
}}

// ── Refresh after any state change ───────────────────────────────────────────
function refresh() {{
  if (activeIdx === null) {{ clearAll(); return; }}
  document.querySelectorAll('.tok').forEach(s => {{
    s.classList.remove('selected');
    s.style.background = '';
    s.style.boxShadow  = '';
  }});
  const sel = spanByIdx(activeIdx);
  if (sel) sel.classList.add('selected');
  clearArrows();               // wipe everything first
  drawAttnArrows(activeIdx);   // blue attn arrows
  drawAnnotArrows();           // red annot arcs
  updateInfoPanel(activeIdx);
}}

// ── Events ────────────────────────────────────────────────────────────────────
document.querySelectorAll('.tok').forEach(span => {{
  span.addEventListener('click', e => {{
    e.stopPropagation();
    const idx = parseInt(span.dataset.idx, 10);
    if (activeIdx === idx) {{ clearAll(); return; }}   // second click = toggle off
    activeIdx = idx;
    refresh();
  }});
}});

document.getElementById('code-area').addEventListener('click', e => {{
  if (!e.target.classList.contains('tok')) clearAll();
}});

// Redraw on scroll — getBoundingClientRect values change as user scrolls
document.getElementById('code-area').addEventListener('scroll', () => {{
  clearArrows();
  drawAnnotArrows();
  drawInitArrows();
  if (activeIdx !== null) drawAttnArrows(activeIdx);
}}, {{ passive: true }});

document.getElementById('topk-slider').addEventListener('input', e => {{
  topK = parseInt(e.target.value, 10);
  document.getElementById('topk-val').textContent = topK;
  if (activeIdx !== null) refresh();
}});

// ── Init ──────────────────────────────────────────────────────────────────────
ANNOT_PAIRS.forEach(([qi]) => {{
  const el = spanByIdx(qi);
  if (el) el.classList.add('annot-src');
}});
requestAnimationFrame(() => {{ drawAnnotArrows(); drawInitArrows(); }});
</script>
</body>
</html>"""


# ── Public builder ────────────────────────────────────────────────────────────

def build_attn_html(
    tokens: list[str],
    attn_matrix,
    annot_pairs: list[tuple[int, int]],
    title: str = "Attention Viewer",
    top_k: int = 20,
    init_attn_matrix=None,
    init_top_pairs: list[tuple[int, int]] | None = None,
) -> str:
    T   = len(tokens)
    def to_list(m):
        if m is None: return None
        mat = m.tolist() if hasattr(m, "tolist") else [list(r) for r in m]
        return [row[:T] for row in mat[:T]]
    return _render_attn_html(tokens, to_list(attn_matrix), list(annot_pairs), title,
                             top_k=top_k,
                             init_attn_matrix=to_list(init_attn_matrix),
                             init_top_pairs=init_top_pairs)


# ── Trainer callback ──────────────────────────────────────────────────────────

class AttentionVisualizationCallback(transformers.TrainerCallback):
    def __init__(self, dataset,
                 tokenizer,
                 output_dir: str,
                 num_samples: int = 10,
                 top_k: int = 100):
        self.dataset     = dataset
        self.tokenizer   = tokenizer
        self.output_dir  = output_dir
        self.num_samples = num_samples
        self.top_k       = top_k
        # Populated in on_train_begin: sample_idx → full attn_map [T, T]
        self._init_attn: dict[int, "np.ndarray"] = {}
        self._init_top_pairs: dict[int, list[tuple[int, int]]] = {}

    def _run_forward(self, model, item) -> "np.ndarray":
        """Run a forward pass and return attn_avg [T, T] (last 4 layers, avg heads)."""
        import numpy as np
        device    = next(model.parameters()).device
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        attn_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_attentions=True,
            )
        last4    = outputs.attentions[-4:]
        attn_avg = torch.stack(last4, dim=0).mean(0).mean(1)  # [1, T, T]
        return attn_avg[0].float().cpu().numpy()               # [T, T]

    def _top_k_pairs(self, attn_map: "np.ndarray") -> list[tuple[int, int]]:
        """Return the global top-K (qi, qj) pairs by attention weight across all rows."""
        T = attn_map.shape[0]
        # Flatten, exclude diagonal (self-attention), take top-K
        pairs = []
        for j in range(T):
            row     = attn_map[j]
            indexed = [(i, row[i]) for i in range(T) if i != j and j > i]  # causal only
            indexed.sort(key=lambda x: -x[1])
            for i, _ in indexed[:self.top_k]:
                pairs.append((i, j))   # (qi, qj) convention matching annot_pairs
        # deduplicate and take global top-K by weight
        pair_weights = [(i, j, attn_map[j, i]) for (i, j) in pairs]
        pair_weights.sort(key=lambda x: -x[2])
        seen = set()
        result = []
        for i, j, _ in pair_weights:
            key = (i, j)
            if key not in seen:
                seen.add(key)
                result.append(key)
            if len(result) >= self.top_k:
                break
        return result

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Snapshot full attention matrix before any training steps."""
        if not state.is_world_process_zero or model is None:
            return
        model.eval()
        for sample_idx in range(min(self.num_samples, len(self.dataset))):
            item     = self.dataset[sample_idx]
            attn_map = self._run_forward(model, item)
            self._init_attn[sample_idx]      = attn_map
            self._init_top_pairs[sample_idx] = self._top_k_pairs(attn_map)
            logger.info(f"Captured step-0 attention for sample {sample_idx}")
        model.train()

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if model is None:
            return

        device = next(model.parameters()).device
        model.eval()
        os.makedirs(self.output_dir, exist_ok=True)

        for sample_idx in range(min(self.num_samples, len(self.dataset))):
            item        = self.dataset[sample_idx]
            annot_pairs = item["annot_pairs"].tolist()
            attn_map    = self._run_forward(model, item)
            tokens      = self.tokenizer.convert_ids_to_tokens(
                              item["input_ids"].tolist())

            html_str = build_attn_html(
                tokens=tokens,
                attn_matrix=attn_map,
                annot_pairs=annot_pairs,
                title=f"Attention · step {state.global_step} · sample {sample_idx}",
                top_k=self.top_k,
                init_attn_matrix=self._init_attn.get(sample_idx),
                init_top_pairs=self._init_top_pairs.get(sample_idx),
            )

            save_path = os.path.join(
                self.output_dir,
                f"attn_step{state.global_step:06d}_s{sample_idx}.html",
            )
            Path(save_path).write_text(html_str, encoding="utf-8")
            logger.info(f"Saved attention viewer → {save_path}")

            if wandb.run is not None:
                artifact = wandb.Artifact(
                    name=f"attn_s{sample_idx}_step{state.global_step}",
                    type="attention_viz",
                )
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)
                wandb.log(
                    {f"attn_viz/sample_{sample_idx}": wandb.Html(save_path)},
                    step=state.global_step,
                )

        model.train()