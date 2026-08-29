

import os
import json
import webbrowser
import tempfile
import html as html_lib
from pathlib import Path
import sys
import dataclasses



SUBTYPE_COLORS: dict[str, str] = {
    "bracket": "#64748b",  # slate  — 括号匹配
    "defuse": "#0ea5e9",  # sky    — def-use
    "call": "#f97316",  # orange — 函数调用
    "return": "#ef4444",  # red    — return
    "type": "#22c55e",  # green  — 类型标注
    "dataflow": "#fb923c",  # amber  — 数据流
    "semantic": "#a855f7",  # purple — 语义
    "api": "#ec4899",  # pink   — API
    "": "#94a3b8",  # grey   — fallback
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def _build_graph_data(correlations, subwords=None) -> dict:
    # Node key = token_idx (int). Every unique occurrence is a separate node.
    node_set: dict[int, dict] = {}
    links: list[dict] = []

    if subwords:
        for i, sw in enumerate(subwords):
            node_set[i] = {
                "id": i,
                "label": sw.clean,
                "char_start": sw.char_start,
                "char_end": sw.char_end,
            }

    for c in correlations:
        if c.token_i_idx == -1 or c.token_j_idx == -1:
            continue  # skip correlations without resolved indices
        if c.token_i_idx not in node_set:
            node_set[c.token_i_idx] = {"id": c.token_i_idx, "label": c.token_i,
                                       "char_start": -1, "char_end": -1}
        if c.token_j_idx not in node_set:
            node_set[c.token_j_idx] = {"id": c.token_j_idx, "label": c.token_j,
                                       "char_start": -1, "char_end": -1}
        links.append({
            "source": c.token_i_idx,
            "target": c.token_j_idx,
            "subtype": c.subtype or "unknown",
            "source_label": c.source,
            "color": SUBTYPE_COLORS.get(c.subtype, SUBTYPE_COLORS[""]),
        })

    return {"nodes": list(node_set.values()), "links": links}


def _annotate_code_html(code: str, nodes: list[dict], subwords=None) -> str:
    """
    Wrap every token in a <span data-idx=N> so the JS can look them up by index.
    When subwords is provided every token gets a span; otherwise only nodes with
    known char positions are spanned.
    """
    if subwords:
        occurrences = [
            (sw.char_start, sw.char_end, sw.clean, i)
            for i, sw in enumerate(subwords)
            if sw.char_end > sw.char_start
        ]
    else:
        occurrences = [
            (node["char_start"], node["char_end"], node["label"], node["id"])
            for node in nodes
            if node.get("char_start", -1) >= 0
        ]

    occurrences.sort(key=lambda x: x[0])

    # Deduplicate overlapping spans (shouldn't happen with simple tokenizer, but guard anyway)
    filtered: list[tuple[int, int, str, int]] = []
    last_end = 0
    for start, end, label, idx in occurrences:
        if start >= last_end and end > start:
            filtered.append((start, end, label, idx))
            last_end = end

    parts: list[str] = []
    cursor = 0
    for start, end, label, idx in filtered:
        if cursor < start:
            parts.append(html_lib.escape(code[cursor:start]))
        safe_label = html_lib.escape(label, quote=True)
        parts.append(
            f'<span class="tok" data-idx="{idx}" data-label="{safe_label}">'
            f'{html_lib.escape(code[start:end])}</span>'
        )
        cursor = end
    if cursor < len(code):
        parts.append(html_lib.escape(code[cursor:]))

    return "".join(parts)


# ── HTML renderer ─────────────────────────────────────────────────────────────

def _render_html(graph_data: dict, title: str, code_html: str = "") -> str:
    clean_graph = {
        "nodes": [{"id": n["id"], "label": n["label"]} for n in graph_data["nodes"]],
        "links": graph_data["links"],
    }
    graph_json = json.dumps(clean_graph, ensure_ascii=False)
    subtypes_json = json.dumps([st for st in SUBTYPE_COLORS if st])

    legend_rows = "\n".join(
        f'<label class="legend-item">'
        f'<input type="checkbox" class="subtype-cb" value="{st}" checked>'
        f'<span class="dot" style="background:{color}"></span>{st}'
        f'</label>'
        for st, color in SUBTYPE_COLORS.items() if st
    )
    legend_rows += '\n<p style="font-size:9px;color:var(--dim);margin-top:8px;line-height:1.5">Click once to solo · click again to restore all</p>'

    return f"""
    <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
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
            --code-bg: #1e1e2e;
            --code-fg: #cdd6f4;
          }}
          * {{ box-sizing: border-box; margin: 0; padding: 0; }}

          body {{
            font-family: 'JetBrains Mono', monospace;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            background: var(--bg);
            color: var(--text);
          }}

          /* ── Header ─────────────────────────────────────────── */
          header {{
            flex-shrink: 0;
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 10px 20px;
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            box-shadow: 0 1px 4px rgba(0,0,0,.06);
          }}
          header h1 {{
            font-family: 'Syne', sans-serif;
            font-size: 13px;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            color: var(--accent);
          }}
          .stats {{ font-size: 11px; color: var(--dim); }}

          /* ── Main layout ─────────────────────────────────────── */
          .layout {{ display: flex; flex: 1; overflow: hidden; }}

          /* ── Sidebar ─────────────────────────────────────────── */
          aside {{
            width: 260px;
            flex-shrink: 0;
            background: var(--surface);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow-y: auto;
          }}
          .sec {{
            padding: 13px 15px;
            border-bottom: 1px solid var(--border);
            flex-shrink: 0;
          }}
          .sec-title {{
            font-size: 9px;
            font-weight: 700;
            letter-spacing: .15em;
            text-transform: uppercase;
            color: var(--dim);
            margin-bottom: 10px;
          }}

          /* Legend */
          .legend-item {{
            display: flex;
            align-items: center;
            gap: 7px;
            font-size: 11px;
            color: var(--text);
            margin-bottom: 7px;
            cursor: pointer;
            user-select: none;
          }}
          .legend-item input {{ accent-color: var(--accent); cursor: pointer; }}
          .dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}

          /* Direction */
          .dir-group {{ display: flex; gap: 6px; }}
          .dir-btn {{
            flex: 1;
            background: var(--surface2);
            border: 1px solid var(--border);
            color: var(--dim);
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            padding: 5px 0;
            border-radius: 4px;
            cursor: pointer;
            text-align: center;
            transition: all .15s;
          }}
          .dir-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
          .dir-btn.active {{
            background: var(--accent);
            border-color: var(--accent);
            color: #fff;
            font-weight: 600;
          }}

          /* Info panel */
          #info-panel {{
            flex: 1;
            padding: 14px 15px;
            overflow-y: auto;
          }}
          .ph {{ font-size: 11px; color: var(--dim); line-height: 1.7; }}
          .info-name {{
            font-size: 15px;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 3px;
            word-break: break-all;
          }}
          .info-count {{
            font-size: 10px;
            color: var(--dim);
            margin-bottom: 10px;
          }}
          .edge-row {{
            display: flex;
            align-items: center;
            gap: 7px;
            padding: 4px 8px;
            background: var(--surface2);
            border-left: 3px solid;
            border-radius: 0 4px 4px 0;
            margin-bottom: 4px;
            font-size: 11px;
          }}
          .edge-dir  {{ color: var(--dim); font-size: 10px; flex-shrink: 0; }}
          .edge-tok  {{ flex: 1; color: var(--text); font-weight: 600; word-break: break-all; }}
          .edge-sub  {{ color: var(--dim); font-size: 10px; flex-shrink: 0; }}

          /* ── Code area ───────────────────────────────────────── */
          main {{
            flex: 1;
            overflow: auto;
            background: var(--code-bg);
            position: relative;
          }}
          #code-pre {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 13.5px;
            line-height: 2;
            padding: 24px 36px;
            color: var(--code-fg);
            white-space: pre;
            tab-size: 4;
            margin: 0;
            min-height: 100%;
          }}

          /* ── Token spans ─────────────────────────────────────── */
          .tok {{
            border-radius: 3px;
            cursor: pointer;
            transition: background .1s, box-shadow .1s;
          }}
          .tok:hover {{ background: rgba(66,99,235,.30) !important; box-shadow: 0 0 0 1px rgba(66,99,235,.5) !important; }}
          .tok.selected {{
            background: rgba(66,99,235,.60) !important;
            box-shadow: 0 0 0 1.5px #4263eb !important;
            color: #fff !important;
          }}
          /* Connected tokens use inline style for per-edge colour */

          /* ── SVG arrow overlay (fixed, full-viewport) ───────── */
          #arrow-svg {{
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100vh;
            pointer-events: none;
            z-index: 999;
          }}

          ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
          ::-webkit-scrollbar-thumb {{ background: #3a3a52; border-radius: 3px; }}
          aside::-webkit-scrollbar-thumb {{ background: var(--border); }}
          #info-panel::-webkit-scrollbar-thumb {{ background: var(--border); }}
        </style>
        </head>
        <body>

        <header>
          <h1>{title}</h1>
          <span class="stats" id="stats-label"></span>
        </header>

        <div class="layout">

          <!-- ── Sidebar ── -->
          <aside>

            <div class="sec">
              <div class="sec-title">Edge types</div>
              {legend_rows}
            </div>

            <div class="sec">
              <div class="sec-title">Arrow direction</div>
              <div class="dir-group">
                <button class="dir-btn" data-dir="out" title="Show outgoing edges (selected → other)">→ Out</button>
                <button class="dir-btn active" data-dir="both" title="Show all edges">↔ Both</button>
                <button class="dir-btn" data-dir="in"  title="Show incoming edges (other → selected)">← In</button>
              </div>
            </div>

            <div id="info-panel">
              <p class="ph">Click a highlighted token in the code to inspect its correlations.</p>
            </div>

          </aside>

          <!-- ── Code view ── -->
          <main id="code-area">
            <pre id="code-pre">{code_html}</pre>
          </main>

        </div>

        <!-- Arrow overlay SVG -->
        <svg id="arrow-svg" xmlns="http://www.w3.org/2000/svg">
          <defs id="arrow-defs"></defs>
        </svg>

        <script>
        // ── Data ───────────────────────────────────────────────────────────────────
        const GRAPH       = {graph_json};
        const ALL_SUBTYPES = {subtypes_json};

        document.getElementById('stats-label').textContent =
          `${{GRAPH.nodes.length}} tokens · ${{GRAPH.links.length}} edges`;

        // idx (int) → label string, for display in info panel
        const nodeLabel = {{}};
        GRAPH.nodes.forEach(n => {{ nodeLabel[n.id] = n.label; }});

        // ── Adjacency map: idx -> [{{other: idx, color, subtype, dir}}] ─────────────
        // JS object keys are strings, but parseInt on dataset.idx gives us back ints —
        // we keep the keys as numeric strings (JS auto-converts) so adj[79] works fine.
        const adj = {{}};
        GRAPH.links.forEach(l => {{
          (adj[l.source] ??= []).push({{ other: l.target, color: l.color, subtype: l.subtype, dir: 'out' }});
          (adj[l.target] ??= []).push({{ other: l.source, color: l.color, subtype: l.subtype, dir: 'in'  }});
        }});

        // ── SVG arrow markers (one per unique colour) ──────────────────────────────
        const NS       = 'http://www.w3.org/2000/svg';
        const arrowSvg = document.getElementById('arrow-svg');
        const defs     = document.getElementById('arrow-defs');

        [...new Set(GRAPH.links.map(l => l.color))].forEach(color => {{
          const marker = document.createElementNS(NS, 'marker');
          marker.setAttribute('id',           'arr-' + color.replace('#', ''));
          marker.setAttribute('viewBox',      '0 -5 10 10');
          marker.setAttribute('refX',         '9');
          marker.setAttribute('refY',         '0');
          marker.setAttribute('markerWidth',  '5');
          marker.setAttribute('markerHeight', '5');
          marker.setAttribute('orient',       'auto');
          const path = document.createElementNS(NS, 'path');
          path.setAttribute('d',    'M0,-5L10,0L0,5Z');
          path.setAttribute('fill', color);
          marker.appendChild(path);
          defs.appendChild(marker);
        }});

        // ── State ──────────────────────────────────────────────────────────────────
        let activeIdx     = null;   // int idx of the selected token, or null
        let showDir       = 'both'; // 'out' | 'in' | 'both'
        let activeSubtypes = new Set(ALL_SUBTYPES);

        // ── Helpers ────────────────────────────────────────────────────────────────
        const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

        // Each idx maps to exactly ONE span (unique occurrence in source).
        function spanByIdx(idx) {{
          return document.querySelector(`.tok[data-idx="${{idx}}"]`);
        }}

        function visibleEdges() {{
          return (adj[activeIdx] || []).filter(e =>
            activeSubtypes.has(e.subtype) &&
            (showDir === 'both' || e.dir === showDir)
          );
        }}

        // ── Clear everything ───────────────────────────────────────────────────────
        function clearAll() {{
          activeIdx = null;
          document.querySelectorAll('.tok').forEach(s => {{
            s.classList.remove('selected');
            s.style.background = '';
            s.style.boxShadow  = '';
          }});
          arrowSvg.querySelectorAll('path.arrow').forEach(el => el.remove());
          document.getElementById('info-panel').innerHTML =
            '<p class="ph">Click a highlighted token in the code to inspect its correlations.</p>';
        }}

        // ── Main refresh (called after any state change) ───────────────────────────
        function refresh() {{
          if (activeIdx === null) return;
          applyHighlights();
          drawArrows();
          updateInfoPanel();
        }}

        // ── Global subtype view ────────────────────────────────────────────────────
        // When globalSubtype is set, show ALL edges of that type across all tokens,
        // regardless of which token is selected.
        let globalSubtype = null;  // null = off, string = active global view

        function drawGlobalEdges() {{
          arrowSvg.querySelectorAll('path.arrow').forEach(el => el.remove());
          document.querySelectorAll('.tok').forEach(s => {{
            s.classList.remove('selected');
            s.style.background = '';
            s.style.boxShadow  = '';
          }});

          if (!globalSubtype) return;

          const color = GRAPH.links.find(l => l.subtype === globalSubtype)?.color || '#94a3b8';
          const seen  = new Set();

          GRAPH.links.forEach(l => {{
            if (l.subtype !== globalSubtype) return;
            const key = `${{l.source}}-${{l.target}}`;
            if (seen.has(key)) return;
            seen.add(key);

            const fromEl = spanByIdx(l.source);
            const toEl   = spanByIdx(l.target);
            if (!fromEl || !toEl) return;

            // Highlight both endpoints
            fromEl.style.background = color + '4a';
            fromEl.style.boxShadow  = `0 0 0 1px ${{color}}99`;
            toEl.style.background   = color + '4a';
            toEl.style.boxShadow    = `0 0 0 1px ${{color}}99`;

            const fr = fromEl.getBoundingClientRect();
            const tr = toEl.getBoundingClientRect();
            appendArrow(
              fr.left + fr.width / 2, fr.top + fr.height / 2,
              tr.left + tr.width / 2, tr.top + tr.height / 2,
              color
            );
          }});

          // Update info panel with count
          const count = [...seen].length;
          document.getElementById('info-panel').innerHTML =
            `<div class="info-name">${{esc(globalSubtype)}}</div>` +
            `<div class="info-count">${{count}} edge${{count !== 1 ? 's' : ''}} across all tokens</div>` +
            `<p class="ph" style="margin-top:8px">Click a token to inspect its edges,<br>or click the edge type again to exit.</p>`;
        }}

        // ── Highlight tokens in code ───────────────────────────────────────────────
        function applyHighlights() {{
          document.querySelectorAll('.tok').forEach(s => {{
            s.classList.remove('selected');
            s.style.background = '';
            s.style.boxShadow  = '';
          }});

          const sel = spanByIdx(activeIdx);
          if (sel) sel.classList.add('selected');

          visibleEdges().forEach(edge => {{
            const s = spanByIdx(edge.other);
            if (s) {{
              s.style.background = edge.color + '3a';
              s.style.boxShadow  = `0 0 0 1px ${{edge.color}}88`;
            }}
          }});
        }}

        // ── Draw SVG arrows ────────────────────────────────────────────────────────
        function drawArrows() {{
          arrowSvg.querySelectorAll('path.arrow').forEach(el => el.remove());

          const fromEl = spanByIdx(activeIdx);
          if (!fromEl) return;

          visibleEdges().forEach(edge => {{
            const toEl = spanByIdx(edge.other);
            if (!toEl || toEl === fromEl) return;

            const fr = fromEl.getBoundingClientRect();
            const tr = toEl.getBoundingClientRect();

            const fx = fr.left + fr.width  / 2;
            const fy = fr.top  + fr.height / 2;
            const tx = tr.left + tr.width  / 2;
            const ty = tr.top  + tr.height / 2;

            const [x1, y1, x2, y2] = edge.dir === 'out'
              ? [fx, fy, tx, ty]
              : [tx, ty, fx, fy];

            appendArrow(x1, y1, x2, y2, edge.color);
          }});
        }}

        function appendArrow(x1, y1, x2, y2, color) {{
          const dx  = x2 - x1;
          const dy  = y2 - y1;
          const len = Math.sqrt(dx * dx + dy * dy) || 1;

          const bend = Math.min(len * 0.30, 60);
          const cx = (x1 + x2) / 2 - (dy / len) * bend;
          const cy = (y1 + y2) / 2 + (dx / len) * bend;

          const p = document.createElementNS(NS, 'path');
          p.classList.add('arrow');
          p.setAttribute('d',              `M${{x1}},${{y1}} Q${{cx}},${{cy}} ${{x2}},${{y2}}`);
          p.setAttribute('stroke',         color);
          p.setAttribute('stroke-width',   '1.8');
          p.setAttribute('fill',           'none');
          p.setAttribute('stroke-opacity', '0.72');
          p.setAttribute('marker-end',     `url(#arr-${{color.replace('#', '')}})`);
          arrowSvg.appendChild(p);
        }}

        // ── Info panel ─────────────────────────────────────────────────────────────
        function updateInfoPanel() {{
          const total = (adj[activeIdx] || []).length;
          const edges = visibleEdges();
          const rows  = edges.map(e => `
            <div class="edge-row" style="border-color:${{e.color}}">
              <span class="edge-dir">${{e.dir === 'out' ? '→' : '←'}}</span>
              <span class="edge-tok">${{esc(nodeLabel[e.other] ?? String(e.other))}}</span>
              <span class="edge-sub">${{esc(e.subtype)}}</span>
            </div>`).join('');

         document.getElementById('info-panel').innerHTML = `
            <div class="info-name">${{esc(label)}}</div>
            <div class="info-count">#${{activeIdx}} · ${{edges.length}} / ${{total}} edges visible</div>
            ${{rows || '<p class="ph" style="margin-top:6px">No edges match the current filters.</p>'}}`;
        }}

        // ── Event: token click ─────────────────────────────────────────────────────
        document.querySelectorAll('.tok').forEach(span => {{
          span.addEventListener('click', e => {{
            e.stopPropagation();
            // Exit global view if active
            if (globalSubtype !== null) {{
              globalSubtype = null;
              document.querySelectorAll('.subtype-cb').forEach(c => {{
                c.checked = true;
                c.style.outline = '';
              }});
              activeSubtypes = new Set(ALL_SUBTYPES);
            }}
            activeIdx = parseInt(span.dataset.idx, 10);
            refresh();
          }});
        }});

        // ── Event: background click → clear ───────────────────────────────────────
        document.getElementById('code-area').addEventListener('click', e => {{
          if (!e.target.classList.contains('tok')) clearAll();
        }});

        // ── Event: redraw arrows on scroll (token positions change) ───────────────
        document.getElementById('code-area').addEventListener('scroll', () => {{
          if (activeIdx !== null) drawArrows();
        }}, {{ passive: true }});

        // ── Event: direction toggle ────────────────────────────────────────────────
        document.querySelectorAll('.dir-btn').forEach(btn => {{
          btn.addEventListener('click', () => {{
            document.querySelectorAll('.dir-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            showDir = btn.dataset.dir;
            refresh();
          }});
        }});

        // ── Event: subtype filter ──────────────────────────────────────────────────
        // Click a checkbox label → toggle global view for that edge type.
        // When global view is active: show ALL edges of that type across all tokens.
        // Click the same type again → exit global view, return to normal token-click mode.
        document.querySelectorAll('.subtype-cb').forEach(cb => {{
          cb.addEventListener('click', e => {{
            e.preventDefault();  // prevent default checkbox toggle — we manage state manually

            if (globalSubtype === cb.value) {{
              // Exit global view
              globalSubtype = null;
              cb.dataset.global = '';
              document.querySelectorAll('.subtype-cb').forEach(c => {{
                c.checked = true;
                c.style.outline = '';
              }});
              activeSubtypes = new Set(ALL_SUBTYPES);
              // Restore normal view
              arrowSvg.querySelectorAll('path.arrow').forEach(el => el.remove());
              document.querySelectorAll('.tok').forEach(s => {{
                s.style.background = '';
                s.style.boxShadow  = '';
              }});
              document.getElementById('info-panel').innerHTML =
                '<p class="ph">Click a highlighted token in the code to inspect its correlations.</p>';
            }} else {{
              // Enter global view for this subtype
              globalSubtype = cb.value;
              document.querySelectorAll('.subtype-cb').forEach(c => {{
                c.checked = (c.value === cb.value);
                c.style.outline = (c.value === cb.value) ? '2px solid var(--accent)' : '';
              }});
              activeSubtypes = new Set([cb.value]);
              activeIdx = null;  // deselect any token
              drawGlobalEdges();
            }}
          }});
        }});
        </script>
        </body>
        </html>
"""


# ── Public API ────────────────────────────────────────────────────────────────

def visualize_correlations(
        correlations,
        title: str = "Token Correlation Graph",
        code: str = "",
        subwords=None,
        output_path: str | None = None,
        open_browser: bool = True,
) -> str:
    graph_data = _build_graph_data(correlations, subwords)
    code_html = _annotate_code_html(code, graph_data["nodes"], subwords) if code else ""
    html = _render_html(graph_data, title, code_html)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".html", prefix="corr_graph_")
        os.close(fd)

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Graph saved → {output_path}")

    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(output_path)}")

    return output_path
