from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e4 or av < 1e-3:
        return f"{v:.{digits}e}"
    return f"{v:.{digits}f}".rstrip("0").rstrip(".")


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_records(records: list[dict[str, Any]], *, steps: str, last: int | None) -> list[dict[str, Any]]:
    if steps and steps != "all":
        wanted = {int(x.strip()) for x in steps.split(",") if x.strip()}
        records = [r for r in records if int(r["step"]) in wanted]
    if last is not None and last > 0:
        records = records[-last:]
    return records


def token_label(row: dict[str, Any]) -> str:
    display = str(row.get("display") or row.get("token_text") or "")
    return display.replace("|", "\\|")


def render_step_markdown(record: dict[str, Any], *, max_tokens: int) -> str:
    sm = record["sample_metrics"]
    tm = record["token_metrics"]
    lines: list[str] = []
    lines.append(f"## Step {record['step']} ({record['stage']})")
    lines.append("")
    lines.append(
        f"- q: `#{tm['query']}` `{tm['query_token_text']}`; "
        f"|Pq|={tm['num_P']}; |Nq|={tm['num_N']}; top_k={tm['top_k']}"
    )
    lines.append(
        "- sample: "
        f"Lce={fmt(sm['Lce'])}, Lsal={fmt(sm['Lsal'])}, "
        f"gamma*Lsal/Lce={fmt(sm['gamma_Lsal_over_Lce'])}, "
        f"CE_grad={fmt(sm['ce_grad_norm'])}, SAL_grad={fmt(sm['sal_grad_norm'])}, "
        f"cos={fmt(sm['ce_sal_grad_cosine'])}, total_grad={fmt(sm['sample_total_grad_norm'])}"
    )
    lines.append(
        "- token: "
        f"Pbar={fmt(tm['Pbar'])}, Nbar={fmt(tm['Nbar'])}, ratio={fmt(tm['ratio'])}, "
        f"Lq={fmt(tm['Lq'])}, Lq_norm={fmt(tm['Lq_norm'])}, "
        f"neg_den={fmt(tm.get('negative_denominator_value'))}"
    )
    lines.append("")
    lines.append("### Nq cap Tq20")
    lines.append("")
    lines.append("| rank | source | token | streak | Cqs | lqs | exp(max(lqs,eps)) | pqs_mean |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|")
    for row in tm.get("Nq_top20", [])[:max_tokens]:
        lines.append(
            f"| {row.get('rank', '')} | {row.get('source', '')} | `{token_label(row)}` | "
            f"{row.get('top20_streak', '')} | {fmt(row.get('Cqs'))} | {fmt(row.get('lqs'))} | "
            f"{fmt(row.get('exp_max_lqs_floor'))} | {fmt(row.get('pqs_mean_over_P'))} |"
        )
    lines.append("")
    lines.append("### Pq cap Tq20")
    lines.append("")
    lines.append("| rank | source | token | Cqs | lqs | exp(lqs) | pqs | loss_qs |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|")
    for row in tm.get("Pq_top20", [])[:max_tokens]:
        lines.append(
            f"| {row.get('rank', '')} | {row.get('source', '')} | `{token_label(row)}` | "
            f"{fmt(row.get('Cqs'))} | {fmt(row.get('lqs'))} | {fmt(row.get('exp_lqs'))} | "
            f"{fmt(row.get('pqs'))} | {fmt(row.get('loss_qs'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_markdown(records: list[dict[str, Any]], *, max_tokens: int) -> str:
    if not records:
        return "# Row/Token Trace Report\n\nNo records selected.\n"
    first = records[0]
    last = records[-1]
    lines: list[str] = []
    lines.append("# Row/Token Trace Report")
    lines.append("")
    lines.append(
        f"- sample row: `{first['sample_index']}`\n"
        f"- query: `#{first['query_token_index']}` `{first['query_token_text']}`\n"
        f"- loss: `{first['saliency_loss_name']}` (`{first['saliency_loss_type']}`)\n"
        f"- lambda: `{first['saliency_lambda']}`\n"
        f"- tau: `{first['saliency_temperature_tau']}`\n"
        f"- floor_logit_eps: `{first['saliency_floor_logit_eps']}`\n"
        f"- records: `{len(records)}`; step range: `{first['step']}` to `{last['step']}`"
    )
    lines.append("")
    lines.append("## Step Summary")
    lines.append("")
    lines.append("| step | Lce | Lsal | gamma*Lsal/Lce | CE grad | SAL grad | cos | total grad | Lq | Lq_norm | Pbar | Nbar | ratio |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rec in records:
        sm = rec["sample_metrics"]
        tm = rec["token_metrics"]
        lines.append(
            f"| {rec['step']} | {fmt(sm['Lce'])} | {fmt(sm['Lsal'])} | {fmt(sm['gamma_Lsal_over_Lce'])} | "
            f"{fmt(sm['ce_grad_norm'])} | {fmt(sm['sal_grad_norm'])} | {fmt(sm['ce_sal_grad_cosine'])} | "
            f"{fmt(sm['sample_total_grad_norm'])} | {fmt(tm['Lq'])} | {fmt(tm['Lq_norm'])} | "
            f"{fmt(tm['Pbar'])} | {fmt(tm['Nbar'])} | {fmt(tm['ratio'])} |"
        )
    lines.append("")
    for rec in records:
        lines.append("<details open>")
        lines.append(f"<summary>Step {rec['step']}</summary>")
        lines.append("")
        lines.append(render_step_markdown(rec, max_tokens=max_tokens))
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def render_text(records: list[dict[str, Any]], *, max_tokens: int) -> str:
    chunks: list[str] = []
    for rec in records:
        sm = rec["sample_metrics"]
        tm = rec["token_metrics"]
        chunks.append(
            f"STEP {rec['step']} [{rec['stage']}] q=#{tm['query']} {tm['query_token_text']} "
            f"|P|={tm['num_P']} |N|={tm['num_N']}\n"
            f"  sample: Lce={fmt(sm['Lce'])} Lsal={fmt(sm['Lsal'])} "
            f"gammaLsal/Lce={fmt(sm['gamma_Lsal_over_Lce'])} "
            f"CEg={fmt(sm['ce_grad_norm'])} SALg={fmt(sm['sal_grad_norm'])} "
            f"cos={fmt(sm['ce_sal_grad_cosine'])} total_g={fmt(sm['sample_total_grad_norm'])}\n"
            f"  token : Pbar={fmt(tm['Pbar'])} Nbar={fmt(tm['Nbar'])} ratio={fmt(tm['ratio'])} "
            f"Lq={fmt(tm['Lq'])} Lq_norm={fmt(tm['Lq_norm'])}\n"
        )
        chunks.append("  Nq cap Tq20:\n")
        chunks.append("    rank src  token                 streak Cqs      lqs      exp(max)   pqs_mean\n")
        for row in tm.get("Nq_top20", [])[:max_tokens]:
            chunks.append(
                f"    {int(row.get('rank', 0)):>4} {int(row.get('source', 0)):>3}  "
                f"{token_label(row)[:20]:<20} {int(row.get('top20_streak', 0)):>6} "
                f"{fmt(row.get('Cqs')):>8} {fmt(row.get('lqs')):>8} "
                f"{fmt(row.get('exp_max_lqs_floor')):>10} {fmt(row.get('pqs_mean_over_P')):>10}\n"
            )
        chunks.append("  Pq cap Tq20:\n")
        chunks.append("    rank src  token                 Cqs      lqs      exp(lqs)   pqs       loss_qs\n")
        for row in tm.get("Pq_top20", [])[:max_tokens]:
            chunks.append(
                f"    {int(row.get('rank', 0)):>4} {int(row.get('source', 0)):>3}  "
                f"{token_label(row)[:20]:<20} {fmt(row.get('Cqs')):>8} "
                f"{fmt(row.get('lqs')):>8} {fmt(row.get('exp_lqs')):>10} "
                f"{fmt(row.get('pqs')):>9} {fmt(row.get('loss_qs')):>9}\n"
            )
        chunks.append("\n")
    return "".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a readable report from row/token trace.jsonl.")
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-txt", type=Path, default=None)
    parser.add_argument("--steps", default="all", help="Comma-separated steps, or all.")
    parser.add_argument("--last", type=int, default=None, help="Only render the last N selected records.")
    parser.add_argument("--max-tokens", type=int, default=20)
    args = parser.parse_args()

    records = select_records(load_records(args.trace_jsonl), steps=args.steps, last=args.last)
    out_md = args.out_md or args.trace_jsonl.with_name("trace_readable.md")
    out_txt = args.out_txt or args.trace_jsonl.with_name("trace_readable.txt")
    out_md.write_text(render_markdown(records, max_tokens=args.max_tokens), encoding="utf-8")
    out_txt.write_text(render_text(records, max_tokens=args.max_tokens), encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_txt}")


if __name__ == "__main__":
    main()
