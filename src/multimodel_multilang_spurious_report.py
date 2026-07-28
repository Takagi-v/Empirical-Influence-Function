from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CATEGORIES = (
    "training_set_cooccurrence",
    "local_proximity",
    "comment_string_leakage",
    "lexical_similarity",
    "scope_confusion",
    "prompt_format_shortcut",
)

CATEGORY_LABELS = {
    "lexical_similarity": "Lexical similarity",
    "local_proximity": "Local proximity",
    "training_set_cooccurrence": "Training-set co-occurrence",
    "prompt_format_shortcut": "Prompt-format shortcut",
    "scope_confusion": "Scope confusion",
    "comment_string_leakage": "Comment/string leakage",
}

CATEGORY_COLORS = {
    "lexical_similarity": "#4C78A8",
    "local_proximity": "#F58518",
    "training_set_cooccurrence": "#54A24B",
    "prompt_format_shortcut": "#B279A2",
    "scope_confusion": "#E45756",
    "comment_string_leakage": "#72B7B2",
}

MODEL_LABELS = {
    "qwen2p5-coder-0p5b-instruct": "Qwen2.5-Coder-0.5B",
    "qwen2p5-coder-1p5b-instruct": "Qwen2.5-Coder-1.5B",
    "qwen2p5-coder-7b-instruct": "Qwen2.5-Coder-7B",
    "qwen2p5-coder-14b-instruct": "Qwen2.5-Coder-14B",
    "deepseek-coder-6p7b-base": "DeepSeek-Coder-6.7B",
}

MODEL_SHORT_LABELS = {
    "qwen2p5-coder-0p5b-instruct": "0.5B",
    "qwen2p5-coder-1p5b-instruct": "1.5B",
    "qwen2p5-coder-7b-instruct": "7B",
    "qwen2p5-coder-14b-instruct": "Q14",
    "deepseek-coder-6p7b-base": "DS6.7",
}

LANGUAGE_LABELS = {
    "go": "Go",
    "java": "Java",
    "javascript": "JavaScript",
    "python": "Python",
}


def _read_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_csv_arg(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def collect_rows(
    input_dir: Path,
    languages: list[str],
    model_tags: list[str],
    rate_key: str,
    category_key: str,
) -> list[dict]:
    rows = []
    for language in languages:
        for model_tag in model_tags:
            summary_path = input_dir / model_tag / language / "summary.json"
            summary = _read_summary(summary_path)
            metrics = summary.get("metrics", {})
            rate = float(metrics[rate_key])
            proportions = metrics.get(category_key, {})
            row = {
                "language": language,
                "language_label": LANGUAGE_LABELS.get(language, language),
                "model_tag": model_tag,
                "model_label": MODEL_LABELS.get(model_tag, model_tag),
                "model_short_label": MODEL_SHORT_LABELS.get(model_tag, model_tag),
                "ok_cases": int(summary.get("ok_count", 0)),
                "input_count": int(summary.get("input_count", 0)),
                "spurious_rate": rate,
                "non_spurious_rate": max(0.0, 1.0 - rate),
            }
            for category in CATEGORIES:
                prop = float(proportions.get(category, 0.0) or 0.0)
                row[f"{category}_within_spurious"] = prop
                row[f"{category}_overall"] = rate * prop
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "language",
        "model_tag",
        "input_count",
        "ok_cases",
        "spurious_rate",
        "non_spurious_rate",
    ]
    for category in CATEGORIES:
        fieldnames.append(f"{category}_within_spurious")
    for category in CATEGORIES:
        fieldnames.append(f"{category}_overall")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_latex_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{table*}[t]\n")
        handle.write("\\centering\n")
        handle.write("\\scriptsize\n")
        handle.write("\\caption{Spurious-correlation prevalence and category breakdown across languages and Qwen2.5-Coder model sizes. Category values are percentages within spurious correlations.}\n")
        handle.write("\\label{tab:multimodel-multilang-spurious-breakdown}\n")
        handle.write("\\begin{tabular}{llccccccccc}\n")
        handle.write("\\toprule\n")
        handle.write("Language & Model & Cases & Spur. & Co-occur. & Local & Comment/string & Lex. & Scope & Format \\\\\n")
        handle.write("\\midrule\n")
        last_language = None
        for row in rows:
            if last_language is not None and row["language"] != last_language:
                handle.write("\\addlinespace\n")
            last_language = row["language"]
            handle.write(
                f"{row['language_label']} & {row['model_label']} & {row['ok_cases']} & "
                f"{100 * row['spurious_rate']:.1f} & "
                f"{100 * row['training_set_cooccurrence_within_spurious']:.1f} & "
                f"{100 * row['local_proximity_within_spurious']:.1f} & "
                f"{100 * row['comment_string_leakage_within_spurious']:.1f} & "
                f"{100 * row['lexical_similarity_within_spurious']:.1f} & "
                f"{100 * row['scope_confusion_within_spurious']:.1f} & "
                f"{100 * row['prompt_format_shortcut_within_spurious']:.1f} \\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\end{table*}\n")


def write_figure(path: Path, rows: list[dict], languages: list[str], model_tags: list[str]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 5.8,
            "axes.labelsize": 5.8,
            "axes.titlesize": 6.4,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "legend.fontsize": 5.3,
        }
    )

    rows_by_key = {(row["language"], row["model_tag"]): row for row in rows}
    n_cols = len(languages)
    fig, axes = plt.subplots(1, n_cols, figsize=(6.85, 1.55), sharex=True, sharey=True)
    if n_cols == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    legend_handles = [
        Patch(facecolor=CATEGORY_COLORS[category], edgecolor="none", label=CATEGORY_LABELS[category])
        for category in CATEGORIES
    ]
    legend_handles.append(Patch(facecolor="#D9D9D9", edgecolor="none", label="Structurally supported"))

    y_positions = list(range(len(model_tags) - 1, -1, -1))
    y_labels = [MODEL_LABELS.get(model_tag, model_tag) for model_tag in model_tags]
    for ax, language in zip(axes, languages):
        ax.set_title(LANGUAGE_LABELS.get(language, language), fontweight="bold", pad=2.5)
        ax.set_xlim(0, 112)
        ax.set_ylim(-0.55, len(model_tags) - 0.45)
        ax.set_xticks([0, 50, 100])
        ax.grid(axis="x", color="#E8E8E8", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0, pad=1.2)
        ax.tick_params(axis="y", length=0, pad=2.0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for ypos, model_tag in zip(y_positions, model_tags):
            row = rows_by_key[(language, model_tag)]
            left = 0.0
            for category in CATEGORIES:
                value = 100.0 * row[f"{category}_overall"]
                ax.barh(
                    ypos,
                    value,
                    left=left,
                    height=0.48,
                    color=CATEGORY_COLORS[category],
                    edgecolor="white",
                    linewidth=0.35,
                )
                if value >= 6.5:
                    ax.text(
                        left + value / 2.0,
                        ypos,
                        f"{value:.0f}",
                        ha="center",
                        va="center",
                        fontsize=4.9,
                        color="white",
                        fontweight="bold",
                        clip_on=True,
                    )
                left += value

            supported = 100.0 * row["non_spurious_rate"]
            ax.barh(
                ypos,
                supported,
                left=left,
                height=0.48,
                color="#D9D9D9",
                edgecolor="white",
                linewidth=0.35,
            )
            if supported >= 9.0:
                ax.text(
                    left + supported / 2.0,
                    ypos,
                    f"{supported:.0f}",
                    ha="center",
                    va="center",
                    fontsize=4.9,
                    color="#333333",
                    fontweight="bold",
                    clip_on=True,
                )
            ax.text(
                103.0,
                ypos,
                f"{100.0 * row['spurious_rate']:.1f}%",
                ha="left",
                va="center",
                fontsize=5.8,
                color="#111111",
                fontweight="bold",
            )

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    legend_order = [0, 4, 1, 5, 2, 6, 3]
    fig.legend(
        [legend_handles[idx] for idx in legend_order],
        [legend_handles[idx].get_label() for idx in legend_order],
        loc="lower center",
        bbox_to_anchor=(0.53, 0.010),
        ncol=4,
        frameon=False,
        columnspacing=0.72,
        handlelength=0.75,
        handletextpad=0.20,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.145, right=0.985, top=0.82, bottom=0.255, wspace=0.28)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.010)
    preview_path = path.with_name(path.stem + "_preview.png")
    fig.savefig(preview_path, dpi=260, bbox_inches="tight", pad_inches=0.010)
    plt.close(fig)


def write_include_tex(path: Path, figure_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{figure*}[t]\n")
        handle.write("\\centering\n")
        handle.write(f"\\includegraphics[width=\\textwidth]{{figures/{figure_name}}}\n")
        handle.write("\\caption{Failure pattern breakdown for high-ALTI correlations at the first erroneous token across languages and Qwen2.5-Coder model sizes. Rows correspond to model sizes and columns correspond to programming languages. Colored segments show the percentage of all analyzed high-ALTI correlations assigned to each spurious failure pattern, the gray segment shows structurally supported correlations, in-segment labels report segment percentages, and the bold value reports the total spurious-correlation rate.}\n")
        handle.write("\\label{fig:multimodel-multilang-spurious-breakdown}\n")
        handle.write("\\end{figure*}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CSV, LaTeX, and figure outputs for multilingual/multimodel spurious results."
    )
    parser.add_argument("--input-dir", default="spurious_correlation_results/multilang_multimodel_1epoch")
    parser.add_argument("--output-dir", default="spurious_correlation_results/paper_draft/figures")
    parser.add_argument("--languages", default="go,java,javascript,python")
    parser.add_argument(
        "--models",
        default="qwen2p5-coder-0p5b-instruct,qwen2p5-coder-1p5b-instruct,qwen2p5-coder-7b-instruct",
    )
    parser.add_argument("--rate-key", default="loose_mapped_spurious@50")
    parser.add_argument("--category-key", default="loose_mapped_category_proportions@50")
    parser.add_argument("--figure-name", default="figure_multimodel_multilang_spurious_breakdown.pdf")
    args = parser.parse_args()

    languages = _parse_csv_arg(args.languages)
    model_tags = _parse_csv_arg(args.models)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    rows = collect_rows(input_dir, languages, model_tags, args.rate_key, args.category_key)
    write_csv(output_dir / "multimodel_multilang_spurious_breakdown.csv", rows)
    write_latex_table(output_dir / "table_multimodel_multilang_spurious_breakdown.tex", rows)
    write_figure(output_dir / args.figure_name, rows, languages, model_tags)
    write_include_tex(
        output_dir / "figure_multimodel_multilang_spurious_breakdown.tex",
        Path(args.figure_name).name,
    )
    print(f"Wrote {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
