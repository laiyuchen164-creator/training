from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "runs" / "belief_r_api_pilot_openai_medium_ablation" / "summary.csv"
OUTPUT_DIR = ROOT / "paper_assets"


def load_rows() -> list[dict]:
    with SUMMARY_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def draw_grouped_bar_chart(
    rows: list[dict],
    *,
    filename: str,
    title: str,
    metric: str,
    condition: str,
    y_max: float = 1.0,
) -> None:
    rows = [row for row in rows if row["condition"] == condition]
    systems = [row["system"] for row in rows]
    values = [float(row[metric]) for row in rows]
    colors = {
        "raw_history": "#334155",
        "running_summary": "#0f766e",
        "structured_no_source": "#2563eb",
        "source_no_revision": "#7c3aed",
        "source_revision": "#dc2626",
    }

    width, height = 960, 540
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 70, 120
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    bar_gap = 24
    bar_width = (plot_width - bar_gap * (len(rows) - 1)) / len(rows)

    def y_pos(value: float) -> float:
        return margin_top + plot_height * (1 - (value / y_max))

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    svg.append('<rect width="100%" height="100%" fill="#f8fafc"/>')
    svg.append(
        f'<text x="{width/2}" y="36" text-anchor="middle" '
        f'font-family="Arial" font-size="24" fill="#0f172a">{title}</text>'
    )

    for tick in range(6):
        value = y_max * tick / 5
        y = y_pos(value)
        svg.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{width - margin_right}" y2="{y}" '
            'stroke="#cbd5e1" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{margin_left - 12}" y="{y + 5}" text-anchor="end" '
            f'font-family="Arial" font-size="14" fill="#475569">{value:.1f}</text>'
        )

    svg.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" '
        'stroke="#0f172a" stroke-width="2"/>'
    )
    svg.append(
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" '
        'stroke="#0f172a" stroke-width="2"/>'
    )

    for index, (system, value) in enumerate(zip(systems, values)):
        x = margin_left + index * (bar_width + bar_gap)
        y = y_pos(value)
        bar_height = margin_top + plot_height - y
        color = colors.get(system, "#64748b")
        svg.append(
            f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" '
            f'fill="{color}" rx="4"/>'
        )
        svg.append(
            f'<text x="{x + bar_width/2}" y="{y - 10}" text-anchor="middle" '
            f'font-family="Arial" font-size="14" fill="#0f172a">{value:.2f}</text>'
        )
        svg.append(
            f'<text x="{x + bar_width/2}" y="{height - margin_bottom + 24}" text-anchor="end" '
            f'transform="rotate(-30 {x + bar_width/2} {height - margin_bottom + 24})" '
            f'font-family="Arial" font-size="14" fill="#334155">{system}</text>'
        )

    svg.append(
        f'<text x="{margin_left - 55}" y="{margin_top + plot_height/2}" text-anchor="middle" '
        f'transform="rotate(-90 {margin_left - 55} {margin_top + plot_height/2})" '
        f'font-family="Arial" font-size="16" fill="#0f172a">{metric}</text>'
    )
    svg.append("</svg>")
    (OUTPUT_DIR / filename).write_text("\n".join(svg), encoding="utf-8")


def draw_tradeoff_chart(rows: list[dict], *, filename: str) -> None:
    rows = [row for row in rows if row["condition"] == "incremental_overturn_reasoning"]
    lookup = {row["system"]: row for row in rows}
    systems = list(lookup)
    width, height = 960, 540
    margin_left, margin_right, margin_top, margin_bottom = 90, 50, 70, 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    colors = {
        "raw_history": "#334155",
        "running_summary": "#0f766e",
        "structured_no_source": "#2563eb",
        "source_no_revision": "#7c3aed",
        "source_revision": "#dc2626",
    }

    def x_pos(value: float) -> float:
        return margin_left + plot_width * value

    def y_pos(value: float) -> float:
        return margin_top + plot_height * (1 - value)

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    svg.append('<rect width="100%" height="100%" fill="#f8fafc"/>')
    svg.append(
        f'<text x="{width/2}" y="36" text-anchor="middle" '
        f'font-family="Arial" font-size="24" fill="#0f172a">OpenAI Belief-Update Trade-off</text>'
    )

    for tick in range(6):
        value = tick / 5
        x = x_pos(value)
        y = y_pos(value)
        svg.append(
            f'<line x1="{x}" y1="{margin_top}" x2="{x}" y2="{height - margin_bottom}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        svg.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{width - margin_right}" y2="{y}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x}" y="{height - margin_bottom + 22}" text-anchor="middle" '
            f'font-family="Arial" font-size="13" fill="#475569">{value:.1f}</text>'
        )
        svg.append(
            f'<text x="{margin_left - 12}" y="{y + 4}" text-anchor="end" '
            f'font-family="Arial" font-size="13" fill="#475569">{value:.1f}</text>'
        )

    svg.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" '
        'stroke="#0f172a" stroke-width="2"/>'
    )
    svg.append(
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" '
        'stroke="#0f172a" stroke-width="2"/>'
    )

    for system in systems:
        row = lookup[system]
        x = x_pos(float(row["assistant_assumption_survival"]))
        y = y_pos(float(row["accuracy"]))
        color = colors.get(system, "#64748b")
        svg.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{color}"/>')
        svg.append(
            f'<text x="{x + 12}" y="{y - 10}" font-family="Arial" font-size="14" fill="#0f172a">{system}</text>'
        )

    svg.append(
        f'<text x="{width/2}" y="{height - 18}" text-anchor="middle" '
        f'font-family="Arial" font-size="16" fill="#0f172a">assistant_assumption_survival</text>'
    )
    svg.append(
        f'<text x="{28}" y="{margin_top + plot_height/2}" text-anchor="middle" '
        f'transform="rotate(-90 28 {margin_top + plot_height/2})" '
        f'font-family="Arial" font-size="16" fill="#0f172a">accuracy on overturn subset</text>'
    )
    svg.append("</svg>")
    (OUTPUT_DIR / filename).write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    draw_grouped_bar_chart(
        rows,
        filename="belief_r_api_openai_ablation_overturn_accuracy.svg",
        title="OpenAI Medium Ablation: Overturn Accuracy",
        metric="accuracy",
        condition="incremental_overturn_reasoning",
    )
    draw_grouped_bar_chart(
        rows,
        filename="belief_r_api_openai_ablation_no_overturn_accuracy.svg",
        title="OpenAI Medium Ablation: No-Overturn Accuracy",
        metric="accuracy",
        condition="incremental_no_overturn",
    )
    draw_tradeoff_chart(rows, filename="belief_r_api_openai_ablation_tradeoff.svg")


if __name__ == "__main__":
    main()
