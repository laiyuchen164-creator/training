from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUTS = [
    (
        "belief_r_api_openai_medium_ablation",
        ROOT / "runs" / "belief_r_api_pilot_openai_medium_ablation" / "summary.csv",
    ),
    (
        "atomic_explicit_openai_small",
        ROOT / "runs" / "atomic_explicit_openai_small" / "summary.csv",
    ),
    (
        "reviseqa_openai_small_v2_exploratory",
        ROOT / "runs" / "reviseqa_openai_small_v2" / "summary.csv",
    ),
]

OUT_CSV = ROOT / "paper_assets" / "consolidated_results.csv"
OUT_MD = ROOT / "paper_assets" / "consolidated_results.md"


def load_rows() -> list[dict]:
    rows = []
    for dataset_name, path in INPUTS:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["dataset_name"] = dataset_name
                rows.append(row)
    return rows


def write_csv(rows: list[dict]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset_name"] + [key for key in rows[0].keys() if key != "dataset_name"]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict]) -> None:
    header = (
        "| dataset | system | condition | n | accuracy | "
        "assistant_assumption_survival | correction_uptake |\n"
    )
    separator = "|---|---|---|---:|---:|---:|---:|\n"
    lines = [header, separator]
    for row in rows:
        lines.append(
            f"| {row['dataset_name']} | {row['system']} | {row['condition']} | "
            f"{row['n']} | {row['accuracy']} | "
            f"{row['assistant_assumption_survival']} | {row['correction_uptake']} |\n"
        )
    OUT_MD.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    rows = load_rows()
    write_csv(rows)
    write_markdown(rows)


if __name__ == "__main__":
    main()
