from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "runs" / "belief_r_api_pilot_openai_medium_ablation" / "predictions.jsonl"
OUT = ROOT / "analysis" / "belief_r_full_info_confusion_report.md"


def load_records() -> list[dict]:
    return [
        json.loads(line)
        for line in PREDICTIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    records = [record for record in load_records() if record["condition"] == "full_info"]
    counts: dict[tuple[str, str], int] = defaultdict(int)
    examples: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for record in records:
        key = (record["gold_final_label"], record["final_prediction"])
        counts[key] += 1
        if record["gold_final_label"] != record["final_prediction"] and len(examples[key]) < 3:
            examples[key].append(record)

    lines = [
        "# Belief-R Full-Info Confusion Report\n\n",
        f"- Source run: `{PREDICTIONS.parent.name}`\n",
        f"- Total full-info predictions: {len(records)}\n\n",
        "## Confusion Counts\n\n",
        "| gold | predicted | count |\n",
        "|---|---|---:|\n",
    ]

    for (gold, predicted), count in sorted(counts.items()):
        lines.append(f"| {gold} | {predicted} | {count} |\n")

    lines.append("\n## Sample Mistakes\n\n")
    for key, mistake_examples in sorted(examples.items()):
        gold, predicted = key
        lines.append(f"### gold `{gold}` -> predicted `{predicted}`\n\n")
        for record in mistake_examples:
            trace = record["trace"]["turns"][0]
            lines.append(f"- Example: `{record['example_id']}`\n")
            lines.append(f"  Raw output: `{trace.get('raw_output', '')[:220]}`\n")
        lines.append("\n")

    OUT.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
