from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.commitment_metrics import (
    aggregate_condition_metrics,
    compute_commitment_metrics,
    render_metrics_markdown,
)
from src.utils import read_jsonl, write_json


def write_condition_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_prompt_predictions(path: Path, system_name: str) -> list[dict]:
    predictions = []
    for row in read_jsonl(path):
        if row["system"] != system_name:
            continue
        predicted_control = (
            "preserve" if row["final_prediction"] == row["gold_initial_label"] else "replace"
        )
        predictions.append(
            {
                "example_id": row["example_id"],
                "predicted_control_decision": predicted_control,
                "predicted_final_answer": row["final_prediction"],
            }
        )
    return predictions


def align(examples: list[dict], predictions: list[dict]) -> list[dict]:
    by_id = {row["example_id"]: row for row in predictions}
    return [by_id[example["example_id"]] for example in examples]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-run", required=True)
    parser.add_argument("--prompt-system", default="source_revision")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    examples = read_jsonl(
        PROJECT_ROOT / "data" / "processed" / f"belief_r_commitment_control_{args.split}.jsonl"
    )
    predictions = read_prompt_predictions(
        PROJECT_ROOT / "runs" / args.prompt_run / "predictions.jsonl",
        args.prompt_system,
    )
    aligned_predictions = align(examples, predictions)
    summary = {args.split: compute_commitment_metrics(examples, aligned_predictions)}
    condition_rows = aggregate_condition_metrics(examples, aligned_predictions)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", summary)
    write_condition_csv(output_dir / "metrics_by_condition.csv", condition_rows)
    (output_dir / "summary.md").write_text(
        render_metrics_markdown(summary, condition_rows),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "prompt_run": args.prompt_run,
                "prompt_system": args.prompt_system,
                "split": args.split,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
