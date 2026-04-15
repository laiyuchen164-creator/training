from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.commitment_metrics import aggregate_condition_metrics, compute_commitment_metrics
from src.utils import read_json, read_jsonl


def _read_prompt_predictions(path: Path, system_name: str) -> list[dict]:
    rows = []
    for row in read_jsonl(path):
        if row["system"] != system_name:
            continue
        predicted_control = "preserve" if row["final_prediction"] == row["gold_initial_label"] else "replace"
        rows.append(
            {
                "example_id": row["example_id"],
                "condition": row["condition"],
                "predicted_control_decision": predicted_control,
                "predicted_final_answer": row["final_prediction"],
            }
        )
    return rows


def _read_cipc_predictions(path: Path) -> list[dict]:
    return [
        {
            "example_id": row["example_id"],
            "condition": row["condition"],
            "predicted_control_decision": row["predicted_control_decision"],
            "predicted_final_answer": row["predicted_final_answer"],
        }
        for row in read_jsonl(path)
    ]


def _align(examples: list[dict], predictions: list[dict]) -> tuple[list[dict], list[dict]]:
    lookup = {row["example_id"]: row for row in predictions}
    aligned_examples = []
    aligned_predictions = []
    for example in examples:
        if example["example_id"] not in lookup:
            raise KeyError(f"Missing prediction for example_id={example['example_id']}")
        aligned_examples.append(example)
        aligned_predictions.append(lookup[example["example_id"]])
    return aligned_examples, aligned_predictions


def _render_report(
    split: str,
    cipc_summary: dict,
    prompt_summary: dict,
    cipc_condition_rows: list[dict],
    prompt_condition_rows: list[dict],
) -> str:
    def pick(rows: list[dict], condition: str) -> dict:
        for row in rows:
            if row["condition"] == condition:
                return row
        raise KeyError(condition)

    lines = [
        "# CIPC vs Frozen Prompt Baseline",
        "",
        f"Split compared: `{split}`",
        "",
        "## Overall",
        "",
        "| method | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| CIPC | {cipc_summary['control_decision_accuracy']} | "
            f"{cipc_summary['final_answer_accuracy']} | {cipc_summary['joint_accuracy']} | "
            f"{cipc_summary['consistency_rate_gold']} | {cipc_summary['early_commitment_persistence']} | "
            f"{cipc_summary['late_evidence_takeover']} |"
        ),
        (
            f"| frozen_prompt_source_revision | {prompt_summary['control_decision_accuracy']} | "
            f"{prompt_summary['final_answer_accuracy']} | {prompt_summary['joint_accuracy']} | "
            f"{prompt_summary['consistency_rate_gold']} | {prompt_summary['early_commitment_persistence']} | "
            f"{prompt_summary['late_evidence_takeover']} |"
        ),
        "",
        "## By Condition",
        "",
        "| condition | method | control_acc | answer_acc | joint_acc | early_persistence | late_takeover |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for condition in ("full_info", "incremental_no_overturn", "incremental_overturn_reasoning"):
        cipc_row = pick(cipc_condition_rows, condition)
        prompt_row = pick(prompt_condition_rows, condition)
        lines.append(
            f"| {condition} | CIPC | {cipc_row['control_decision_accuracy']} | "
            f"{cipc_row['final_answer_accuracy']} | {cipc_row['joint_accuracy']} | "
            f"{cipc_row['early_commitment_persistence']} | {cipc_row['late_evidence_takeover']} |"
        )
        lines.append(
            f"| {condition} | frozen_prompt_source_revision | {prompt_row['control_decision_accuracy']} | "
            f"{prompt_row['final_answer_accuracy']} | {prompt_row['joint_accuracy']} | "
            f"{prompt_row['early_commitment_persistence']} | {prompt_row['late_evidence_takeover']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The prompt baseline control score is a binary proxy derived from whether",
            "  the final answer preserves the gold early commitment or switches away",
            "  from it.",
            "- This comparison is therefore valid for the current Belief-R binary",
            "  `preserve/replace` setup, but it is not yet the final comparison design",
            "  for a future 3-way `preserve/weaken/replace` setting.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--cipc-run", required=True)
    parser.add_argument("--prompt-run", required=True)
    parser.add_argument("--prompt-system", default="source_revision")
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    examples = read_jsonl(
        PROJECT_ROOT / "data" / "processed" / f"belief_r_commitment_control_{args.split}.jsonl"
    )
    cipc_predictions = _read_cipc_predictions(
        PROJECT_ROOT / "runs" / args.cipc_run / f"{args.split}_predictions.jsonl"
    )
    prompt_predictions = _read_prompt_predictions(
        PROJECT_ROOT / "runs" / args.prompt_run / "predictions.jsonl",
        args.prompt_system,
    )

    cipc_examples, cipc_predictions = _align(examples, cipc_predictions)
    prompt_examples, prompt_predictions = _align(examples, prompt_predictions)
    cipc_summary = compute_commitment_metrics(cipc_examples, cipc_predictions)
    prompt_summary = compute_commitment_metrics(prompt_examples, prompt_predictions)
    cipc_condition_rows = aggregate_condition_metrics(cipc_examples, cipc_predictions)
    prompt_condition_rows = aggregate_condition_metrics(prompt_examples, prompt_predictions)
    for row in cipc_condition_rows:
        row["method"] = "CIPC"
    for row in prompt_condition_rows:
        row["method"] = "frozen_prompt_source_revision"

    report_text = _render_report(
        args.split,
        cipc_summary,
        prompt_summary,
        cipc_condition_rows,
        prompt_condition_rows,
    )
    output_report = PROJECT_ROOT / args.output_report
    output_report.write_text(report_text, encoding="utf-8")
    output_csv = PROJECT_ROOT / args.output_csv
    _write_csv(output_csv, cipc_condition_rows + prompt_condition_rows)

    summary_payload = {
        "split": args.split,
        "cipc": cipc_summary,
        "frozen_prompt_source_revision": prompt_summary,
        "prompt_run_manifest": read_json(PROJECT_ROOT / "runs" / args.prompt_run / "run_manifest.json"),
    }
    (output_report.with_suffix(".json")).write_text(
        __import__("json").dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    print(report_text)


if __name__ == "__main__":
    main()
