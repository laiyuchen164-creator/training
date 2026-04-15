from __future__ import annotations

from collections import defaultdict

from src.utils import format_ratio


def _is_consistent(control_decision: str, early_label: str, answer_label: str) -> bool:
    if control_decision == "preserve":
        return answer_label == early_label
    if control_decision == "replace":
        return answer_label != early_label
    if control_decision == "weaken":
        return answer_label not in {early_label} and answer_label == "c"
    return False


def compute_commitment_metrics(examples: list[dict], predictions: list[dict]) -> dict:
    if len(examples) != len(predictions):
        raise ValueError("Examples and predictions must be aligned.")

    total = len(examples)
    gold_replace_total = 0
    control_correct = 0
    answer_correct = 0
    joint_correct = 0
    consistency_predicted = 0
    consistency_gold = 0
    early_commitment_persistence = 0
    late_evidence_takeover = 0
    assistant_assumption_survival = 0
    correction_uptake = 0

    for example, prediction in zip(examples, predictions):
        gold_control = example["control_label"]
        gold_answer = example["final_answer_label"]
        early_label = example["early_commitment_label"]
        pred_control = prediction["predicted_control_decision"]
        pred_answer = prediction["predicted_final_answer"]
        gold_requires_change = gold_control in {"replace", "weaken"}

        control_correct += int(pred_control == gold_control)
        answer_correct += int(pred_answer == gold_answer)
        joint_correct += int(pred_control == gold_control and pred_answer == gold_answer)
        consistency_predicted += int(_is_consistent(pred_control, early_label, pred_answer))
        consistency_gold += int(_is_consistent(gold_control, early_label, pred_answer))

        if gold_requires_change:
            gold_replace_total += 1
            early_commitment_persistence += int(pred_answer == early_label)
            assistant_assumption_survival += int(
                example["source_type"] == "assistant_inferred" and pred_answer == early_label
            )
            late_evidence_takeover += int(pred_answer == gold_answer and pred_answer != early_label)
            correction_uptake += int(pred_answer == gold_answer)

    return {
        "n": total,
        "control_decision_accuracy": format_ratio(control_correct, total),
        "final_answer_accuracy": format_ratio(answer_correct, total),
        "joint_accuracy": format_ratio(joint_correct, total),
        "consistency_rate_predicted": format_ratio(consistency_predicted, total),
        "consistency_rate_gold": format_ratio(consistency_gold, total),
        "early_commitment_persistence": format_ratio(
            early_commitment_persistence, gold_replace_total
        ),
        "late_evidence_takeover": format_ratio(late_evidence_takeover, gold_replace_total),
        "assistant_assumption_survival": format_ratio(
            assistant_assumption_survival, gold_replace_total
        ),
        "correction_uptake": format_ratio(correction_uptake, gold_replace_total),
        "replace_or_weaken_count": gold_replace_total,
    }


def aggregate_condition_metrics(examples: list[dict], predictions: list[dict]) -> list[dict]:
    grouped_examples: dict[tuple[str, str], list[dict]] = defaultdict(list)
    grouped_predictions: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for example, prediction in zip(examples, predictions):
        key = (example["metadata"]["split"], example["condition"])
        grouped_examples[key].append(example)
        grouped_predictions[key].append(prediction)

    rows = []
    for key in sorted(grouped_examples):
        split, condition = key
        metrics = compute_commitment_metrics(grouped_examples[key], grouped_predictions[key])
        rows.append({"split": split, "condition": condition, **metrics})
    return rows


def render_metrics_markdown(summary_by_split: dict[str, dict], condition_rows: list[dict]) -> str:
    lines = [
        "# Commitment-Control Evaluation",
        "",
        "## Split Summary",
        "",
    ]
    for split in ("train", "dev", "test"):
        if split not in summary_by_split:
            continue
        payload = summary_by_split[split]
        lines.extend(
            [
                f"### {split}",
                "",
                f"- n: `{payload['n']}`",
                f"- control_decision_accuracy: `{payload['control_decision_accuracy']}`",
                f"- final_answer_accuracy: `{payload['final_answer_accuracy']}`",
                f"- joint_accuracy: `{payload['joint_accuracy']}`",
                f"- consistency_rate_gold: `{payload['consistency_rate_gold']}`",
                f"- early_commitment_persistence: `{payload['early_commitment_persistence']}`",
                f"- late_evidence_takeover: `{payload['late_evidence_takeover']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## By Condition",
            "",
            "| split | condition | n | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in condition_rows:
        lines.append(
            f"| {row['split']} | {row['condition']} | {row['n']} | "
            f"{row['control_decision_accuracy']} | {row['final_answer_accuracy']} | "
            f"{row['joint_accuracy']} | {row['consistency_rate_gold']} | "
            f"{row['early_commitment_persistence']} | {row['late_evidence_takeover']} |"
        )
    lines.append("")
    return "\n".join(lines)
