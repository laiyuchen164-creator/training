from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from src.utils import format_ratio


def enrich_record(record: dict) -> dict:
    is_incremental = record["condition"].startswith("incremental_")
    gold_change = (
        is_incremental and record["gold_initial_label"] != record["gold_final_label"]
    )
    initial_correct = (
        record["initial_prediction"] == record["gold_initial_label"]
        if record["initial_prediction"] is not None
        else None
    )
    final_correct = record["final_prediction"] == record["gold_final_label"]
    stale_persistence = gold_change and record["final_prediction"] == record["gold_initial_label"]
    correction_uptake = gold_change and final_correct

    belief_states = [
        belief
        for belief in record["trace"]["final_snapshot"]["beliefs"]
        if belief["belief_type"] == "intermediate_conclusion"
    ]
    assistant_survival = any(
        belief["source"] == "assistant_inferred"
        and belief["metadata"].get("label") == record["gold_initial_label"]
        and belief["status"] in {"tentative", "confirmed"}
        for belief in belief_states
    )
    deprecated_leakage = any(
        belief["status"] == "deprecated"
        and belief["metadata"].get("label") == record["final_prediction"]
        for belief in belief_states
    )
    wrong_turn_recovery = bool(gold_change and initial_correct is True and final_correct)

    record.update(
        {
            "gold_change": gold_change,
            "initial_correct": initial_correct,
            "final_correct": final_correct,
            "stale_belief_persistence": stale_persistence,
            "assistant_assumption_survival": assistant_survival if gold_change else False,
            "correction_uptake": correction_uptake,
            "deprecated_belief_leakage": deprecated_leakage,
            "wrong_turn_recovery": wrong_turn_recovery,
            "cost_normalized_task_success": round(
                (1.0 if final_correct else 0.0) * 1000 / max(1, record["prompt_tokens_approx"]),
                4,
            ),
        }
    )
    return record


def write_summary(records: list[dict], destination: Path) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["system"], record["condition"])].append(record)

    rows = []
    for (system, condition), group in sorted(grouped.items()):
        overturn_group = [record for record in group if record["gold_change"]]
        row = {
            "system": system,
            "condition": condition,
            "n": len(group),
            "accuracy": format_ratio(sum(record["final_correct"] for record in group), len(group)),
            "avg_prompt_tokens_approx": round(
                sum(record["prompt_tokens_approx"] for record in group) / len(group), 2
            ),
            "avg_cost_normalized_task_success": round(
                sum(record["cost_normalized_task_success"] for record in group) / len(group), 4
            ),
            "wrong_turn_recovery_rate": format_ratio(
                sum(record["wrong_turn_recovery"] for record in overturn_group),
                len(overturn_group),
            ),
            "stale_belief_persistence": format_ratio(
                sum(record["stale_belief_persistence"] for record in overturn_group),
                len(overturn_group),
            ),
            "assistant_assumption_survival": format_ratio(
                sum(record["assistant_assumption_survival"] for record in overturn_group),
                len(overturn_group),
            ),
            "correction_uptake": format_ratio(
                sum(record["correction_uptake"] for record in overturn_group),
                len(overturn_group),
            ),
            "deprecated_belief_leakage": format_ratio(
                sum(record["deprecated_belief_leakage"] for record in group),
                len(group),
            ),
        }
        rows.append(row)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def render_markdown_summary(rows: list[dict]) -> str:
    header = (
        "| system | condition | n | accuracy | wrong_turn_recovery_rate | "
        "stale_belief_persistence | assistant_assumption_survival | "
        "correction_uptake | avg_prompt_tokens_approx |\n"
    )
    separator = "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    body = ""
    for row in rows:
        body += (
            f"| {row['system']} | {row['condition']} | {row['n']} | "
            f"{row['accuracy']} | {row['wrong_turn_recovery_rate']} | "
            f"{row['stale_belief_persistence']} | "
            f"{row['assistant_assumption_survival']} | "
            f"{row['correction_uptake']} | {row['avg_prompt_tokens_approx']} |\n"
        )
    return header + separator + body
