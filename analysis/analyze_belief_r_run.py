from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


AGGRESSIVE_RELATIONS = {"contradict", "replace"}
PRESERVE_RELATIONS = {"confirm", "elaborate", "unrelated"}


def _load_predictions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _label_confusion(records: list[dict]) -> list[tuple[tuple[str, str], int]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for record in records:
        counts[(record["gold_final_label"], record["final_prediction"])] += 1
    return sorted(counts.items())


def _bucket_failure(record: dict, *, expected_relation: set[str], expected_role: set[str]) -> str:
    turn2 = record["trace"]["turns"][1]
    relation = turn2.get("relation_to_prior", "missing")
    premise_role = turn2.get("premise_role", "missing")
    signal = record.get("revision_signal", {})

    if signal.get("turn2_label_from_model") is None or signal.get("turn2_parse_mode") == "unparsed":
        return "generation_instability"
    if relation not in expected_relation:
        return "wrong_relation_to_prior"
    if premise_role not in expected_role:
        return "wrong_premise_role"
    return "label_semantic_confusion"


def _representative_lines(records: list[dict], limit: int = 5) -> list[str]:
    lines = []
    for record in records[:limit]:
        turn2 = record["trace"]["turns"][1]
        lines.append(f"- `{record['example_id']}`")
        lines.append(
            f"  gold `{record['gold_initial_label']} -> {record['gold_final_label']}`, "
            f"pred `{record['initial_prediction']} -> {record['final_prediction']}`"
        )
        lines.append(
            f"  relation_to_prior `{turn2.get('relation_to_prior', 'missing')}`, "
            f"premise_role `{turn2.get('premise_role', 'missing')}`, "
            f"model_prediction `{turn2.get('model_prediction', 'missing')}`"
        )
        lines.append(f"  raw `{turn2.get('raw_output', '')[:260]}`")
    return lines


def write_report(run_dir: Path, out_path: Path) -> None:
    predictions = _load_predictions(run_dir / "predictions.jsonl")
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in predictions:
        grouped[(record["system"], record["condition"])].append(record)

    lines = [
        f"# Belief-R Run Analysis: `{run_dir.name}`\n\n",
        "## Aggregate Metrics\n\n",
        "| system | condition | n | accuracy | avg_prompt_tokens | avg_api_calls | assistant_assumption_survival | correction_uptake |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]

    for (system, condition), records in sorted(grouped.items()):
        accuracy = _avg([1.0 if record["final_correct"] else 0.0 for record in records])
        survival = _avg([1.0 if record["assistant_assumption_survival"] else 0.0 for record in records])
        uptake = _avg([1.0 if record["correction_uptake"] else 0.0 for record in records])
        avg_tokens = _avg([float(record["prompt_tokens_approx"]) for record in records])
        avg_api_calls = _avg([float(record["api_calls"]) for record in records])
        lines.append(
            f"| {system} | {condition} | {len(records)} | {accuracy} | "
            f"{avg_tokens} | {avg_api_calls} | {survival} | {uptake} |\n"
        )

    lines.append("\n## Final-Label Confusions\n\n")
    for (system, condition), records in sorted(grouped.items()):
        lines.append(f"### {system} / {condition}\n\n")
        lines.append("| gold | predicted | count |\n")
        lines.append("|---|---|---:|\n")
        for (gold, predicted), count in _label_confusion(records):
            lines.append(f"| {gold} | {predicted} | {count} |\n")
        lines.append("\n")

    lines.append("## Relation-To-Prior Distributions\n\n")
    for system in ["source_no_revision", "source_revision"]:
        lines.append(f"### {system}\n\n")
        lines.append("| condition | relation_to_prior | count |\n")
        lines.append("|---|---|---:|\n")
        system_records = [record for record in predictions if record["system"] == system and record["condition"].startswith("incremental_")]
        counter = Counter(
            (
                record["condition"],
                record["trace"]["turns"][1].get("relation_to_prior", "missing"),
            )
            for record in system_records
        )
        for (condition, relation), count in sorted(counter.items()):
            lines.append(f"| {condition} | {relation} | {count} |\n")
        lines.append("\n")

    lines.append("## Premise-Role Distributions\n\n")
    for system in ["source_no_revision", "source_revision"]:
        lines.append(f"### {system}\n\n")
        lines.append("| condition | premise_role | count |\n")
        lines.append("|---|---|---:|\n")
        system_records = [record for record in predictions if record["system"] == system and record["condition"].startswith("incremental_")]
        counter = Counter(
            (
                record["condition"],
                record["trace"]["turns"][1].get("premise_role", "missing"),
            )
            for record in system_records
        )
        for (condition, premise_role), count in sorted(counter.items()):
            lines.append(f"| {condition} | {premise_role} | {count} |\n")
        lines.append("\n")

    no_overturn_failures = [
        record
        for record in predictions
        if record["system"] == "source_revision"
        and record["condition"] == "incremental_no_overturn"
        and not record["final_correct"]
    ]
    failure_buckets = Counter(
        _bucket_failure(
            record,
            expected_relation=PRESERVE_RELATIONS,
            expected_role={"alternative_pathway"},
        )
        for record in no_overturn_failures
    )

    lines.append("## Source-Revision No-Overturn Failures\n\n")
    lines.append(f"- Total failures: {len(no_overturn_failures)}\n")
    for bucket, count in sorted(failure_buckets.items()):
        lines.append(f"- {bucket}: {count}\n")
    lines.append("\nRepresentative examples:\n")
    lines.extend(line + "\n" for line in _representative_lines(no_overturn_failures, limit=8))
    lines.append("\n")

    missed_overturns = [
        record
        for record in predictions
        if record["system"] == "source_revision"
        and record["condition"] == "incremental_overturn_reasoning"
        and not record["final_correct"]
    ]
    missed_buckets = Counter(
        _bucket_failure(
            record,
            expected_relation=AGGRESSIVE_RELATIONS,
            expected_role={"extra_requirement", "contradiction"},
        )
        for record in missed_overturns
    )

    lines.append("## Source-Revision Missed Overturns\n\n")
    lines.append(f"- Total missed overturns: {len(missed_overturns)}\n")
    for bucket, count in sorted(missed_buckets.items()):
        lines.append(f"- {bucket}: {count}\n")
    lines.append("\nRepresentative examples:\n")
    lines.extend(line + "\n" for line in _representative_lines(missed_overturns, limit=8))
    lines.append("\n")

    no_revision_missed_overturns = [
        record
        for record in predictions
        if record["system"] == "source_no_revision"
        and record["condition"] == "incremental_overturn_reasoning"
        and not record["final_correct"]
    ]
    no_revision_buckets = Counter(
        _bucket_failure(
            record,
            expected_relation=AGGRESSIVE_RELATIONS,
            expected_role={"extra_requirement", "contradiction"},
        )
        for record in no_revision_missed_overturns
    )

    lines.append("## Source-No-Revision Missed Overturns\n\n")
    lines.append(f"- Total missed overturns: {len(no_revision_missed_overturns)}\n")
    for bucket, count in sorted(no_revision_buckets.items()):
        lines.append(f"- {bucket}: {count}\n")
    lines.append("\nRepresentative examples:\n")
    lines.extend(line + "\n" for line in _representative_lines(no_revision_missed_overturns, limit=8))
    lines.append("\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def append_summary_csv(run_dir: Path, out_path: Path) -> None:
    predictions = _load_predictions(run_dir / "predictions.jsonl")
    rows = []
    for system in ["source_no_revision", "source_revision"]:
        for condition in ["incremental_no_overturn", "incremental_overturn_reasoning"]:
            records = [
                record
                for record in predictions
                if record["system"] == system and record["condition"] == condition
            ]
            relation_counter = Counter(
                record["trace"]["turns"][1].get("relation_to_prior", "missing")
                for record in records
            )
            premise_counter = Counter(
                record["trace"]["turns"][1].get("premise_role", "missing")
                for record in records
            )
            rows.append(
                {
                    "system": system,
                    "condition": condition,
                    "n": len(records),
                    "accuracy": _avg([1.0 if record["final_correct"] else 0.0 for record in records]),
                    "avg_prompt_tokens": _avg([float(record["prompt_tokens_approx"]) for record in records]),
                    "avg_api_calls": _avg([float(record["api_calls"]) for record in records]),
                    "relation_distribution": json.dumps(relation_counter, ensure_ascii=False, sort_keys=True),
                    "premise_role_distribution": json.dumps(premise_counter, ensure_ascii=False, sort_keys=True),
                }
            )

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--csv-out", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    report_out = Path(args.report_out)
    csv_out = Path(args.csv_out)
    write_report(run_dir, report_out)
    append_summary_csv(run_dir, csv_out)


if __name__ == "__main__":
    main()
