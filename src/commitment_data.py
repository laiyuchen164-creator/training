from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Iterable

from src.utils import format_ratio

CONTROL_LABELS = ("preserve", "weaken", "replace")
ANSWER_LABELS = ("a", "b", "c")
DEFAULT_SPLIT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}


def _pair_key(record: dict) -> str:
    return f"{record['pair_id']}::{record['modus']}"


def _render_premise_block(premises: Iterable[str]) -> str:
    lines = []
    for index, premise in enumerate(premises, start=1):
        lines.append(f"Premise {index}: {premise}")
    return "\n".join(lines)


def _render_late_evidence(premises: Iterable[str]) -> str:
    lines = []
    for index, premise in enumerate(premises, start=1):
        lines.append(f"Late Evidence {index}: {premise}")
    return "\n".join(lines)


def _control_label_from_pair(incremental_record: dict) -> str:
    if incremental_record["gold_initial_label"] == incremental_record["gold_final_label"]:
        return "preserve"
    return "replace"


def _stable_sort(values: list[str], seed: int) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.md5(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )


def _split_ids_by_label(
    pair_to_control: dict[str, str],
    *,
    seed: int,
    split_ratios: dict[str, float] | None = None,
) -> dict[str, str]:
    ratios = split_ratios or DEFAULT_SPLIT_RATIOS
    assignments: dict[str, str] = {}
    by_control: dict[str, list[str]] = defaultdict(list)
    for pair_id, label in pair_to_control.items():
        by_control[label].append(pair_id)

    for label, pair_ids in sorted(by_control.items()):
        ordered = _stable_sort(pair_ids, seed)
        total = len(ordered)
        train_count = int(total * ratios["train"])
        dev_count = int(total * ratios["dev"])
        test_count = total - train_count - dev_count

        if total >= 3 and dev_count == 0:
            dev_count = 1
            train_count = max(1, train_count - 1)
            test_count = total - train_count - dev_count
        if total >= 3 and test_count == 0:
            test_count = 1
            train_count = max(1, train_count - 1)

        cut_train = train_count
        cut_dev = train_count + dev_count
        for pair_id in ordered[:cut_train]:
            assignments[pair_id] = "train"
        for pair_id in ordered[cut_train:cut_dev]:
            assignments[pair_id] = "dev"
        for pair_id in ordered[cut_dev:]:
            assignments[pair_id] = "test"

    return assignments


def _convert_record(record: dict, *, control_label: str, split: str) -> dict:
    answer_options = record["answer_options"]
    gold_initial_label = record["gold_initial_label"]
    gold_final_label = record["gold_final_label"]
    pair_key = _pair_key(record)
    return {
        "example_id": record["example_id"],
        "condition": record["condition"],
        "early_context": _render_premise_block(record["initial_premises"]),
        "early_commitment_text": answer_options[gold_initial_label],
        "early_commitment_label": gold_initial_label,
        "late_evidence": _render_late_evidence(record["update_premises"]),
        "source_type": "assistant_inferred",
        "control_label": control_label,
        "final_answer_label": gold_final_label,
        "final_answer_text": answer_options[gold_final_label],
        "answer_options": answer_options,
        "question": record["revised_query"],
        "task_metadata": {
            "condition": record["condition"],
            "pair_id": record["pair_id"],
            "pair_key": pair_key,
            "modus": record["modus"],
            "relation_type": record["relation_type"],
            "late_source_type": "user_explicit",
        },
        "metadata": {
            "dataset": "belief_r",
            "split": split,
            "original_label": gold_initial_label,
            "pair_id": record["pair_id"],
            "pair_key": pair_key,
            "modus": record["modus"],
            "relation_type": record["relation_type"],
            "control_label_space": "binary_v1_ready_for_3way",
            "gold_initial_label": gold_initial_label,
            "gold_final_label": gold_final_label,
        },
    }


def build_belief_r_commitment_control_records(
    incremental_records: list[dict],
    *,
    seed: int = 7,
    split_ratios: dict[str, float] | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    pair_buckets: dict[str, dict[str, dict]] = defaultdict(dict)
    pair_to_control: dict[str, str] = {}
    skipped_pairs = 0

    for record in incremental_records:
        if record.get("dataset") != "belief_r":
            continue
        pair_buckets[_pair_key(record)][record["condition"]] = record

    for pair_key, bucket in pair_buckets.items():
        incremental_record = None
        for condition in ("incremental_no_overturn", "incremental_overturn_reasoning"):
            if condition in bucket:
                incremental_record = bucket[condition]
                break
        if "full_info" not in bucket or incremental_record is None:
            skipped_pairs += 1
            continue
        pair_to_control[pair_key] = _control_label_from_pair(incremental_record)

    split_assignments = _split_ids_by_label(pair_to_control, seed=seed, split_ratios=split_ratios)
    split_records: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    split_pair_counts: dict[str, int] = defaultdict(int)
    condition_counts: dict[str, Counter] = defaultdict(Counter)
    control_counts: dict[str, Counter] = defaultdict(Counter)
    answer_counts: dict[str, Counter] = defaultdict(Counter)

    for pair_key in sorted(pair_to_control):
        bucket = pair_buckets[pair_key]
        split = split_assignments[pair_key]
        split_pair_counts[split] += 1
        control_label = pair_to_control[pair_key]
        for condition in ("full_info", "incremental_no_overturn", "incremental_overturn_reasoning"):
            if condition not in bucket:
                continue
            converted = _convert_record(bucket[condition], control_label=control_label, split=split)
            split_records[split].append(converted)
            condition_counts[split][condition] += 1
            control_counts[split][control_label] += 1
            answer_counts[split][converted["final_answer_label"]] += 1

    total_examples = sum(len(records) for records in split_records.values())
    total_pairs = sum(split_pair_counts.values())
    all_control_counts = Counter()
    all_answer_counts = Counter()
    all_condition_counts = Counter()
    for split in split_records:
        all_control_counts.update(control_counts[split])
        all_answer_counts.update(answer_counts[split])
        all_condition_counts.update(condition_counts[split])

    stats = {
        "dataset": "belief_r_commitment_control",
        "seed": seed,
        "control_label_space": list(CONTROL_LABELS),
        "active_control_labels": sorted(label for label, count in all_control_counts.items() if count),
        "answer_label_space": list(ANSWER_LABELS),
        "total_pairs": total_pairs,
        "total_examples": total_examples,
        "skipped_pairs": skipped_pairs,
        "split_pair_counts": dict(split_pair_counts),
        "split_example_counts": {split: len(records) for split, records in split_records.items()},
        "overall_condition_counts": dict(all_condition_counts),
        "overall_control_counts": dict(all_control_counts),
        "overall_answer_counts": dict(all_answer_counts),
        "by_split": {
            split: {
                "pairs": split_pair_counts[split],
                "examples": len(split_records[split]),
                "condition_counts": dict(condition_counts[split]),
                "control_counts": dict(control_counts[split]),
                "answer_counts": dict(answer_counts[split]),
            }
            for split in split_records
        },
    }
    return split_records, stats


def render_commitment_control_stats(stats: dict) -> str:
    lines = [
        "# Belief-R Commitment-Control Dataset Stats",
        "",
        "## Summary",
        "",
        f"- Total pairs: `{stats['total_pairs']}`",
        f"- Total examples: `{stats['total_examples']}`",
        f"- Active control labels: `{', '.join(stats['active_control_labels'])}`",
        f"- Skipped pairs: `{stats['skipped_pairs']}`",
        "",
        "## Overall Distribution",
        "",
        f"- Conditions: `{stats['overall_condition_counts']}`",
        f"- Control labels: `{stats['overall_control_counts']}`",
        f"- Final answers: `{stats['overall_answer_counts']}`",
        "",
        "## Split Breakdown",
        "",
    ]
    for split in ("train", "dev", "test"):
        payload = stats["by_split"][split]
        pair_ratio = format_ratio(payload["pairs"], max(1, stats["total_pairs"]))
        example_ratio = format_ratio(payload["examples"], max(1, stats["total_examples"]))
        lines.extend(
            [
                f"### {split}",
                "",
                f"- Pairs: `{payload['pairs']}` (`{pair_ratio}` of total)",
                f"- Examples: `{payload['examples']}` (`{example_ratio}` of total)",
                f"- Conditions: `{payload['condition_counts']}`",
                f"- Control labels: `{payload['control_counts']}`",
                f"- Final answers: `{payload['answer_counts']}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_spotcheck_sample(
    split_records: dict[str, list[dict]],
    *,
    sample_size: int = 50,
    seed: int = 7,
) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for split, records in split_records.items():
        grouped[split].extend(sorted(records, key=lambda item: item["example_id"]))

    targets = {"train": 20, "dev": 15, "test": 15}
    sample: list[dict] = []
    for split in ("train", "dev", "test"):
        ordered = _stable_sort([record["example_id"] for record in grouped[split]], seed)
        lookup = {record["example_id"]: record for record in grouped[split]}
        for example_id in ordered[: targets[split]]:
            sample.append(lookup[example_id])
    return sample[:sample_size]


def render_spotcheck_report(sample: list[dict]) -> str:
    lines = [
        "# Belief-R Commitment-Control Spot Check",
        "",
        "This memo records a deterministic 50-example inspection slice for the",
        "first Belief-R commitment-control conversion.",
        "",
        "## High-Level Notes",
        "",
        "- `incremental_no_overturn` rows consistently map to `preserve`.",
        "- `incremental_overturn_reasoning` rows consistently map to `replace` in",
        "  the binary v1 label space.",
        "- `full_info` rows share the same control label as their paired",
        "  incremental example, so the dataset can supervise both control and",
        "  full-information answer prediction without losing pair alignment.",
        "- No `weaken` examples are introduced in v1; the label space remains",
        "  3-way-ready but the current Belief-R proof-of-concept is binary on",
        "  control decisions.",
        "",
        "## Sampled Examples",
        "",
    ]
    for index, record in enumerate(sample, start=1):
        lines.extend(
            [
                f"### Example {index}",
                "",
                f"- Example ID: `{record['example_id']}`",
                f"- Split: `{record['metadata']['split']}`",
                f"- Condition: `{record['condition']}`",
                f"- Control label: `{record['control_label']}`",
                f"- Early commitment: `{record['early_commitment_label']} :: {record['early_commitment_text']}`",
                f"- Final answer: `{record['final_answer_label']} :: {record['final_answer_text']}`",
                f"- Pair metadata: `{record['task_metadata']['modus']} / {record['task_metadata']['relation_type']}`",
                "",
                "Early context:",
                "```text",
                record["early_context"],
                "```",
                "Late evidence:",
                "```text",
                record["late_evidence"],
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
