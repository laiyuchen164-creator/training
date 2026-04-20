from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.commitment_data import _render_late_evidence, _render_premise_block
from src.data import transform_reviseqa_incremental
from src.utils import read_jsonl, write_json, write_jsonl


def convert_record(record: dict) -> dict:
    answer_options = record["answer_options"]
    gold_initial_label = record["gold_initial_label"]
    gold_final_label = record["gold_final_label"]
    control_label = "preserve" if gold_initial_label == gold_final_label else "replace"
    return {
        "example_id": record["example_id"],
        "condition": record["condition"],
        "early_context": _render_premise_block(record["initial_premises"]),
        "early_commitment_text": answer_options[gold_initial_label],
        "early_commitment_label": gold_initial_label,
        "late_evidence": _render_late_evidence(record["update_premises"]),
        "source_type": "user_explicit",
        "control_label": control_label,
        "final_answer_label": gold_final_label,
        "final_answer_text": answer_options[gold_final_label],
        "answer_options": answer_options,
        "question": record["revised_query"],
        "task_metadata": {
            "condition": record["condition"],
            "pair_id": record["pair_id"],
            "modus": record["modus"],
            "relation_type": record["relation_type"],
            "late_source_type": "user_explicit",
        },
        "metadata": {
            "dataset": "reviseqa_incremental",
            "split": "reviseqa_full",
            "original_label": gold_initial_label,
            "pair_id": record["pair_id"],
            "modus": record["modus"],
            "relation_type": record["relation_type"],
            "control_label_space": "binary_v1_ready_for_3way",
            "gold_initial_label": gold_initial_label,
            "gold_final_label": gold_final_label,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default="data/processed/reviseqa_incremental.jsonl")
    parser.add_argument("--output-path", default="data/processed/reviseqa_commitment_control_full.jsonl")
    parser.add_argument("--stats-path", default="data/processed/reviseqa_commitment_control_full_stats.json")
    parser.add_argument("--raw-dir", default="data/raw/reviseqa")
    parser.add_argument("--source-stats-path", default="data/processed/reviseqa_incremental_stats.json")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input_path
    if not input_path.exists():
        transform_reviseqa_incremental(
            PROJECT_ROOT / args.raw_dir,
            input_path,
            PROJECT_ROOT / args.source_stats_path,
            refresh=False,
        )

    records = read_jsonl(input_path)
    converted = [convert_record(record) for record in records]
    write_jsonl(PROJECT_ROOT / args.output_path, converted)

    stats = {
        "dataset": "reviseqa_commitment_control_full",
        "source_path": args.input_path,
        "total_examples": len(converted),
        "condition_counts": {},
        "control_counts": {},
        "answer_counts": {},
    }
    for row in converted:
        stats["condition_counts"][row["condition"]] = stats["condition_counts"].get(row["condition"], 0) + 1
        stats["control_counts"][row["control_label"]] = stats["control_counts"].get(row["control_label"], 0) + 1
        stats["answer_counts"][row["final_answer_label"]] = stats["answer_counts"].get(row["final_answer_label"], 0) + 1
    write_json(PROJECT_ROOT / args.stats_path, stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
