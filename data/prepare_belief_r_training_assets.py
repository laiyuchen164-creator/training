from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.commitment_data import (
    build_belief_r_commitment_control_records,
    build_spotcheck_sample,
    render_commitment_control_stats,
    render_spotcheck_report,
)
from src.data import transform_belief_r
from src.utils import read_jsonl, write_json, write_jsonl


def ensure_belief_r_incremental() -> None:
    raw_dir = PROJECT_ROOT / "data" / "raw" / "belief_r"
    processed_path = PROJECT_ROOT / "data" / "processed" / "belief_r_incremental.jsonl"
    stats_path = PROJECT_ROOT / "data" / "processed" / "belief_r_incremental_stats.json"
    if processed_path.exists() and stats_path.exists():
        return
    transform_belief_r(raw_dir, processed_path, stats_path, refresh=False)


def ensure_commitment_control_splits() -> None:
    train_path = PROJECT_ROOT / "data" / "processed" / "belief_r_commitment_control_train.jsonl"
    dev_path = PROJECT_ROOT / "data" / "processed" / "belief_r_commitment_control_dev.jsonl"
    test_path = PROJECT_ROOT / "data" / "processed" / "belief_r_commitment_control_test.jsonl"
    stats_json_path = PROJECT_ROOT / "data" / "processed" / "belief_r_commitment_control_stats.json"
    stats_md_path = PROJECT_ROOT / "analysis" / "belief_r_commitment_control_stats.md"
    spotcheck_path = PROJECT_ROOT / "analysis" / "belief_r_commitment_control_spotcheck.md"
    if (
        train_path.exists()
        and dev_path.exists()
        and test_path.exists()
        and stats_json_path.exists()
        and stats_md_path.exists()
        and spotcheck_path.exists()
    ):
        return

    incremental_path = PROJECT_ROOT / "data" / "processed" / "belief_r_incremental.jsonl"
    incremental_records = read_jsonl(incremental_path)
    split_records, stats = build_belief_r_commitment_control_records(incremental_records, seed=7)

    write_jsonl(train_path, split_records["train"])
    write_jsonl(dev_path, split_records["dev"])
    write_jsonl(test_path, split_records["test"])
    write_json(stats_json_path, stats)
    stats_md_path.write_text(render_commitment_control_stats(stats), encoding="utf-8")

    sample = build_spotcheck_sample(split_records, sample_size=50, seed=7)
    spotcheck_path.write_text(render_spotcheck_report(sample), encoding="utf-8")


def ensure_commitment_eval_subset(split: str = "test") -> None:
    subset_path = (
        PROJECT_ROOT / "data" / "processed" / f"belief_r_incremental_commitment_{split}_subset.jsonl"
    )
    stats_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / f"belief_r_incremental_commitment_{split}_subset_stats.json"
    )
    if subset_path.exists() and stats_path.exists():
        return

    commitment_path = (
        PROJECT_ROOT / "data" / "processed" / f"belief_r_commitment_control_{split}.jsonl"
    )
    original_path = PROJECT_ROOT / "data" / "processed" / "belief_r_incremental.jsonl"

    commitment_records = read_jsonl(commitment_path)
    original_records = read_jsonl(original_path)
    wanted_ids = {record["example_id"] for record in commitment_records}
    subset = [record for record in original_records if record["example_id"] in wanted_ids]
    subset.sort(key=lambda record: record["example_id"])

    counts: dict[str, int] = {}
    for record in subset:
        condition = record["condition"]
        counts[condition] = counts.get(condition, 0) + 1

    write_jsonl(subset_path, subset)
    write_json(
        stats_path,
        {
            "source_split": split,
            "subset_examples": len(subset),
            "condition_counts": counts,
        },
    )


def main() -> None:
    ensure_belief_r_incremental()
    ensure_commitment_control_splits()
    ensure_commitment_eval_subset("test")
    print("Belief-R training assets are ready.")


if __name__ == "__main__":
    main()
