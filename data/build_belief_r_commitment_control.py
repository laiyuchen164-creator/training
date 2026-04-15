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
from src.utils import read_jsonl, write_json, write_jsonl


def main() -> None:
    project_root = PROJECT_ROOT
    source_path = project_root / "data" / "processed" / "belief_r_incremental.jsonl"
    train_path = project_root / "data" / "processed" / "belief_r_commitment_control_train.jsonl"
    dev_path = project_root / "data" / "processed" / "belief_r_commitment_control_dev.jsonl"
    test_path = project_root / "data" / "processed" / "belief_r_commitment_control_test.jsonl"
    stats_json_path = project_root / "data" / "processed" / "belief_r_commitment_control_stats.json"
    stats_md_path = project_root / "analysis" / "belief_r_commitment_control_stats.md"
    spotcheck_path = project_root / "analysis" / "belief_r_commitment_control_spotcheck.md"

    incremental_records = read_jsonl(source_path)
    split_records, stats = build_belief_r_commitment_control_records(incremental_records, seed=7)

    write_jsonl(train_path, split_records["train"])
    write_jsonl(dev_path, split_records["dev"])
    write_jsonl(test_path, split_records["test"])
    write_json(stats_json_path, stats)
    stats_md_path.write_text(render_commitment_control_stats(stats), encoding="utf-8")

    sample = build_spotcheck_sample(split_records, sample_size=50, seed=7)
    spotcheck_path.write_text(render_spotcheck_report(sample), encoding="utf-8")

    print("Belief-R commitment-control dataset built.")
    print(f"Train: {len(split_records['train'])}")
    print(f"Dev: {len(split_records['dev'])}")
    print(f"Test: {len(split_records['test'])}")


if __name__ == "__main__":
    main()
