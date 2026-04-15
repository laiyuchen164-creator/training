from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    args = parser.parse_args()

    commitment_path = (
        PROJECT_ROOT / "data" / "processed" / f"belief_r_commitment_control_{args.split}.jsonl"
    )
    original_path = PROJECT_ROOT / "data" / "processed" / "belief_r_incremental.jsonl"
    output_path = (
        PROJECT_ROOT / "data" / "processed" / f"belief_r_incremental_commitment_{args.split}_subset.jsonl"
    )
    stats_path = (
        PROJECT_ROOT / "data" / "processed" / f"belief_r_incremental_commitment_{args.split}_subset_stats.json"
    )

    commitment_records = read_jsonl(commitment_path)
    wanted_ids = {record["example_id"] for record in commitment_records}
    original_records = read_jsonl(original_path)
    subset = [record for record in original_records if record["example_id"] in wanted_ids]
    subset.sort(key=lambda record: record["example_id"])

    counts = Counter(record["condition"] for record in subset)
    stats = {
        "source_split": args.split,
        "subset_examples": len(subset),
        "condition_counts": dict(counts),
    }
    write_jsonl(output_path, subset)
    write_json(stats_path, stats)
    print(f"Wrote subset: {output_path}")
    print(stats)


if __name__ == "__main__":
    main()
