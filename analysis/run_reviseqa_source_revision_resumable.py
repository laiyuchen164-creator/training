from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import transform_reviseqa_incremental
from src.metrics import enrich_record, render_markdown_summary, write_summary
from src.runner import _sample_examples
from src.systems import run_system_on_example
from src.utils import read_json, read_jsonl, write_json, write_jsonl


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {row["example_id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--system", default="source_revision")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = read_json(config_path)
    data_config = config["data"]
    processed_path = PROJECT_ROOT / data_config["processed_path"]
    stats_path = PROJECT_ROOT / data_config["stats_path"]
    raw_dir = PROJECT_ROOT / data_config["raw_dir"]

    if data_config.get("refresh_data") or not processed_path.exists():
        transform_reviseqa_incremental(
            raw_dir,
            processed_path,
            stats_path,
            refresh=data_config.get("refresh_data", False),
            max_original_examples=data_config.get("max_original_examples"),
            max_edits_per_example=data_config.get("max_edits_per_example"),
        )

    all_records = read_jsonl(processed_path)
    sampled_examples = _sample_examples(all_records, config)
    run_dir = PROJECT_ROOT / "runs" / config["run_name"]
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    predictions_by_id = load_existing(predictions_path)

    for index, example in enumerate(sampled_examples, start=1):
        if example["example_id"] in predictions_by_id:
            continue
        record = run_system_on_example(
            example,
            args.system,
            backend_config=config.get("backend"),
        )
        payload = enrich_record(asdict(record))
        predictions_by_id[example["example_id"]] = payload
        trace_path = trace_dir / f"{args.system}__{example['example_id'].replace(':', '_')}.json"
        write_json(trace_path, payload["trace"])
        ordered = [
            predictions_by_id[item["example_id"]]
            for item in sampled_examples
            if item["example_id"] in predictions_by_id
        ]
        write_jsonl(predictions_path, ordered)
        print(
            json.dumps(
                {
                    "completed": len(ordered),
                    "total": len(sampled_examples),
                    "index": index,
                    "example_id": example["example_id"],
                    "condition": example["condition"],
                    "final_prediction": payload["final_prediction"],
                    "gold_final_label": payload["gold_final_label"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    final_records = [
        predictions_by_id[item["example_id"]]
        for item in sampled_examples
        if item["example_id"] in predictions_by_id
    ]
    summary_rows = write_summary(final_records, run_dir / "summary.csv")
    (run_dir / "summary.md").write_text(
        render_markdown_summary(summary_rows),
        encoding="utf-8",
    )
    write_json(
        run_dir / "run_manifest.json",
        {
            "config": config,
            "system": args.system,
            "dataset_stats": read_json(stats_path),
            "sampled_examples": len(sampled_examples),
            "completed_examples": len(final_records),
            "prediction_path": str(predictions_path.relative_to(PROJECT_ROOT)),
        },
    )
    print(f"Run finished: {run_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
