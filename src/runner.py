from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

from src.data import (
    transform_atomic_explicit_revision,
    transform_belief_r,
    transform_reviseqa_incremental,
)
from src.metrics import enrich_record, render_markdown_summary, write_summary
from src.systems import run_system_on_example
from src.utils import read_json, read_jsonl, write_json, write_jsonl


def _sample_examples(records: list[dict], config: dict) -> list[dict]:
    seed = config["seed"]
    include_conditions = set(config["dataset"]["include_conditions"])
    sample_size_per_condition = config["dataset"]["sample_size_per_condition"]
    rng = random.Random(seed)

    selected = []
    for condition in sorted(include_conditions):
        condition_records = [record for record in records if record["condition"] == condition]
        rng.shuffle(condition_records)
        selected.extend(condition_records[: sample_size_per_condition[condition]])
    return selected


def run_experiment(config_path: Path) -> dict:
    config = read_json(config_path)
    data_config = config["data"]
    backend_config = config.get("backend")
    dataset_name = data_config.get("dataset_name", "belief_r")
    processed_path = Path(data_config["processed_path"])
    stats_path = Path(data_config["stats_path"])
    raw_dir = Path(data_config["raw_dir"])

    if data_config.get("refresh_data") or not processed_path.exists():
        if dataset_name == "belief_r":
            transform_belief_r(
                raw_dir,
                processed_path,
                stats_path,
                refresh=data_config.get("refresh_data", False),
            )
        elif dataset_name == "atomic_explicit_revision":
            transform_atomic_explicit_revision(
                raw_dir,
                processed_path,
                stats_path,
                refresh=data_config.get("refresh_data", False),
                max_seed_examples=data_config.get("max_seed_examples", 600),
            )
        elif dataset_name == "reviseqa_incremental":
            transform_reviseqa_incremental(
                raw_dir,
                processed_path,
                stats_path,
                refresh=data_config.get("refresh_data", False),
                max_original_examples=data_config.get("max_original_examples"),
                max_edits_per_example=data_config.get("max_edits_per_example"),
            )
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    all_records = read_jsonl(processed_path)
    sampled_examples = _sample_examples(all_records, config)

    run_dir = Path("runs") / config["run_name"]
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    run_records = []
    prompt_overrides = config.get("prompt_overrides", {})
    for system_name in config["systems"]:
        for example in sampled_examples:
            example_payload = {**example, **prompt_overrides}
            record = run_system_on_example(
                example_payload,
                system_name,
                backend_config=backend_config,
            )
            payload = enrich_record(asdict(record))
            run_records.append(payload)
            trace_path = trace_dir / f"{system_name}__{example['example_id'].replace(':', '_')}.json"
            write_json(trace_path, payload["trace"])

    predictions_path = run_dir / "predictions.jsonl"
    write_jsonl(predictions_path, run_records)
    summary_rows = write_summary(run_records, run_dir / "summary.csv")
    (run_dir / "summary.md").write_text(
        render_markdown_summary(summary_rows),
        encoding="utf-8",
    )
    write_json(
        run_dir / "run_manifest.json",
        {
            "config": config,
            "dataset_stats": read_json(stats_path),
            "sampled_examples": len(sampled_examples),
            "prediction_path": str(predictions_path),
        },
    )
    return {
        "run_dir": str(run_dir),
        "sampled_examples": len(sampled_examples),
        "summary_rows": summary_rows,
    }
