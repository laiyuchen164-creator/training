from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import transform_reviseqa_incremental
from src.llm_client import ChatAPIClient
from src.metrics import enrich_record, render_markdown_summary, write_summary
from src.utils import read_json, read_jsonl, write_json, write_jsonl


def build_messages(example: dict) -> list[dict]:
    options = example["answer_options"]
    if example["condition"] == "full_info":
        context = "Premises after all edits:\n" + "\n".join(
            f"- {premise}" for premise in example["revised_premises"]
        )
    else:
        context = (
            "Early premises:\n"
            + "\n".join(f"- {premise}" for premise in example["initial_premises"])
            + "\n\nLate evidence / edits:\n"
            + "\n".join(f"- {premise}" for premise in example["update_premises"])
        )
    return [
        {
            "role": "system",
            "content": (
                "You are solving a logical multiple-choice task. "
                "Use all provided evidence and return only compact JSON: "
                "{\"label\":\"a\"}, {\"label\":\"b\"}, or {\"label\":\"c\"}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{context}\n\n"
                f"Question:\n{example['revised_query']}\n\n"
                "Answer options:\n"
                f"(a) {options['a']}\n"
                f"(b) {options['b']}\n"
                f"(c) {options['c']}\n\n"
                "Return only the JSON object."
            ),
        },
    ]


def load_existing_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {row["example_id"]: row for row in rows}


def write_condition_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_direct_summary(records: list[dict], run_dir: Path) -> None:
    summary_rows = write_summary(records, run_dir / "summary.csv")
    (run_dir / "summary.md").write_text(
        render_markdown_summary(summary_rows),
        encoding="utf-8",
    )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["condition"]].append(record)
    condition_rows = []
    for condition, group in sorted(grouped.items()):
        correct = sum(record["final_correct"] for record in group)
        stale = sum(record["stale_belief_persistence"] for record in group)
        uptake = sum(record["correction_uptake"] for record in group)
        condition_rows.append(
            {
                "system": "direct_gpt_api",
                "condition": condition,
                "n": len(group),
                "accuracy": correct / len(group),
                "stale_belief_persistence": stale / len(group),
                "correction_uptake": uptake / len(group),
            }
        )
    write_condition_csv(run_dir / "metrics_by_condition.csv", condition_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openai", choices=["openai", "deepseek"])
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--run-dir", default="runs/reviseqa_direct_openai_gpt54mini_full_v1")
    parser.add_argument("--input-path", default="data/processed/reviseqa_incremental.jsonl")
    parser.add_argument("--raw-dir", default="data/raw/reviseqa")
    parser.add_argument("--stats-path", default="data/processed/reviseqa_incremental_stats.json")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input_path
    stats_path = PROJECT_ROOT / args.stats_path
    if not input_path.exists():
        transform_reviseqa_incremental(
            PROJECT_ROOT / args.raw_dir,
            input_path,
            stats_path,
            refresh=False,
        )

    run_dir = PROJECT_ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    examples = read_jsonl(input_path)
    predictions_by_id = load_existing_predictions(predictions_path)

    client = ChatAPIClient(
        {
            "provider": args.provider,
            "api_key_env": args.api_key_env,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "timeout_seconds": args.timeout_seconds,
            "temperature": args.temperature,
        }
    )

    for index, example in enumerate(examples, start=1):
        if example["example_id"] in predictions_by_id:
            continue
        response = client.complete(build_messages(example))
        predicted_answer = response.label if response.label in {"a", "b", "c"} else "c"
        raw_record = {
            "example_id": example["example_id"],
            "system": "direct_gpt_api",
            "condition": example["condition"],
            "gold_initial_label": example["gold_initial_label"],
            "gold_final_label": example["gold_final_label"],
            "initial_prediction": example["gold_initial_label"]
            if example["condition"].startswith("incremental_")
            else None,
            "final_prediction": predicted_answer,
            "prompt_tokens_approx": response.usage["prompt_tokens"],
            "api_calls": 1,
            "revision_signal": {
                "backend": "llm",
                "provider": args.provider,
                "model": response.model,
                "parse_mode": response.parse_mode,
                "label_from_model": response.label,
            },
            "trace": {
                "example_id": example["example_id"],
                "system": "direct_gpt_api",
                "condition": example["condition"],
                "pair_id": example["pair_id"],
                "modus": example["modus"],
                "turns": [
                    {
                        "turn_id": 1,
                        "prediction": predicted_answer,
                        "raw_output": response.content,
                        "parsed_payload": response.parsed_payload,
                        "usage": response.usage,
                        "messages": build_messages(example),
                    }
                ],
                "final_snapshot": {"beliefs": []},
                "revision_signal": {
                    "backend": "llm",
                    "provider": args.provider,
                    "model": response.model,
                    "parse_mode": response.parse_mode,
                    "label_from_model": response.label,
                },
            },
        }
        payload = enrich_record(raw_record)
        predictions_by_id[example["example_id"]] = payload
        ordered = [
            predictions_by_id[item["example_id"]]
            for item in examples
            if item["example_id"] in predictions_by_id
        ]
        write_jsonl(predictions_path, ordered)
        print(
            json.dumps(
                {
                    "completed": len(ordered),
                    "total": len(examples),
                    "index": index,
                    "example_id": example["example_id"],
                    "condition": example["condition"],
                    "final_prediction": predicted_answer,
                    "gold_final_label": example["gold_final_label"],
                    "prompt_tokens": response.usage["prompt_tokens"],
                    "completion_tokens": response.usage["completion_tokens"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    final_records = [
        predictions_by_id[item["example_id"]]
        for item in examples
        if item["example_id"] in predictions_by_id
    ]
    write_direct_summary(final_records, run_dir)
    write_json(
        run_dir / "config.json",
        {
            "provider": args.provider,
            "model": args.model,
            "api_key_env": args.api_key_env,
            "input_path": args.input_path,
            "max_tokens": args.max_tokens,
            "timeout_seconds": args.timeout_seconds,
            "temperature": args.temperature,
            "completed_examples": len(final_records),
            "dataset_stats": read_json(stats_path),
        },
    )
    print(f"Run finished: {run_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
