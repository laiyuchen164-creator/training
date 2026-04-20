from __future__ import annotations

import argparse
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import sys
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.evaluate_reviseqa_direct_api_baseline import (
    build_messages,
    load_existing_predictions,
    write_direct_summary,
)
from src.llm_client import ChatAPIClient
from src.metrics import enrich_record
from src.utils import read_json, read_jsonl, write_json, write_jsonl


_thread_local = threading.local()


def get_client(args: argparse.Namespace) -> ChatAPIClient:
    client = getattr(_thread_local, "client", None)
    if client is None:
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
        _thread_local.client = client
    return client


def run_one(example: dict, args: argparse.Namespace) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = get_client(args).complete(build_messages(example))
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
            return enrich_record(raw_record)
        except Exception as exc:  # noqa: BLE001 - keep resumable long run alive
            last_error = exc
            if attempt >= args.max_retries:
                break
            time.sleep(args.retry_sleep_seconds * attempt)
    raise RuntimeError(f"Failed {example['example_id']} after {args.max_retries} attempts: {last_error}")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openai", choices=["openai", "deepseek"])
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--run-dir", default="runs/reviseqa_direct_openai_gpt54mini_full_v1")
    parser.add_argument("--input-path", default="data/processed/reviseqa_incremental.jsonl")
    parser.add_argument("--stats-path", default="data/processed/reviseqa_incremental_stats.json")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep-seconds", type=float, default=3.0)
    parser.add_argument("--flush-every", type=int, default=100)
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    examples = read_jsonl(PROJECT_ROOT / args.input_path)
    predictions_by_id = load_existing_predictions(predictions_path)
    remaining = [example for example in examples if example["example_id"] not in predictions_by_id]
    completed = len(predictions_by_id)
    total = len(examples)
    print(json.dumps({"already_completed": completed, "remaining": len(remaining), "total": total}), flush=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        pending = {}
        iterator = iter(remaining)

        def submit_next() -> None:
            try:
                example = next(iterator)
            except StopIteration:
                return
            future = executor.submit(run_one, example, args)
            pending[future] = example

        for _ in range(min(args.max_workers * 2, len(remaining))):
            submit_next()

        since_flush = 0
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                example = pending.pop(future)
                payload = future.result()
                predictions_by_id[example["example_id"]] = payload
                append_jsonl(predictions_path, payload)
                completed += 1
                since_flush += 1
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": total,
                            "example_id": example["example_id"],
                            "condition": example["condition"],
                            "final_prediction": payload["final_prediction"],
                            "gold_final_label": payload["gold_final_label"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                submit_next()
            if since_flush >= args.flush_every:
                ordered = [
                    predictions_by_id[item["example_id"]]
                    for item in examples
                    if item["example_id"] in predictions_by_id
                ]
                write_jsonl(predictions_path, ordered)
                since_flush = 0

    ordered = [
        predictions_by_id[item["example_id"]]
        for item in examples
        if item["example_id"] in predictions_by_id
    ]
    write_jsonl(predictions_path, ordered)
    write_direct_summary(ordered, run_dir)
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
            "max_workers": args.max_workers,
            "completed_examples": len(ordered),
            "dataset_stats": read_json(PROJECT_ROOT / args.stats_path),
        },
    )
    print(f"Run finished: {run_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
