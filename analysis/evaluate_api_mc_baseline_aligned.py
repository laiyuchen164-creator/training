from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.commitment_metrics import (
    aggregate_condition_metrics,
    compute_commitment_metrics,
    render_metrics_markdown,
)
from src.llm_client import ChatAPIClient
from src.utils import read_jsonl, write_json, write_jsonl


def write_condition_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_messages(example: dict) -> list[dict]:
    options = example["answer_options"]
    return [
        {
            "role": "system",
            "content": (
                "You are solving Belief-R suppression-task cases under "
                "belief-revision semantics, not only strict monotonic logic. "
                "Return only compact JSON with schema "
                "{\"label\":\"a|b|c\",\"premise_role\":\"alternative_pathway|extra_requirement|contradiction|unrelated|unclear\",\"relation_to_prior\":\"confirm|elaborate|contradict|replace|unrelated|n_a\"}."
            ),
        },
        {
            "role": "user",
            "content": (
                "Task rules:\n"
                "- Use only answer labels a, b, or c.\n"
                "- Option c means the target conclusion may or may not follow after considering the late evidence.\n"
                "- Do not preserve the early commitment merely because it followed from the early context.\n"
                "- First judge what the late evidence does to the early commitment.\n"
                "- alternative_pathway: an independent second route to the same conclusion; usually preserves the early answer.\n"
                "- extra_requirement: adds a prerequisite, resource, tool, timing condition, checklist, permission, or enabling condition; usually weakens the early answer to c.\n"
                "- contradiction: directly conflicts with or reverses the early route; usually changes or weakens the answer.\n"
                "- unrelated: does not affect the early commitment; preserve the early answer.\n\n"
                "Early context:\n"
                f"{example['early_context']}\n\n"
                "Early commitment:\n"
                f"Label: {example['early_commitment_label']}\n"
                f"Text: {example['early_commitment_text']}\n\n"
                "Late evidence:\n"
                f"{example['late_evidence']}\n\n"
                "Question:\n"
                f"{example['question']}\n\n"
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


def align_predictions(examples: list[dict], predictions_by_id: dict[str, dict]) -> list[dict]:
    return [predictions_by_id[example["example_id"]] for example in examples]


def control_from_answer(predicted_answer: str, example: dict) -> str:
    if predicted_answer == example["early_commitment_label"]:
        return "preserve"
    return "replace"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["openai", "deepseek"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input-path", default="data/processed/belief_r_commitment_control_test.jsonl")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    examples = read_jsonl(PROJECT_ROOT / args.input_path)
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
        predicted_control = control_from_answer(predicted_answer, example)
        predictions_by_id[example["example_id"]] = {
            "example_id": example["example_id"],
            "condition": example["condition"],
            "predicted_control_decision": predicted_control,
            "predicted_final_answer": predicted_answer,
            "raw_content": response.content,
            "parsed_payload": response.parsed_payload,
            "parse_mode": response.parse_mode,
            "provider": args.provider,
            "model": response.model,
            "prompt_tokens": response.usage["prompt_tokens"],
            "completion_tokens": response.usage["completion_tokens"],
            "total_tokens": response.usage["total_tokens"],
            "index": index,
        }
        write_jsonl(
            predictions_path,
            [
                predictions_by_id[item["example_id"]]
                for item in examples
                if item["example_id"] in predictions_by_id
            ],
        )
        print(
            json.dumps(
                {
                    "completed": len(predictions_by_id),
                    "total": len(examples),
                    "example_id": example["example_id"],
                    "condition": example["condition"],
                    "predicted_final_answer": predicted_answer,
                    "gold_final_answer": example["final_answer_label"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    aligned_predictions = align_predictions(examples, predictions_by_id)
    summary = {"test": compute_commitment_metrics(examples, aligned_predictions)}
    condition_rows = aggregate_condition_metrics(examples, aligned_predictions)
    write_json(run_dir / "metrics.json", summary)
    write_condition_csv(run_dir / "metrics_by_condition.csv", condition_rows)
    (run_dir / "summary.md").write_text(
        render_metrics_markdown(summary, condition_rows),
        encoding="utf-8",
    )
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
            "prompt_style": "belief_r_aligned_v1",
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
