from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.commitment_metrics import (
    aggregate_condition_metrics,
    compute_commitment_metrics,
    render_metrics_markdown,
)
from src.utils import read_jsonl, write_json, write_jsonl


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ANSWER_LABELS = ("a", "b", "c")


def write_condition_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_prompt(example: dict) -> str:
    options = example["answer_options"]
    return (
        "You are evaluating an incremental reasoning example.\n"
        "Read the early context, the early commitment, and the late evidence.\n"
        "Choose the final answer option that necessarily follows after considering the late evidence.\n"
        "Answer with exactly one lowercase letter: a, b, or c.\n\n"
        "Early context:\n"
        f"{example['early_context']}\n\n"
        "Early commitment:\n"
        f"{example['early_commitment_text']}\n\n"
        "Late evidence:\n"
        f"{example['late_evidence']}\n\n"
        "Question:\n"
        f"{example['question']}\n\n"
        "Options:\n"
        f"(a) {options['a']}\n"
        f"(b) {options['b']}\n"
        f"(c) {options['c']}\n\n"
        "Final answer letter:"
    )


def build_messages(example: dict) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You solve logical multiple-choice tasks. Return only a single lowercase answer letter.",
        },
        {
            "role": "user",
            "content": render_prompt(example),
        },
    ]


def prepare_prompts(tokenizer, examples: list[dict]) -> list[str]:
    prompts = []
    for example in examples:
        messages = build_messages(example)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


def score_candidate_batch(
    *,
    tokenizer,
    model,
    prompts: list[str],
    candidate_text: str,
    max_length: int,
    device: torch.device,
) -> list[float]:
    candidate_ids = tokenizer.encode(candidate_text, add_special_tokens=False)
    if not candidate_ids:
        raise ValueError(f"Candidate text {candidate_text!r} produced no tokens.")

    prompt_encodings = tokenizer(
        prompts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - len(candidate_ids),
        padding=False,
        return_attention_mask=False,
    )

    sequences = []
    prompt_lengths = []
    for input_ids in prompt_encodings["input_ids"]:
        prompt_lengths.append(len(input_ids))
        sequences.append(input_ids + candidate_ids)

    padded = tokenizer.pad(
        {"input_ids": sequences},
        padding=True,
        return_tensors="pt",
    )
    input_ids = padded["input_ids"].to(device)
    attention_mask = padded["attention_mask"].to(device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits, dim=-1)

    scores: list[float] = []
    for row_index, sequence in enumerate(sequences):
        prompt_len = prompt_lengths[row_index]
        score = 0.0
        for offset, token_id in enumerate(candidate_ids):
            token_position = prompt_len + offset
            score += float(log_probs[row_index, token_position - 1, token_id].item())
        scores.append(score)
    return scores


def predict_batch(
    *,
    tokenizer,
    model,
    examples: list[dict],
    max_length: int,
    device: torch.device,
) -> list[dict]:
    prompts = prepare_prompts(tokenizer, examples)
    candidate_scores = {
        label: score_candidate_batch(
            tokenizer=tokenizer,
            model=model,
            prompts=prompts,
            candidate_text=f" {label}",
            max_length=max_length,
            device=device,
        )
        for label in ANSWER_LABELS
    }

    predictions = []
    for index, example in enumerate(examples):
        per_label_scores = {label: candidate_scores[label][index] for label in ANSWER_LABELS}
        predicted_answer = max(per_label_scores, key=per_label_scores.get)
        predicted_control = (
            "preserve"
            if predicted_answer == example["early_commitment_label"]
            else "replace"
        )
        predictions.append(
            {
                "example_id": example["example_id"],
                "predicted_control_decision": predicted_control,
                "predicted_final_answer": predicted_answer,
                "score_a": round(per_label_scores["a"], 4),
                "score_b": round(per_label_scores["b"], 4),
                "score_c": round(per_label_scores["c"], 4),
            }
        )
    return predictions


def evaluate_split(
    *,
    tokenizer,
    model,
    examples: list[dict],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> list[dict]:
    predictions: list[dict] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        predictions.extend(
            predict_batch(
                tokenizer=tokenizer,
                model=model,
                examples=batch,
                max_length=max_length,
                device=device,
            )
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument(
        "--run-dir",
        default="runs/frozen_qwen25_05b_mc_baseline_v1",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test"],
    )
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    split_examples = {
        split: read_jsonl(PROJECT_ROOT / f"data/processed/belief_r_commitment_control_{split}.jsonl")
        for split in args.splits
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    summary = {}
    condition_rows = []
    all_predictions = []
    for split, examples in split_examples.items():
        predictions = evaluate_split(
            tokenizer=tokenizer,
            model=model,
            examples=examples,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
        )
        all_predictions.extend(
            [{**prediction, "split": split} for prediction in predictions]
        )
        summary[split] = compute_commitment_metrics(examples, predictions)
        condition_rows.extend(aggregate_condition_metrics(examples, predictions))

    write_json(run_dir / "metrics.json", summary)
    write_condition_csv(run_dir / "metrics_by_condition.csv", condition_rows)
    write_jsonl(run_dir / "predictions.jsonl", all_predictions)
    (run_dir / "summary.md").write_text(
        render_metrics_markdown(summary, condition_rows),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "splits": args.splits,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
