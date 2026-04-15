from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.commitment_metrics import (
    aggregate_condition_metrics,
    compute_commitment_metrics,
    render_metrics_markdown,
)
from src.models.hf_commitment_control_model import HFCommitmentControlModel
from src.utils import read_jsonl, write_json, write_jsonl


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def deterministic_subset(examples: list[dict], max_examples: int | None, seed: int) -> list[dict]:
    if not max_examples or max_examples >= len(examples):
        return examples
    ordered = sorted(
        examples,
        key=lambda row: hashlib.md5(f"{seed}:{row['example_id']}".encode("utf-8")).hexdigest(),
    )
    return ordered[:max_examples]


def render_input_text(example: dict) -> str:
    answer_options = example["answer_options"]
    return (
        "Task: Commitment Integration and Propagation Control.\n\n"
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
        f"(a) {answer_options['a']}\n"
        f"(b) {answer_options['b']}\n"
        f"(c) {answer_options['c']}\n\n"
        "Predict:\n"
        "1. control_decision\n"
        "2. final_answer"
    )


class CommitmentDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int, control_to_idx: dict, answer_to_idx: dict):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.control_to_idx = control_to_idx
        self.answer_to_idx = answer_to_idx

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        encoded = self.tokenizer(
            render_input_text(example),
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "example": example,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "control_labels": self.control_to_idx[example["control_label"]],
            "answer_labels": self.answer_to_idx[example["final_answer_label"]],
        }


@dataclass
class TrainEpochState:
    epoch: int
    train_loss: float
    dev_control_accuracy: float
    dev_answer_accuracy: float
    dev_joint_accuracy: float


def collate_batch(batch: list[dict], tokenizer) -> dict:
    input_ids = [row["input_ids"] for row in batch]
    attention_mask = [row["attention_mask"] for row in batch]
    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding=True,
        return_tensors="pt",
    )
    return {
        "examples": [row["example"] for row in batch],
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "control_labels": torch.tensor([row["control_labels"] for row in batch], dtype=torch.long),
        "answer_labels": torch.tensor([row["answer_labels"] for row in batch], dtype=torch.long),
    }


def compute_predictions_metrics(examples: list[dict], predictions: list[dict]) -> dict:
    aligned = {prediction["example_id"]: prediction for prediction in predictions}
    aligned_predictions = [aligned[example["example_id"]] for example in examples]
    return compute_commitment_metrics(examples, aligned_predictions)


def aggregate_predictions_metrics(examples: list[dict], predictions: list[dict]) -> list[dict]:
    aligned = {prediction["example_id"]: prediction for prediction in predictions}
    aligned_predictions = [aligned[example["example_id"]] for example in examples]
    return aggregate_condition_metrics(examples, aligned_predictions)


def write_condition_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_weighted_sampler(
    examples: list[dict],
    *,
    label_key: str,
) -> WeightedRandomSampler:
    counts: dict[str, int] = {}
    for example in examples:
        label = example[label_key]
        counts[label] = counts.get(label, 0) + 1
    weights = [1.0 / counts[example[label_key]] for example in examples]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(examples),
        replacement=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)
    set_seed(config["seed"])

    run_dir = PROJECT_ROOT / config["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)

    def load_examples(split: str) -> list[dict]:
        examples = read_jsonl(PROJECT_ROOT / config["data"][f"{split}_path"])
        max_examples = config["data"].get(f"max_{split}_examples")
        return deterministic_subset(examples, max_examples, config["seed"])

    train_examples = load_examples("train")
    dev_examples = load_examples("dev")
    test_examples = load_examples("test")

    control_labels = config["labels"]["control_labels"]
    answer_labels = config["labels"]["answer_labels"]
    control_to_idx = {label: index for index, label in enumerate(control_labels)}
    answer_to_idx = {label: index for index, label in enumerate(answer_labels)}

    bundle = HFCommitmentControlModel.build_bundle(
        model_name=config["model"]["name"],
        control_label_count=len(control_labels),
        answer_label_count=len(answer_labels),
        lora_r=config["model"]["lora_r"],
        lora_alpha=config["model"]["lora_alpha"],
        lora_dropout=config["model"]["lora_dropout"],
        use_bf16=config["model"].get("use_bf16", True),
    )
    tokenizer = bundle.tokenizer
    model = bundle.model
    if config["model"].get("gradient_checkpointing", False):
        model.backbone.gradient_checkpointing_enable()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = CommitmentDataset(
        train_examples,
        tokenizer,
        config["model"]["max_length"],
        control_to_idx,
        answer_to_idx,
    )
    dev_dataset = CommitmentDataset(
        dev_examples,
        tokenizer,
        config["model"]["max_length"],
        control_to_idx,
        answer_to_idx,
    )
    test_dataset = CommitmentDataset(
        test_examples,
        tokenizer,
        config["model"]["max_length"],
        control_to_idx,
        answer_to_idx,
    )

    collate = lambda batch: collate_batch(batch, tokenizer)
    train_sampler = None
    if config["training"].get("oversample_control_labels", False):
        train_sampler = build_weighted_sampler(train_examples, label_key="control_label")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        collate_fn=collate,
    )

    control_counts = np.bincount(
        [control_to_idx[example["control_label"]] for example in train_examples],
        minlength=len(control_labels),
    )
    control_class_weights = torch.tensor(
        [
            len(train_examples) / (len(control_labels) * count) if count else 0.0
            for count in control_counts
        ],
        dtype=torch.float32,
        device=device,
    )
    answer_counts = np.bincount(
        [answer_to_idx[example["final_answer_label"]] for example in train_examples],
        minlength=len(answer_labels),
    )
    answer_class_weights = torch.tensor(
        [
            len(train_examples) / (len(answer_labels) * count) if count else 0.0
            for count in answer_counts
        ],
        dtype=torch.float32,
        device=device,
    )

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config["training"].get("use_fp16_scaler", False))
    history = []
    best_score = -1.0
    best_state = None
    grad_accumulation = config["training"].get("gradient_accumulation_steps", 1)

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            control_labels_tensor = batch["control_labels"].to(device)
            answer_labels_tensor = batch["answer_labels"].to(device)

            autocast_enabled = device.type == "cuda"
            autocast_dtype = torch.bfloat16 if config["model"].get("use_bf16", True) else torch.float16
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    control_labels=control_labels_tensor,
                    answer_labels=answer_labels_tensor,
                    control_class_weights=control_class_weights,
                    answer_class_weights=answer_class_weights,
                    answer_loss_weight=config["training"]["answer_loss_weight"],
                )
                loss = outputs["loss"] / grad_accumulation

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % grad_accumulation == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            total_loss += float(outputs["loss"].detach().cpu())

        dev_predictions = []
        model.eval()
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                ctrl_pred = outputs["control_logits"].argmax(dim=-1).cpu().tolist()
                ans_pred = outputs["answer_logits"].argmax(dim=-1).cpu().tolist()
                for example, ctrl_idx, ans_idx in zip(batch["examples"], ctrl_pred, ans_pred):
                    dev_predictions.append(
                        {
                            "example_id": example["example_id"],
                            "condition": example["condition"],
                            "predicted_control_decision": control_labels[ctrl_idx],
                            "predicted_final_answer": answer_labels[ans_idx],
                        }
                    )
        dev_metrics = compute_predictions_metrics(dev_examples, dev_predictions)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total_loss / max(1, len(train_loader)), 6),
                "dev_control_accuracy": dev_metrics["control_decision_accuracy"],
                "dev_answer_accuracy": dev_metrics["final_answer_accuracy"],
                "dev_joint_accuracy": dev_metrics["joint_accuracy"],
            }
        )
        score = 0.5 * (
            dev_metrics["control_decision_accuracy"] + dev_metrics["final_answer_accuracy"]
        )
        if score > best_score:
            best_score = score
            best_state = {
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "epoch": epoch,
                "score": score,
            }

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    tokenizer.save_pretrained(run_dir / "tokenizer")
    model.backbone.save_pretrained(run_dir / "adapter")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "control_labels": control_labels,
            "answer_labels": answer_labels,
            "config": config,
            "best_state": best_state,
        },
        run_dir / "hf_model.pt",
    )
    write_json(run_dir / "config_snapshot.json", config)
    write_jsonl(run_dir / "train_log.jsonl", history)

    split_examples = {"train": train_examples, "dev": dev_examples, "test": test_examples}
    split_loaders = {"train": train_eval_loader, "dev": dev_loader, "test": test_loader}
    summary = {}
    condition_rows = []
    for split in ("train", "dev", "test"):
        predictions = []
        model.eval()
        with torch.no_grad():
            for batch in split_loaders[split]:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                ctrl_pred = outputs["control_logits"].argmax(dim=-1).cpu().tolist()
                ans_pred = outputs["answer_logits"].argmax(dim=-1).cpu().tolist()
                for example, ctrl_idx, ans_idx in zip(batch["examples"], ctrl_pred, ans_pred):
                    predictions.append(
                        {
                            "example_id": example["example_id"],
                            "condition": example["condition"],
                            "predicted_control_decision": control_labels[ctrl_idx],
                            "predicted_final_answer": answer_labels[ans_idx],
                            "gold_control_decision": example["control_label"],
                            "gold_final_answer": example["final_answer_label"],
                            "output_json": {
                                "control_decision": control_labels[ctrl_idx],
                                "final_answer": answer_labels[ans_idx],
                            },
                        }
                    )
        write_jsonl(run_dir / f"{split}_predictions.jsonl", predictions)
        summary[split] = compute_predictions_metrics(split_examples[split], predictions)
        rows = aggregate_predictions_metrics(split_examples[split], predictions)
        condition_rows.extend(rows)

    write_json(run_dir / "metrics.json", summary)
    write_condition_csv(run_dir / "metrics_by_condition.csv", condition_rows)
    summary_md = render_metrics_markdown(summary, condition_rows)
    (run_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    analysis_report = config.get("analysis_report_path")
    if analysis_report:
        (PROJECT_ROOT / analysis_report).write_text(summary_md, encoding="utf-8")

    print(f"HF training run finished: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
