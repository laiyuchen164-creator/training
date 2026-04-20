from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.build_reviseqa_commitment_control import main as build_reviseqa_commitment_control
from src.eval.commitment_metrics import (
    aggregate_condition_metrics,
    compute_commitment_metrics,
    render_metrics_markdown,
)
from src.models.hf_commitment_control_model import HFCommitmentControlModel
from src.utils import read_jsonl, write_json, write_jsonl
from training.train_commitment_control_hf import CommitmentDataset, collate_batch


def write_condition_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_reviseqa_commitment_control(path: Path) -> None:
    if path.exists():
        return
    old_argv = sys.argv
    try:
        sys.argv = ["build_reviseqa_commitment_control"]
        build_reviseqa_commitment_control()
    finally:
        sys.argv = old_argv


def ensure_split_metadata(examples: list[dict]) -> list[dict]:
    for example in examples:
        example.setdefault("metadata", {})
        example["metadata"].setdefault("split", "reviseqa_full")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="runs/cipc_belief_r_qwen05b_lora_highrank_v1")
    parser.add_argument("--input-path", default="data/processed/reviseqa_commitment_control_full.jsonl")
    parser.add_argument("--run-dir", default="runs/reviseqa_cipc_highrank_v1_full_eval")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    checkpoint_dir = PROJECT_ROOT / args.checkpoint_dir
    checkpoint = torch.load(checkpoint_dir / "hf_model.pt", map_location="cpu")
    config = checkpoint["config"]
    control_labels = checkpoint["control_labels"]
    answer_labels = checkpoint["answer_labels"]
    control_to_idx = {label: index for index, label in enumerate(control_labels)}
    answer_to_idx = {label: index for index, label in enumerate(answer_labels)}

    input_path = PROJECT_ROOT / args.input_path
    ensure_reviseqa_commitment_control(input_path)
    examples = ensure_split_metadata(read_jsonl(input_path))

    bundle = HFCommitmentControlModel.build_bundle(
        model_name=config["model"]["name"],
        control_label_count=len(control_labels),
        answer_label_count=len(answer_labels),
        control_to_idx=control_to_idx,
        lora_r=config["model"]["lora_r"],
        lora_alpha=config["model"]["lora_alpha"],
        lora_dropout=config["model"]["lora_dropout"],
        use_bf16=config["model"].get("use_bf16", True),
    )
    tokenizer = bundle.tokenizer
    model = bundle.model
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = CommitmentDataset(
        examples,
        tokenizer,
        config["model"]["max_length"],
        control_to_idx,
        answer_to_idx,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, tokenizer),
    )

    predictions = []
    with torch.no_grad():
        for batch in loader:
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

    run_dir = PROJECT_ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    summary = {"reviseqa_full": compute_commitment_metrics(examples, predictions)}
    condition_rows = aggregate_condition_metrics(examples, predictions)
    write_json(run_dir / "metrics.json", summary)
    write_condition_csv(run_dir / "metrics_by_condition.csv", condition_rows)
    (run_dir / "summary.md").write_text(
        render_metrics_markdown(summary, condition_rows),
        encoding="utf-8",
    )
    write_json(
        run_dir / "config.json",
        {
            "checkpoint_dir": args.checkpoint_dir,
            "input_path": args.input_path,
            "batch_size": args.batch_size,
            "source_model_config": config,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
