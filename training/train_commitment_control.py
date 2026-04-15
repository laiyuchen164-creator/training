from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.commitment_metrics import (
    aggregate_condition_metrics,
    compute_commitment_metrics,
    render_metrics_markdown,
)
from src.models.commitment_control_model import CommitmentControlModel
from src.utils import read_jsonl, write_json, write_jsonl


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Config {path} is expected to be JSON-compatible YAML in this environment."
        ) from exc


def write_history(path: Path, history: list) -> None:
    rows = [
        {
            "epoch": state.epoch,
            "train_loss": round(state.train_loss, 6),
            "dev_control_accuracy": round(state.dev_control_accuracy, 6),
            "dev_answer_accuracy": round(state.dev_answer_accuracy, 6),
            "dev_joint_accuracy": round(state.dev_joint_accuracy, 6),
        }
        for state in history
    ]
    write_jsonl(path, rows)


def write_condition_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def oversample_examples_by_label(examples: list[dict], *, label_key: str, seed: int) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for example in examples:
        grouped.setdefault(example[label_key], []).append(example)
    max_size = max(len(group) for group in grouped.values())
    balanced: list[dict] = []
    for label in sorted(grouped):
        group = sorted(grouped[label], key=lambda item: item["example_id"])
        repeats, remainder = divmod(max_size, len(group))
        balanced.extend(group * repeats)
        if remainder:
            balanced.extend(group[:remainder])
    balanced.sort(key=lambda item: f"{seed}:{item['example_id']}:{item[label_key]}")
    return balanced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    config_path = (project_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)
    run_dir = project_root / config["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)

    train_examples = read_jsonl(project_root / config["data"]["train_path"])
    dev_examples = read_jsonl(project_root / config["data"]["dev_path"])
    test_examples = read_jsonl(project_root / config["data"]["test_path"])
    train_eval_examples = list(train_examples)

    max_train_examples = config["training"].get("max_train_examples")
    if max_train_examples:
        train_examples = train_examples[:max_train_examples]
        train_eval_examples = train_eval_examples[:max_train_examples]
    if config["training"].get("oversample_control_labels"):
        train_examples = oversample_examples_by_label(
            train_examples,
            label_key="control_label",
            seed=config["seed"],
        )

    model = CommitmentControlModel(
        control_labels=config["labels"]["control_labels"],
        answer_labels=config["labels"]["answer_labels"],
        feature_dim=config["model"]["feature_dim"],
        seed=config["seed"],
    )

    history = model.fit(
        train_examples,
        dev_examples,
        num_epochs=config["training"]["num_epochs"],
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        l2=config["training"]["l2"],
        answer_loss_weight=config["training"]["answer_loss_weight"],
    )

    model_path = run_dir / "model.npz"
    model.save(model_path)
    write_history(run_dir / "train_log.jsonl", history)
    write_json(run_dir / "config_snapshot.json", config)

    split_examples = {
        "train": train_eval_examples,
        "dev": dev_examples,
        "test": test_examples,
    }
    split_predictions = {}
    summary = {}

    for split, examples in split_examples.items():
        predictions = model.predict(examples)
        split_predictions[split] = predictions
        write_jsonl(run_dir / f"{split}_predictions.jsonl", predictions)
        summary[split] = compute_commitment_metrics(examples, predictions)

    condition_rows = []
    for split in ("train", "dev", "test"):
        condition_rows.extend(aggregate_condition_metrics(split_examples[split], split_predictions[split]))

    write_json(run_dir / "metrics.json", summary)
    write_condition_csv(run_dir / "metrics_by_condition.csv", condition_rows)
    summary_md = render_metrics_markdown(summary, condition_rows)
    (run_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    analysis_report = config.get("analysis_report_path")
    if analysis_report:
        (project_root / analysis_report).write_text(summary_md, encoding="utf-8")

    print(f"Training run finished: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
