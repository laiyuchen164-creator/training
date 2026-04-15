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
from src.utils import read_jsonl, write_json


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Config {path} is expected to be JSON-compatible YAML in this environment."
        ) from exc


def write_condition_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    config_path = (project_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)
    run_dir = project_root / (args.run_dir or config["run_dir"])

    model = CommitmentControlModel.load(run_dir / "model.npz")
    split_examples = {
        "train": read_jsonl(project_root / config["data"]["train_path"]),
        "dev": read_jsonl(project_root / config["data"]["dev_path"]),
        "test": read_jsonl(project_root / config["data"]["test_path"]),
    }

    summary = {}
    condition_rows = []
    for split, examples in split_examples.items():
        predictions = model.predict(examples)
        summary[split] = compute_commitment_metrics(examples, predictions)
        condition_rows.extend(aggregate_condition_metrics(examples, predictions))

    write_json(run_dir / "metrics_recomputed.json", summary)
    write_condition_csv(run_dir / "metrics_by_condition_recomputed.csv", condition_rows)
    (run_dir / "summary_recomputed.md").write_text(
        render_metrics_markdown(summary, condition_rows),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
