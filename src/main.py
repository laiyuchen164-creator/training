from __future__ import annotations

import argparse
from pathlib import Path

from src.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Belief-R pilot.")
    parser.add_argument("--config", required=True, help="Path to the experiment config JSON.")
    args = parser.parse_args()
    result = run_experiment(Path(args.config))
    print(f"Run finished: {result['run_dir']}")
    print(f"Sampled examples: {result['sampled_examples']}")


if __name__ == "__main__":
    main()
