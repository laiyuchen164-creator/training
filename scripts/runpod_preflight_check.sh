#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
  bash scripts/runpod_setup.sh
fi

source .venv/bin/activate

python - <<'PY'
from pathlib import Path
import json
import torch

required_files = [
    "data/processed/belief_r_incremental.jsonl",
    "data/processed/belief_r_incremental_stats.json",
    "data/processed/belief_r_commitment_control_train.jsonl",
    "data/processed/belief_r_commitment_control_dev.jsonl",
    "data/processed/belief_r_commitment_control_test.jsonl",
]

missing = [path for path in required_files if not Path(path).exists()]
if missing:
    raise SystemExit("Missing required data files: " + ", ".join(missing))

config = json.loads(
    Path("configs/train_cipc_belief_r_qwen05b_lora_balanced_full_v1.json").read_text(
        encoding="utf-8"
    )
)
for split_key in ("train_path", "dev_path", "test_path"):
    split_path = Path(config["data"][split_key])
    if not split_path.exists():
        raise SystemExit(f"Missing config data path: {split_path}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available.")

print("Preflight OK")
print(f"torch={torch.__version__}")
print(f"gpu={torch.cuda.get_device_name(0)}")
for split_key in ("train_path", "dev_path", "test_path"):
    split_path = Path(config["data"][split_key])
    print(f"{split_key}={split_path} size={split_path.stat().st_size}")
PY
