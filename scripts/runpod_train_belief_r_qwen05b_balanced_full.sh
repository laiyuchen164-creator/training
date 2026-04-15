#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
  bash scripts/runpod_setup.sh
fi

source .venv/bin/activate

mkdir -p .cache/huggingface
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

python data/prepare_belief_r_training_assets.py
python training/train_commitment_control_hf.py --config configs/train_cipc_belief_r_qwen05b_lora_balanced_full_v1.json
