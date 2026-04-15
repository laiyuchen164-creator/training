#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python was not found. Install Python 3.11+ on the Runpod image first." >&2
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
then
  echo "Python 3.11+ is required." >&2
  exit 1
fi

create_or_replace_venv() {
  rm -rf .venv
  "$PYTHON_BIN" -m venv .venv --system-site-packages
}

if [ ! -f ".venv/bin/activate" ]; then
  create_or_replace_venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

if ! python - <<'PY' >/dev/null 2>&1
import torch
PY
then
  deactivate || true
  create_or_replace_venv
  source .venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel
fi

python - <<'PY'
import sys

try:
    import torch
except Exception:
    print(
        "PyTorch is not installed in this environment. Start from a Runpod PyTorch/CUDA image first.",
        file=sys.stderr,
    )
    raise SystemExit(1)

if not torch.cuda.is_available():
    print(
        "CUDA is not available. Use a GPU-enabled Runpod image before launching training.",
        file=sys.stderr,
    )
    raise SystemExit(1)

print(f"Using torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
PY

pip install -r requirements-runpod.txt
pip install -e . --no-deps
python data/prepare_belief_r_training_assets.py
python -m unittest discover -s tests

echo "Runpod setup finished."
