# Source-Aware Belief Revision

This repository now contains two layers of the EMNLP project:

- a frozen prompt-based diagnosis track on Belief-R
- an active training-based `CIPC` track for commitment integration and
  propagation control

Current scope:

- Real benchmark data: Belief-R strong paired subset.
- Prompt-based baselines: `raw_history`, `running_summary`,
  `structured_no_source`, `source_no_revision`, frozen `source_revision`.
- Training path: Belief-R commitment-control train/dev/test exports plus a
  local proof-of-concept multitask trainer.
- Outputs: transformed JSONL data, prediction logs, mechanism metrics,
  training/evaluation reports, aggregate summary tables, and progress notes.

The prompt-based pilot remains available for diagnosis. The active method work
has now moved to the training pipeline described in
`analysis/cipc_belief_r_lora_v1_report.md`.

## Quick Start

```bash
python -m unittest discover -s tests
python -m src.main --config configs/pilot_belief_r.json
```

After the run finishes, inspect:

- `runs/belief_r_pilot_v1/predictions.jsonl`
- `runs/belief_r_pilot_v1/summary.csv`
- `runs/belief_r_pilot_v1/summary.md`
- `runs/belief_r_pilot_v1/traces/`
- `docs/progress_report.md`

## API Pilot

The repo now supports a small API-backed pilot using chat-completions compatible
providers.

OpenAI:

```powershell
$env:OPENAI_API_KEY="..."
python -m src.main --config configs/api_pilot_belief_r_openai_small.json
python -m src.main --config configs/api_pilot_belief_r_openai_medium.json
```

DeepSeek:

```powershell
$env:DEEPSEEK_API_KEY="..."
python -m src.main --config configs/api_pilot_belief_r_deepseek_small.json
```

The API pilot keeps the same experiment interface and trace format as the
heuristic pilot, but replaces the answer backend with real model calls.

## Commitment-Control Training

Build the supervised Belief-R commitment-control dataset:

```bash
python data/build_belief_r_commitment_control.py
```

Run the first local proof-of-concept training cycle:

```bash
python training/train_commitment_control.py --config configs/train_cipc_belief_r_lora_v1.yaml
python training/evaluate_commitment_control.py --config configs/train_cipc_belief_r_lora_v1.yaml
```

Important note:

- the current executable trainer is a NumPy multitask baseline
- the next intended upgrade is a HuggingFace LoRA backend

HF/LoRA smoke path:

```bash
python training/train_commitment_control_hf.py --config configs/train_cipc_belief_r_qwen05b_lora_smoke.json
```

This smoke config uses `Qwen/Qwen2.5-0.5B-Instruct` with LoRA on a reduced
subset to validate the real GPU-backed training path before scaling up.

## Second Dataset Path

The repo also includes a second transformed dataset built from the official
ATOMIC release:

```bash
python -m src.main --config configs/atomic_explicit_pilot.json
python -m src.main --config configs/atomic_explicit_openai_small.json
```

This dataset is intentionally more explicit than Belief-R and acts as a
sanity-check benchmark for directly stated revision cues.

## Repository Layout

- `data/`: raw downloads and processed JSONL datasets
- `transforms/`: notes on dataset transforms
- `prompts/`: versioned prompt templates reserved for later API runs
- `src/`: data loading, belief ledger, systems, metrics, and runner
- `training/`: commitment-control training and evaluation entrypoints
- `configs/`: reproducible experiment configs
- `runs/`: experiment outputs
- `analysis/`: analysis artifacts
- `paper_assets/`: tables and figures for the paper
- `docs/`: progress reports and related-work positioning notes
