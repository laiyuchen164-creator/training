# Runpod Training Handoff

This memo is the short operational brief for continuing the project on a new
GPU training server.

## Current State

The repository contains two historical lines, but only one is the active
method line now.

- Frozen diagnosis line:
  prompt-based `source_revision` and related Belief-R prompt ablations.
- Active method line:
  training-based `CIPC` on the Belief-R commitment-control split.

The active project direction is no longer prompt tuning. The active direction
is real-model training with the HF/LoRA backend.

## What Is Already Done

### Data

- Belief-R incremental dataset built and tracked:
  `data/processed/belief_r_incremental.jsonl`
- Belief-R commitment-control train/dev/test built and tracked:
  `data/processed/belief_r_commitment_control_{train,dev,test}.jsonl`
- Deterministic commitment-aligned original-format test subset built and
  tracked:
  `data/processed/belief_r_incremental_commitment_test_subset.jsonl`

### Training code

- NumPy multitask baseline:
  `training/train_commitment_control.py`
- HF/LoRA backend:
  `training/train_commitment_control_hf.py`
- HF model wrapper:
  `src/models/hf_commitment_control_model.py`
- Local frontier HF configs:
  - `configs/train_cipc_belief_r_qwen05b_lora_control_focused_v1.json`
  - `configs/train_cipc_belief_r_qwen05b_lora_highrank_v1.json`

### Main empirical results

- Strongest current local baseline:
  NumPy `CIPC` on the same Belief-R split
  - overall answer accuracy: `0.7846`
  - overturn answer accuracy: `0.7961`
  - report: `analysis/cipc_belief_r_lora_v1_report.md`
- Current real HF/LoRA baseline:
  `Qwen/Qwen2.5-0.5B-Instruct`
  - old balanced-full baseline:
    - overall answer accuracy: `0.6462`
    - overturn answer accuracy: `0.5728`
    - no-overturn answer accuracy: `0.9259`
  - best balanced HF follow-up:
    - overall answer accuracy: `0.7923`
    - overturn answer accuracy: `0.7670`
    - no-overturn answer accuracy: `0.8889`
    - report: `analysis/cipc_qwen05b_local_followup_report.md`
  - best raw HF follow-up:
    - overall answer accuracy: `0.8462`
    - overturn answer accuracy: `0.8641`
    - no-overturn answer accuracy: `0.7778`
    - report: `analysis/cipc_qwen05b_local_followup_report.md`
- Same-split frozen prompt baseline:
  - overall answer accuracy: `0.3615`
  - overturn answer accuracy: `0.2718`
  - report: `analysis/cipc_vs_prompt_source_revision_test_report.md`

## Current Interpretation

The project is not blocked on infrastructure anymore.

- The prompt line is frozen and only kept as a baseline.
- The HF/LoRA line is already better than the frozen prompt baseline.
- The HF line now has two local frontier points:
  one more balanced, one more aggressive.
- The main remaining problem is a trade-off problem:
  keep the stronger overturn gains while recovering maintain-side stability.

## Main Objective On Runpod

Use the new GPU server to improve the local HF trade-off frontier while keeping
the current Belief-R split and metrics fixed.

Concretely:

- retain or exceed the overturn strength of `highrank_v1`
- while recovering the no-overturn behavior lost relative to
  `control_focused_v1`

## What Not To Do

- Do not return to prompt tuning as the main line.
- Do not spend time on `ReviseQA` or ATOMIC before the HF Belief-R line is
  stronger.
- Do not change the current Belief-R split definitions or the seed unless the
  experiment explicitly studies that change.
- Do not treat old prompt results as the main paper result line.
- Do not push `runs/`, checkpoints, or cached model files to GitHub.

## Read First

If a new Codex session starts on Runpod, read these first in this order:

1. `README.md`
2. `docs/runpod_training_handoff.md`
3. `docs/progress_report.md`
4. `analysis/cipc_qwen05b_local_followup_report.md`
5. `analysis/cipc_vs_prompt_source_revision_test_report.md`

## First Commands On Runpod

```bash
git clone https://github.com/laiyuchen164-creator/training.git
cd training
bash scripts/runpod_preflight_check.sh
```

If preflight passes, continue with either:

```bash
bash scripts/runpod_train_belief_r_qwen05b_smoke.sh
```

or, if the machine is already trusted and the goal is direct scaling:

```bash
bash scripts/runpod_train_belief_r_qwen05b_control_focused.sh
bash scripts/runpod_train_belief_r_qwen05b_highrank.sh
```

## Success Criteria For The Next Round

Minimum success:

- beat `control_focused_v1` on overall answer accuracy
  while not dropping further on `incremental_no_overturn`
- or beat `highrank_v1` on no-overturn while keeping comparable overturn

Current local targets:

- balanced target:
  - overall answer accuracy `> 0.7923`
  - overturn answer accuracy `> 0.7670`
  - no-overturn answer accuracy `>= 0.8889`
- aggressive target:
  - overall answer accuracy `> 0.8462`
  - overturn answer accuracy `> 0.8641`

Stronger success:

- materially reduce the trade-off between overturn and no-overturn
- keep `early_commitment_persistence` low without collapsing maintain cases

## Highest-Value Next Steps

Priority order:

1. Reproduce the tracked HF baseline on the server or run a smoke validation if
   the image is new.
2. Reproduce the two local frontier runs:
   `control_focused_v1` and `highrank_v1`.
3. Combine the better training emphasis from `control_focused_v1` with the
   higher-capacity setting from `highrank_v1`.
4. Only after that, try a somewhat larger instruction model if memory allows.
5. Compare every new run against:
   - HF control-focused baseline
   - HF high-rank baseline
   - NumPy `CIPC` baseline
   - frozen prompt baseline

## Files That Define The Active Training Path

- `scripts/runpod_preflight_check.sh`
- `scripts/runpod_setup.sh`
- `scripts/runpod_train_belief_r_qwen05b_control_focused.sh`
- `scripts/runpod_train_belief_r_qwen05b_highrank.sh`
- `configs/train_cipc_belief_r_qwen05b_lora_control_focused_v1.json`
- `configs/train_cipc_belief_r_qwen05b_lora_highrank_v1.json`
- `training/train_commitment_control_hf.py`
- `src/models/hf_commitment_control_model.py`

## Short Summary

The correct direction on Runpod is:

- stay on Belief-R
- stay on training
- keep the current tracked split
- use HF/LoRA
- improve the HF trade-off frontier rather than reverting to prompt work

Do not let the project drift back into prompt tweaking or benchmark expansion
before the HF line is stronger.
