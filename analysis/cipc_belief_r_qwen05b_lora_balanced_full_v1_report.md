# CIPC Qwen 0.5B LoRA Balanced Full v1

## Scope

This run continues the local GPU-backed HF/LoRA line after the initial smoke
test. The goal is not to beat the stronger NumPy multitask baseline yet; the
goal is to move the real LoRA backend past the most obvious failure mode:
`incremental_no_overturn` collapse.

Run:

- `runs/cipc_belief_r_qwen05b_lora_balanced_full_v1/`

Config:

- `configs/train_cipc_belief_r_qwen05b_lora_balanced_full_v1.json`

Model:

- `Qwen/Qwen2.5-0.5B-Instruct`
- LoRA on attention projections
- local GPU backend on `RTX 4080 SUPER`

## What Changed Relative To The Smoke Run

- increased from a smoke subset to the full train split
- increased training length to `5` epochs
- kept control-label oversampling enabled
- kept answer supervision and class-weighted control loss

This is the first local HF run meant to be a real pilot rather than a pure
backend smoke check.

## Main Result

Test split:

- `control_decision_accuracy = 0.6769`
- `final_answer_accuracy = 0.6462`
- `joint_accuracy = 0.6462`
- `early_commitment_persistence = 0.4272`
- `late_evidence_takeover = 0.5728`

Condition-level test results:

- `full_info`
  - control `0.6769`
  - answer `0.6462`
- `incremental_no_overturn`
  - control `0.9259`
  - answer `0.9259`
- `incremental_overturn_reasoning`
  - control `0.6117`
  - answer `0.5728`
  - early persistence `0.4272`
  - late takeover `0.5728`

## Comparison

### Versus the earlier HF smoke run

The balanced full run is materially better than the one-epoch smoke run.

- smoke test overall answer accuracy:
  `0.5469`
- balanced full overall answer accuracy:
  `0.6462`
- smoke `incremental_no_overturn` answer accuracy:
  `0.2917`
- balanced full `incremental_no_overturn` answer accuracy:
  `0.9259`

So the real gain from this round is that the HF path no longer fails by simply
forgetting how to preserve.

### Versus the frozen prompt baseline

Same-split comparison:

- report:
  `analysis/cipc_qwen05b_balanced_full_vs_prompt_test_report.md`

Overall test answer accuracy:

- HF balanced full `CIPC`: `0.6462`
- frozen prompt `source_revision`: `0.3615`

On `incremental_overturn_reasoning`:

- HF balanced full `CIPC`: `0.5728`
- frozen prompt `source_revision`: `0.2718`

So the local HF LoRA line is already clearly stronger than the frozen
prompt-based baseline.

### Versus the stronger NumPy multitask baseline

Reference run:

- `runs/cipc_belief_r_lora_v1/`

Important test comparison:

- NumPy baseline overall answer accuracy:
  `0.7846`
- HF balanced full overall answer accuracy:
  `0.6462`
- NumPy baseline overturn answer accuracy:
  `0.7961`
- HF balanced full overturn answer accuracy:
  `0.5728`

So the HF line is now directionally right but still not the best local method
we have.

## Interpretation

This run establishes three important points.

- The HF/LoRA backend is no longer only a smoke path; it can train on the real
  Belief-R commitment-control split and produce sensible condition-level
  behavior.
- The main no-overturn failure is largely fixed.
- The remaining weakness is concentrated in overturn propagation, where the
  model still leaves too much weight on the earlier commitment.

In other words, the real backend has now caught up to the paper's intended
failure analysis:

- preserve is mostly handled
- replace is still under-executed

## Current Verdict

`balanced_full_v1` is a useful HF proof-of-concept and is worth keeping as the
current real-model training baseline.

It is not yet the main paper result line because:

- it still trails the stronger NumPy multitask baseline
- `late_evidence_takeover` on overturn is still too low
- `early_commitment_persistence` on overturn is still too high

## Best Next Step

The next high-value local step is not more prompt tuning.

It is one of:

- increase HF training strength while keeping the balanced objective
- move from `0.5B` to a somewhat larger instruction model if memory allows
- or add a consistency-aware training term after the current multitask setup
  stabilizes further
