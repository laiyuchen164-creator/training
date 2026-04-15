# CIPC Belief-R v1 Report

## Scope

This stage follows the new training-first master plan.

- Old prompt-based `source_revision` stays frozen as a diagnosis baseline.
- The active method line is now `CIPC`:
  Commitment Integration and Propagation Control.
- The current executable proof-of-concept is a NumPy multitask baseline,
  not yet a HuggingFace LoRA run, because this environment does not include
  `torch`, `transformers`, or `peft`.

## What Was Built

Dataset conversion and reports:

- `data/build_belief_r_commitment_control.py`
- `data/processed/belief_r_commitment_control_train.jsonl`
- `data/processed/belief_r_commitment_control_dev.jsonl`
- `data/processed/belief_r_commitment_control_test.jsonl`
- `analysis/belief_r_commitment_control_stats.md`
- `analysis/belief_r_commitment_control_spotcheck.md`

Training and evaluation path:

- `training/train_commitment_control.py`
- `training/evaluate_commitment_control.py`
- `src/models/commitment_control_model.py`
- `src/eval/commitment_metrics.py`
- `configs/train_cipc_belief_r_lora_v1.yaml`
- `runs/cipc_belief_r_lora_v1/`

## Dataset Readout

The commitment-control conversion now builds deterministically from Belief-R.

- Total pairs: `1282`
- Total examples: `2564`
- Split sizes:
  - train: `2050`
  - dev: `254`
  - test: `260`
- Active control labels in v1:
  - `preserve`
  - `replace`
- `weaken` is kept in the label space but not used in this first Belief-R pass.

The important implementation correction in this stage was to split by
`pair_key = pair_id + modus`, not bare `pair_id`, because Belief-R reuses the
same numeric pair id across `ponens` and `tollens`.

## Training Setup

- Model family: hashed linear multitask classifier
- Input fields:
  - `early_context`
  - `early_commitment_text`
  - `late_evidence`
  - metadata fields such as condition, source type, modus, relation type
- Outputs:
  - `control_decision`
  - `final_answer`
- Objective:
  - `L = L_ctrl + L_ans`
- Important practical change:
  - deterministic control-label oversampling in train to stop the model from
    collapsing into all-`replace`

## Main Result

Test split summary:

- `control_decision_accuracy = 0.8923`
- `final_answer_accuracy = 0.7846`
- `joint_accuracy = 0.7808`
- `consistency_rate_gold = 0.7846`
- `early_commitment_persistence = 0.2524`
- `late_evidence_takeover = 0.7476`

Most important condition-level numbers on test:

- `incremental_no_overturn`
  - control accuracy: `0.9259`
  - final-answer accuracy: `0.9259`
- `incremental_overturn_reasoning`
  - control accuracy: `0.9806`
  - final-answer accuracy: `0.7961`
  - early commitment persistence: `0.2039`
  - late evidence takeover: `0.7961`
- `full_info`
  - control accuracy: `0.8154`
  - final-answer accuracy: `0.7462`

## Interpretation

This run satisfies the core proof-of-concept requirement for the new method
line:

- the project now has a real training pipeline
- the model produces parseable structured outputs end to end
- the model no longer shows the catastrophic `incremental_no_overturn`
  collapse seen in the first unbalanced training attempt
- overturn recovery remains strong instead of being traded away

The strongest current evidence is the joint condition behavior:

- no-overturn is high: `0.9259`
- overturn answer accuracy remains high enough to be useful: `0.7961`

That is the right directional behavior for the paper's new framing around
early commitment overweighting and propagation control.

## Remaining Weaknesses

The main remaining weakness is not total collapse; it is over-revision on a
smaller residual subset.

- Test `incremental_no_overturn` now has only `2` answer errors.
- Both are `tollens` examples where the model still predicts
  `replace -> c` instead of preserving the early commitment:
  - `belief_r::297-strong::tollens::incremental`
  - `belief_r::639-strong::tollens::incremental`

The main open issue on overturn is that control prediction is stronger than
final-answer propagation:

- test overturn control accuracy: `0.9806`
- test overturn answer accuracy: `0.7961`

So the new pipeline already separates the two failure modes the plan asked us
to disentangle:

- integration is now quite strong
- propagation still lags behind integration

## What Is Still Missing

Two important items are still pending before this becomes the paper's main
result line.

- Same-split comparison against the frozen prompt-based `source_revision`
  baseline has not been run yet.
- The backend is not yet a true LoRA / QLoRA run on a pretrained instruction
  model.

So this stage should be described as:

- Phase A complete
- Phase B proof-of-concept complete
- Phase C partially complete

## Verdict

The repository has successfully switched from prompt-only repair to a trainable
Belief-R commitment-control pipeline.

The current `CIPC` proof-of-concept is good enough to continue on the training
path rather than returning to prompt tuning.
