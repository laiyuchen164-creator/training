# Qwen 0.5B Conditional Consistency V1 Report

## Scope

This run stayed on the active EMNLP Belief-R HF/LoRA line:

- Belief-R only
- training-based CIPC only
- HF/LoRA only
- no prompt-line return
- no ReviseQA
- no dataset expansion
- fixed split and seed

The goal was to replace the old head-matching consistency term with a
conditional propagation loss tied directly to gold control behavior.

## Pre-Change Implementation

Current consistency loss location before this change:

- [src/models/hf_commitment_control_model.py](/workspace/training/src/models/hf_commitment_control_model.py)
- wired from [training/train_commitment_control_hf.py](/workspace/training/training/train_commitment_control_hf.py)

Previous behavior:

- old consistency used a squared-error match between:
  - `P(control = preserve)`
  - `P(answer = early_implied_answer)`

Structural issue:

- it only supervised the preserve side directly
- it shaped replace behavior only indirectly
- that matched the instability seen in `highrank_consistency_v1` and `v2`

## Implemented Redesign

Main loss:

`L = L_ctrl + lambda_ans * L_ans + lambda_prop * L_prop`

Kept fixed:

- `L_ctrl`: control decision cross-entropy
- `L_ans`: final answer cross-entropy

Redesigned:

- `L_prop`: conditional propagation consistency

Implemented behavior:

- `preserve`: `-log p(early_implied_answer)`
- `replace`: `-log p(gold_final_answer)`
- `replace` anti-early margin:
  - `max(0, m - (log p(gold_final_answer) - log p(early_implied_answer)))`
- `weaken`: left neutral

Code changes:

- added `compute_conditional_propagation_loss(...)`
- added config knobs:
  - `lambda_prop`
  - `beta_replace_margin`
  - `margin_m`
- kept legacy `consistency_loss_weight` support for older runs

## Exact Experiment Config

Base:

- inherited from `highrank_v1`
- model: `Qwen/Qwen2.5-0.5B-Instruct`
- seed: `7`
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`
- epochs: `6`
- lr: `1.2e-4`
- train batch size: `4`
- eval batch size: `8`
- gradient accumulation: `4`
- `answer_loss_weight = 1.0`
- sampler unchanged
- split unchanged
- selection unchanged

New propagation settings:

- `lambda_prop = 0.3`
- `beta_replace_margin = 0.2`
- `margin_m = 0.5`

Config path:

- [configs/train_cipc_belief_r_qwen05b_lora_conditional_consistency_v1.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_conditional_consistency_v1.json)

Run path:

- `runs/cipc_belief_r_qwen05b_lora_conditional_consistency_v1`

## Result

Test-set answer metrics:

- overall answer: `0.8923`
- overturn answer: `0.9806`
- no-overturn answer: `0.5556`

## Required Comparison

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| conditional_consistency_v1 | 0.8923 | 0.9806 | 0.5556 |
| highrank_v1 | 0.8538 | 0.9029 | 0.6667 |
| tradeoff_repair_v2 | 0.8000 | 0.7961 | 0.8148 |
| highrank_consistency_v1 | 0.7846 | 0.7670 | 0.8519 |
| highrank_consistency_v2 | 0.7923 | 0.9126 | 0.3333 |
| NumPy baseline | 0.7846 | 0.7961 | 0.9259 |
| frozen prompt baseline | 0.3615 | 0.2718 | 0.5926 |

## Verdict

### Did no-overturn improve relative to `highrank_v1`?

No.

- `highrank_v1` no-overturn: `0.6667`
- `conditional_consistency_v1` no-overturn: `0.5556`
- delta: `-0.1111`

### Did overturn stay competitive?

Yes, and more than that.

- `conditional_consistency_v1` overturn: `0.9806`
- `highrank_v1` overturn: `0.9029`
- `tradeoff_repair_v2` overturn: `0.7961`
- `NumPy baseline` overturn: `0.7961`

This run strongly amplified overturn behavior.

### Is the redesigned consistency term more stable than the previous consistency versions?

Not in the sense that matters for the active repair goal.

What improved:

- it did not collapse to the extremely weak no-overturn point of
  `highrank_consistency_v2`
- it produced a coherent aggressive point with very strong overall and overturn
  accuracy

What did not improve:

- it still failed the main success criterion of recovering no-overturn relative
  to `highrank_v1`
- it pushed the model even further toward aggressive replacement
- it did not produce a balanced frontier point

So the redesign is not yet a stable repair mechanism for the current trade-off.

## Interpretation

The new propagation term appears to have made downstream propagation behavior
more directional, but the current first setting is still too replace-favoring.

Evidence:

- test overturn rose from `0.9029` to `0.9806`
- test no-overturn fell from `0.6667` to `0.5556`
- dev no-overturn was also weak at `0.3846`

This suggests the conditional objective is active, but the first
preserve/replace formulation plus weight setting does not yet create the
intended maintain/overturn balance.

## Bottom Line

`conditional_consistency_v1` is not the right frontier repair.

- It improved overall answer accuracy.
- It strongly improved overturn.
- It did not improve no-overturn.
- Therefore it does not satisfy the current main-line objective.
