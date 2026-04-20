# CIPC Qwen0.5B LoRA Gated Propagation v1

Date: 2026-04-20 UTC

## Purpose

This run tests a new CIPC propagation objective intended to reduce direct
answer-head pressure on the control head.

The hypothesis was:

- control head should decide preserve vs replace
- answer head should learn to propagate according to the detached control
  confidence
- the propagation loss should not backpropagate into the control head

## Baseline

Main comparison point:

- `highrank_v1`
  - overall answer: `0.8538`
  - overturn answer: `0.9029`
  - no-overturn answer: `0.6667`

## Configuration

Run:

- `runs/cipc_belief_r_qwen05b_lora_gated_propagation_v1/`

Config:

- `configs/train_cipc_belief_r_qwen05b_lora_gated_propagation_v1.json`

Base settings match `highrank_v1`:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- LoRA rank: `16`
- LoRA alpha: `32`
- max length: `256`
- epochs: `6`
- learning rate: `0.00012`
- answer loss weight: `1.0`

New gated propagation settings:

- `gated_lambda_pres = 0.2`
- `gated_lambda_rep = 0.2`
- `gated_beta_replace_margin = 0.1`
- `gated_margin_m = 0.3`

## Objective Change

The new objective adds:

`L_gated_prop = lambda_pres * stopgrad(P_ctrl(preserve)) * CE(answer, early_answer) + lambda_rep * stopgrad(P_ctrl(replace)) * CE(answer, gold_final_answer)`

plus a small replace-side anti-early margin weighted by the detached replace
gate.

The intended effect was to let the answer head follow the control head without
letting this propagation loss pull the control logits directly.

## Results

### Test Summary

- n: `260`
- control decision accuracy: `0.8231`
- final answer accuracy: `0.8077`
- joint accuracy: `0.7769`
- early commitment persistence: `0.1359`
- late evidence takeover: `0.8641`

### By Condition

| condition | n | control acc | answer acc | joint acc |
|---|---:|---:|---:|---:|
| full_info | 130 | 0.8231 | 0.8077 | 0.7769 |
| incremental_no_overturn | 27 | 0.5926 | 0.5926 | 0.5556 |
| incremental_overturn_reasoning | 103 | 0.8835 | 0.8641 | 0.8350 |

## Comparison to `highrank_v1`

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| `highrank_v1` | 0.8538 | 0.9029 | 0.6667 |
| `gated_propagation_v1` | 0.8077 | 0.8641 | 0.5926 |

## Interpretation

`gated_propagation_v1` is not a new frontier point.

It underperforms `highrank_v1` on all three main axes:

- overall answer accuracy decreased
- overturn answer accuracy decreased
- no-overturn answer accuracy also decreased

The negative result is informative. The detached control-confidence gate did
prevent the gated propagation loss from directly pulling the control head, but
the soft gate still appears to follow and amplify the model's existing
replace/aggressive tendencies instead of repairing preserve-side behavior.

In particular, the run did not solve the preserve-side problem:

- no-overturn answer accuracy fell from `0.6667` to `0.5926`
- this is worse than both `highrank_v1` and the pure preserve-margin repair
  direction

## Next Implication

The next preserve-side redesign should not rely only on detached predicted
control confidence as the gate.

More promising next variants:

1. **Gold-gated preserve propagation**
   - apply preserve propagation only on gold preserve examples
   - avoid using uncertain predicted control confidence as the preserve weight

2. **Floor-gated preserve propagation**
   - use `max(P_ctrl(preserve).detach(), floor)` on preserve examples
   - prevent preserve-side signal from vanishing when the model is already
     replace-biased

3. **Asymmetric preserve-only correction**
   - add preserve-side anchoring without adding more replace-side pressure
   - this may be better because replace behavior is already strong in the
     aggressive frontier

The immediate next candidate should be `gold_gated_preserve_v1`, not another
soft predicted-gate variant.
