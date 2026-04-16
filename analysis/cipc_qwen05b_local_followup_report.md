# CIPC Qwen 0.5B Local Follow-Up Report

## Scope

This report covers two targeted local follow-up runs after
`cipc_belief_r_qwen05b_lora_balanced_full_v1`.

The objective was narrow:

- keep the same Belief-R split and evaluation stack
- keep the same base model family
- change only training emphasis or LoRA capacity
- test whether local HF/LoRA can close the remaining overturn gap

Runs:

- `runs/cipc_belief_r_qwen05b_lora_control_focused_v1/`
- `runs/cipc_belief_r_qwen05b_lora_highrank_v1/`

## Config Changes

### Control-focused v1

- started from `balanced_full_v1`
- increased epochs from `5` to `8`
- reduced learning rate from `1.5e-4` to `1.2e-4`
- reduced `answer_loss_weight` from `1.0` to `0.7`
- kept LoRA rank at `8`

Intent:

- give more room to learn the control decision
- see whether better control improves overturn propagation

### High-rank v1

- started from `balanced_full_v1`
- increased LoRA rank from `8` to `16`
- increased LoRA alpha from `16` to `32`
- increased epochs from `5` to `6`
- reduced learning rate from `1.5e-4` to `1.2e-4`
- kept `answer_loss_weight = 1.0`

Intent:

- test whether remaining errors are partly capacity-limited

## Main Test Results

| run | control_acc | answer_acc | joint_acc | early_persistence | late_takeover |
|---|---:|---:|---:|---:|---:|
| HF balanced full v1 | 0.6769 | 0.6462 | 0.6462 | 0.4272 | 0.5728 |
| HF control-focused v1 | 0.7692 | 0.7923 | 0.7615 | 0.2330 | 0.7670 |
| HF high-rank v1 | 0.8615 | 0.8462 | 0.8231 | 0.1359 | 0.8641 |
| NumPy CIPC baseline | 0.8923 | 0.7846 | 0.7808 | 0.2524 | 0.7476 |
| frozen prompt baseline | 0.3615 | 0.3615 | 0.3615 | 0.6893 | 0.3107 |

## By Condition

### Control-focused v1

- `full_info`
  - control `0.7692`
  - answer `0.7923`
- `incremental_no_overturn`
  - control `0.9259`
  - answer `0.8889`
- `incremental_overturn_reasoning`
  - control `0.7282`
  - answer `0.7670`

### High-rank v1

- `full_info`
  - control `0.8615`
  - answer `0.8462`
- `incremental_no_overturn`
  - control `0.7778`
  - answer `0.7778`
- `incremental_overturn_reasoning`
  - control `0.8835`
  - answer `0.8641`

## Interpretation

The two follow-ups split the frontier in a useful way.

### Control-focused v1

This run is the first clear sign that the earlier HF gap was not only about
model size. Better training emphasis alone moved the local HF line from
`0.6462` to `0.7923` overall and from `0.5728` to `0.7670` on overturn.

This nearly closes the gap to the stronger NumPy baseline while keeping
`incremental_no_overturn` relatively healthy at `0.8889`.

### High-rank v1

This run is the strongest raw HF result so far.

- it beats the old HF balanced-full run by a large margin
- it beats the NumPy baseline on overall answer accuracy
- it beats the NumPy baseline on overturn answer accuracy

But it does so by reintroducing a maintain-side problem:

- `incremental_no_overturn` falls to `0.7778`

So `highrank_v1` is not yet a clean replacement for the most balanced local
candidate. It is the best raw model, but it pushes too hard toward replacement.

## Current Verdict

Local HF/LoRA is now clearly viable as the main real-model line.

- best balanced local HF candidate:
  `cipc_belief_r_qwen05b_lora_control_focused_v1`
- best raw local HF candidate:
  `cipc_belief_r_qwen05b_lora_highrank_v1`

The local research problem has changed.

It is no longer:

- can HF beat the frozen prompt baseline?

It is now:

- can we keep the `highrank_v1` overturn gains while recovering the
  no-overturn conservatism lost relative to `control_focused_v1`?

## Best Next Step

The highest-value next run on a stronger server is not another prompt baseline.

It is a trade-off repair run built on the new local frontier:

- start from the stronger-capacity line
- keep the better overturn behavior
- add back conservatism on `incremental_no_overturn`

In practice, that means the next server-side experiments should combine:

- the stronger training schedule from `control_focused_v1`
- the higher-capacity LoRA setting from `highrank_v1`
- and then test whether the maintain-side regression can be reduced without
  giving back the overturn gains
