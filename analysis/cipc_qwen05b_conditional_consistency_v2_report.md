# Qwen 0.5B Conditional Consistency V2 Report

## Scope

This follow-up stayed on the same active Belief-R HF/LoRA line and changed only
the propagation-loss weights relative to `conditional_consistency_v1`.

Kept fixed:

- model family
- split
- seed
- LoRA rank/config
- epoch count
- sampler
- `answer_loss_weight`

Changed:

- `lambda_prop`: `0.3 -> 0.1`
- `beta_replace_margin`: `0.2 -> 0.05`
- `margin_m`: kept at `0.5`

## Exact Config

- config:
  [configs/train_cipc_belief_r_qwen05b_lora_conditional_consistency_v2.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_conditional_consistency_v2.json)
- run:
  `runs/cipc_belief_r_qwen05b_lora_conditional_consistency_v2`

## Result

Test-set answer metrics:

- overall answer: `0.8154`
- overturn answer: `0.8738`
- no-overturn answer: `0.5926`

## Comparison

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| conditional_consistency_v2 | 0.8154 | 0.8738 | 0.5926 |
| conditional_consistency_v1 | 0.8923 | 0.9806 | 0.5556 |
| highrank_v1 | 0.8538 | 0.9029 | 0.6667 |
| tradeoff_repair_v2 | 0.8000 | 0.7961 | 0.8148 |
| highrank_consistency_v1 | 0.7846 | 0.7670 | 0.8519 |
| highrank_consistency_v2 | 0.7923 | 0.9126 | 0.3333 |

## Verdict

This run supports the diagnosis that the `v1` conditional propagation loss was
too replace-heavy.

Evidence:

- no-overturn improved relative to `conditional_consistency_v1`
  - `0.5556 -> 0.5926`
- overturn dropped from the extremely aggressive `v1` point
  - `0.9806 -> 0.8738`

But this is still not good enough for the active project goal.

Why:

- no-overturn is still below `highrank_v1`
  - `0.5926 < 0.6667`
- overturn is now also below `highrank_v1`
  - `0.8738 < 0.9029`

So `conditional_consistency_v2` is a partial directional repair, not a new
frontier point.

## Main Takeaway

Reducing `lambda_prop` and the replace margin moved the model in the expected
direction, which means the current failure is not random.

But simple weight reduction alone is insufficient.

The next fix should target the shape of `L_prop` itself:

- strengthen preserve-side influence explicitly
- or change preserve/replace aggregation so replace examples do not dominate the
  propagation signal
