# Qwen 0.5B Preserve Margin Conditional Consistency V2 Report

## Scope

This run was the minimal controlled follow-up to
`preserve_margin_conditional_consistency_v1`.

Kept fixed:

- model family
- split
- seed
- LoRA rank/config
- epoch count
- sampler
- `answer_loss_weight`
- split preserve/replace aggregation
- preserve-side margin form
- replace-side CE plus anti-early margin
- `lambda_pres = 0.20`
- `lambda_rep = 0.08`
- `beta_replace_margin = 0.05`
- `margin_m = 0.5`

Changed:

- weakened only:
  - `preserve_margin_m: 0.5 -> 0.3`

## Exact Config

- config:
  [configs/train_cipc_belief_r_qwen05b_lora_preserve_margin_conditional_consistency_v2.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_preserve_margin_conditional_consistency_v2.json)
- run:
  `runs/cipc_belief_r_qwen05b_lora_preserve_margin_conditional_consistency_v2`
- command:
  `.venv/bin/python training/train_commitment_control_hf.py --config configs/train_cipc_belief_r_qwen05b_lora_preserve_margin_conditional_consistency_v2.json`

## Result

Test-set answer metrics:

- overall answer: `0.8154`
- overturn answer: `0.8544`
- no-overturn answer: `0.6667`

## Comparison

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| preserve_margin_conditional_consistency_v2 | 0.8154 | 0.8544 | 0.6667 |
| preserve_margin_conditional_consistency_v1 | 0.8077 | 0.8350 | 0.7037 |
| split_conditional_consistency_v1 | 0.8462 | 0.9223 | 0.5556 |
| highrank_v1 | 0.8538 | 0.9029 | 0.6667 |
| conditional_consistency_v2 | 0.8154 | 0.8738 | 0.5926 |
| tradeoff_repair_v2 | 0.8000 | 0.7961 | 0.8148 |
| NumPy baseline | 0.7846 | 0.7961 | 0.9259 |
| frozen prompt baseline | 0.3615 | 0.2718 | 0.5926 |

## Verdict

### Did overturn recover relative to preserve-margin `v1`?

Yes.

- preserve-margin `v1` overturn: `0.8350`
- preserve-margin `v2` overturn: `0.8544`
- delta: `+0.0194`

### Did no-overturn stay above `highrank_v1`?

No.

- `highrank_v1` no-overturn: `0.6667`
- preserve-margin `v2` no-overturn: `0.6667`

This exactly matched `highrank_v1`, but did not exceed it.

### Did this become a better balanced point than preserve-margin `v1`?

Only partially.

What improved:

- overall answer rose
  - `0.8077 -> 0.8154`
- overturn rose
  - `0.8350 -> 0.8544`

What weakened:

- no-overturn fell
  - `0.7037 -> 0.6667`

So reducing the preserve margin moved the frontier back toward the aggressive
side, but it gave back the extra maintain-side gain that made `v1`
interesting.

### Did this beat `highrank_v1`?

No.

- overall remained below `highrank_v1`
  - `0.8154 < 0.8538`
- overturn remained below `highrank_v1`
  - `0.8544 < 0.9029`
- no-overturn only matched `highrank_v1`
  - `0.6667 = 0.6667`

So this run still does not dominate the active frontier.

## Interpretation

This follow-up confirms the preserve margin is a smooth control lever rather
than a one-off lucky effect.

Observed movement from `preserve_margin_m = 0.5` to `0.3`:

- overturn increased
- no-overturn decreased
- overall increased slightly

That is the exact directional trade we needed to test.

So the preserve-side margin now looks like a real, controllable axis:

- stronger margin protects no-overturn more
- weaker margin gives overturn back

But at `0.3`, the margin is already weak enough that the no-overturn gain over
`highrank_v1` disappears.

## Training Note

The dev trajectory stayed usable and ended slightly stronger than `v1`:

- epoch 1 dev answer: `0.2126`
- epoch 2 dev answer: `0.4331`
- epoch 3 dev answer: `0.6614`
- epoch 4 dev answer: `0.6772`
- epoch 5 dev answer: `0.6772`
- epoch 6 dev answer: `0.7638`

This run does not suggest training collapse.

The main issue remains frontier quality, not optimization failure.

## Bottom Line

`preserve_margin_conditional_consistency_v2` confirms the new diagnosis.

- preserve-side margin is a real trade-off control lever
- weakening it from `0.5` to `0.3` predictably recovered overturn
- but that same weakening removed the no-overturn gain over `highrank_v1`

So the preserve-margin direction is valid, but the current simple form still
does not yield a frontier point that beats `highrank_v1`.
