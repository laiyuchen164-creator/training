# Qwen 0.5B Split Conditional Consistency V1 Report

## Scope

This run stayed strictly on the active EMNLP Belief-R HF/LoRA line and changed
only the conditional propagation loss shape relative to
`conditional_consistency_v2`.

Kept fixed:

- model family
- split
- seed
- LoRA rank/config
- epoch count
- sampler
- `answer_loss_weight`

Changed:

- replaced single pooled `lambda_prop * L_prop` with split propagation loss
- added separate preserve-side and replace-side aggregation
- added separate coefficients:
  - `lambda_pres = 0.20`
  - `lambda_rep = 0.08`
- kept:
  - `beta_replace_margin = 0.05`
  - `margin_m = 0.5`

## Implementation

Updated code path:

- [src/models/hf_commitment_control_model.py](/workspace/training/src/models/hf_commitment_control_model.py:28)
- [training/train_commitment_control_hf.py](/workspace/training/training/train_commitment_control_hf.py:351)

New propagation shape:

`L = L_ctrl + lambda_ans * L_ans + L_prop`

with:

`L_prop = lambda_pres * L_preserve + lambda_rep * L_replace`

where:

- `L_preserve` = mean CE to early implied answer over preserve examples only
- `L_replace` = mean CE to gold final answer plus anti-early margin over
  replace examples only

Important aggregation change:

- preserve and replace examples are no longer averaged together in one pool
- each side is averaged only within its own mask
- then the two masked means are combined with separate coefficients

## Exact Config

- config:
  [configs/train_cipc_belief_r_qwen05b_lora_split_conditional_consistency_v1.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_split_conditional_consistency_v1.json)
- run:
  `runs/cipc_belief_r_qwen05b_lora_split_conditional_consistency_v1`
- command:
  `python training/train_commitment_control_hf.py --config configs/train_cipc_belief_r_qwen05b_lora_split_conditional_consistency_v1.json`

## Result

Test-set answer metrics:

- overall answer: `0.8462`
- overturn answer: `0.9223`
- no-overturn answer: `0.5556`

## Comparison

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| split_conditional_consistency_v1 | 0.8462 | 0.9223 | 0.5556 |
| highrank_v1 | 0.8538 | 0.9029 | 0.6667 |
| conditional_consistency_v2 | 0.8154 | 0.8738 | 0.5926 |
| tradeoff_repair_v2 | 0.8000 | 0.7961 | 0.8148 |
| NumPy baseline | 0.7846 | 0.7961 | 0.9259 |
| frozen prompt baseline | 0.3615 | 0.2718 | 0.5926 |

## Verdict

### Did no-overturn improve relative to `highrank_v1`?

No.

- `highrank_v1` no-overturn: `0.6667`
- `split_conditional_consistency_v1` no-overturn: `0.5556`
- delta: `-0.1111`

This run failed the main success criterion.

### Did overturn remain competitive?

Yes.

- `split_conditional_consistency_v1` overturn: `0.9223`
- `highrank_v1` overturn: `0.9029`
- `conditional_consistency_v2` overturn: `0.8738`
- `tradeoff_repair_v2` overturn: `0.7961`
- `NumPy baseline` overturn: `0.7961`

So the split design preserved a strong aggressive overturn point.

### Did split preserve/replace aggregation improve stability?

Not in the sense required by the project goal.

Why:

- overall answer recovered strongly relative to `conditional_consistency_v2`
  - `0.8154 -> 0.8462`
- overturn also recovered strongly
  - `0.8738 -> 0.9223`
- but no-overturn fell back to the weaker `v1` level
  - `0.5926 -> 0.5556`

So the redesign did not produce a more balanced frontier point.

At best, it made the objective easier to push back toward aggressive replace
behavior while still not protecting preserve behavior enough.

## Interpretation

This result sharpens the current diagnosis.

What the split design achieved:

- it prevented replace examples from numerically dominating the propagation
  pool through shared averaging
- it kept a strong preserve-side coefficient in the objective

What still appears to be happening:

- the replace-side signal remains the more operationally effective gradient
  for shifting final-answer behavior
- the preserve-side cross-entropy alone is not strong enough to anchor
  no-overturn cases
- separate aggregation by itself does not solve the structural bias in the
  propagation objective

Empirically, the system moved back toward a high-overturn point instead of
forming a new balanced frontier.

## Training Note

The dev trajectory was smooth rather than wildly oscillatory:

- epoch 1 dev answer: `0.2126`
- epoch 2 dev answer: `0.5276`
- epoch 3 dev answer: `0.6693`
- epoch 4 dev answer: `0.7165`
- epoch 5 dev answer: `0.6929`
- epoch 6 dev answer: `0.7638`

So this run does not look unstable in the narrow optimization sense.

But it is still unstable in the frontier-design sense that matters here:

- small objective-shape changes are still not producing a controlled
  overturn/no-overturn balance

## Bottom Line

`split_conditional_consistency_v1` is not the repair.

- It improved overall answer relative to `conditional_consistency_v2`.
- It improved overturn relative to both `highrank_v1` and
  `conditional_consistency_v2`.
- It did not improve no-overturn relative to `highrank_v1`.
- It did not yield a better balanced frontier point.

The next redesign should not stop at separate aggregation alone.

The preserve side likely needs a stronger structural constraint than plain CE
to the early label, or a more explicit preserve-vs-alternative margin, if the
goal is to make no-overturn behavior survive while keeping overturn strong.
