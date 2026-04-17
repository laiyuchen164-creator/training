# Qwen 0.5B Preserve Margin Conditional Consistency V1 Report

## Scope

This run stayed on the same active EMNLP Belief-R HF/LoRA line and changed
only the preserve-side propagation form relative to
`split_conditional_consistency_v1`.

Kept fixed:

- model family
- split
- seed
- LoRA rank/config
- epoch count
- sampler
- `answer_loss_weight`
- split preserve/replace aggregation
- replace-side alignment plus anti-early margin
- `lambda_pres = 0.20`
- `lambda_rep = 0.08`
- `beta_replace_margin = 0.05`
- `margin_m = 0.5`

Changed:

- replaced preserve-side CE-to-early supervision with a preserve-side margin
  loss
- new preserve constraint:
  early implied answer must outrank the strongest alternative answer by
  `preserve_margin_m = 0.5`

## Implementation

Updated code path:

- [src/models/hf_commitment_control_model.py](/workspace/training/src/models/hf_commitment_control_model.py:28)
- [training/train_commitment_control_hf.py](/workspace/training/training/train_commitment_control_hf.py:351)

Preserve-side propagation shape:

- for preserve examples:
  `L_preserve = mean max(0, preserve_margin_m - (log p(early) - max_non_early_log_prob))`

Replace-side propagation stayed:

- CE to the gold final answer
- plus anti-early margin

Overall propagation shape stayed:

`L_prop = lambda_pres * L_preserve + lambda_rep * L_replace`

## Exact Config

- config:
  [configs/train_cipc_belief_r_qwen05b_lora_preserve_margin_conditional_consistency_v1.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_preserve_margin_conditional_consistency_v1.json)
- run:
  `runs/cipc_belief_r_qwen05b_lora_preserve_margin_conditional_consistency_v1`
- command:
  `.venv/bin/python training/train_commitment_control_hf.py --config configs/train_cipc_belief_r_qwen05b_lora_preserve_margin_conditional_consistency_v1.json`

## Result

Test-set answer metrics:

- overall answer: `0.8077`
- overturn answer: `0.8350`
- no-overturn answer: `0.7037`

## Comparison

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| preserve_margin_conditional_consistency_v1 | 0.8077 | 0.8350 | 0.7037 |
| split_conditional_consistency_v1 | 0.8462 | 0.9223 | 0.5556 |
| highrank_v1 | 0.8538 | 0.9029 | 0.6667 |
| conditional_consistency_v2 | 0.8154 | 0.8738 | 0.5926 |
| tradeoff_repair_v2 | 0.8000 | 0.7961 | 0.8148 |
| NumPy baseline | 0.7846 | 0.7961 | 0.9259 |
| frozen prompt baseline | 0.3615 | 0.2718 | 0.5926 |

## Verdict

### Did no-overturn improve relative to `highrank_v1`?

Yes.

- `highrank_v1` no-overturn: `0.6667`
- `preserve_margin_conditional_consistency_v1` no-overturn: `0.7037`
- delta: `+0.0370`

This is the first sign that the preserve-side structural change moved behavior
in the intended direction relative to the current Runpod frontier anchor.

### Did overturn remain competitive?

Partially, but not fully.

- `preserve_margin_conditional_consistency_v1` overturn: `0.8350`
- `highrank_v1` overturn: `0.9029`
- `conditional_consistency_v2` overturn: `0.8738`
- `tradeoff_repair_v2` overturn: `0.7961`
- `NumPy baseline` overturn: `0.7961`

Interpretation:

- overturn stayed clearly above the conservative anchors
- but it gave back a meaningful amount relative to `highrank_v1`

So this run is not a new best balanced frontier point yet.

### Did the preserve-side margin improve stability?

Yes, in the frontier-shaping sense.

Compared with the previous split-aggregation run:

- no-overturn improved strongly
  - `0.5556 -> 0.7037`
- overturn dropped materially
  - `0.9223 -> 0.8350`

This is the expected directional trade:

- stronger preserve-side structure now really anchors maintain cases
- but the current setting pulls too much mass away from overturn

So the margin redesign is directionally more stable than preserve-side CE.

## Interpretation

This run supports the diagnosis that preserve-side CE was too weak as a
structural constraint.

What the preserve margin changed:

- it stopped treating preserve merely as "raise the early label probability"
- it explicitly required the early label to beat the strongest alternative
- that stronger constraint translated into a real no-overturn recovery

What it did not solve:

- the current preserve margin is still too costly on overturn performance at
  this setting
- the new point sits between `highrank_v1` and `tradeoff_repair_v2`, rather
  than dominating the current frontier

So the core direction now looks better founded:

- preserve-side structure matters
- margin-style preserve supervision is more effective than preserve-side CE

## Training Note

The dev trajectory was not collapse-like, but less monotonic than the previous
split run:

- epoch 1 dev answer: `0.2126`
- epoch 2 dev answer: `0.3937`
- epoch 3 dev answer: `0.6535`
- epoch 4 dev answer: `0.7087`
- epoch 5 dev answer: `0.6614`
- epoch 6 dev answer: `0.7717`

So this run still trained cleanly enough to evaluate.

The more important signal is the frontier shift:

- preserve-side margin finally moved no-overturn upward relative to
  `highrank_v1`
- but it currently over-corrects relative to overturn

## Bottom Line

`preserve_margin_conditional_consistency_v1` is the strongest consistency-side
directional result so far, but not the final repair.

- It improved no-overturn relative to `highrank_v1`.
- It also improved no-overturn strongly relative to both
  `conditional_consistency_v2` and `split_conditional_consistency_v1`.
- It gave back overturn relative to `highrank_v1`.
- It therefore does not yet dominate the active frontier.

Still, this run is important because it changes the diagnosis from:

- "separate aggregation alone is insufficient"

to:

- "preserve-side structural constraints are real control levers, but the
  current preserve margin is too strong at this setting"
