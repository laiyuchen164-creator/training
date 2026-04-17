# Qwen 0.5B Preserve Hybrid Conditional Consistency V1 Report

## Scope

This run was the first controlled hybrid preserve-side follow-up after the
pure preserve-margin experiments.

Kept fixed:

- model family
- split
- seed
- LoRA rank/config
- epoch count
- sampler
- `answer_loss_weight`
- split preserve/replace aggregation
- replace-side CE plus anti-early margin
- `lambda_pres = 0.20`
- `lambda_rep = 0.08`
- `beta_replace_margin = 0.05`
- `margin_m = 0.5`

Base preserved from `preserve_margin_conditional_consistency_v2`:

- `preserve_margin_m = 0.3`

Changed:

- preserve-side loss is no longer pure margin
- new preserve-side shape:
  - `CE_to_early`
  - plus a small preserve margin term
- new setting:
  - `beta_preserve_margin = 0.1`

## Implementation

Updated code path:

- [src/models/hf_commitment_control_model.py](/workspace/training/src/models/hf_commitment_control_model.py:28)
- [training/train_commitment_control_hf.py](/workspace/training/training/train_commitment_control_hf.py:351)

New preserve-side shape:

`L_preserve = mean CE(final_answer_dist, early_implied_answer) + beta_preserve_margin * mean max(0, preserve_margin_m - (log p(early) - max_non_early_log_prob))`

Overall propagation shape remained:

`L_prop = lambda_pres * L_preserve + lambda_rep * L_replace`

## Exact Config

- config:
  [configs/train_cipc_belief_r_qwen05b_lora_preserve_hybrid_conditional_consistency_v1.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_preserve_hybrid_conditional_consistency_v1.json)
- run:
  `runs/cipc_belief_r_qwen05b_lora_preserve_hybrid_conditional_consistency_v1`
- command:
  `.venv/bin/python training/train_commitment_control_hf.py --config configs/train_cipc_belief_r_qwen05b_lora_preserve_hybrid_conditional_consistency_v1.json`

## Result

Test-set answer metrics:

- overall answer: `0.8615`
- overturn answer: `0.9223`
- no-overturn answer: `0.6296`

## Comparison

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| preserve_hybrid_conditional_consistency_v1 | 0.8615 | 0.9223 | 0.6296 |
| highrank_v1 | 0.8538 | 0.9029 | 0.6667 |
| preserve_margin_conditional_consistency_v1 | 0.8077 | 0.8350 | 0.7037 |
| preserve_margin_conditional_consistency_v2 | 0.8154 | 0.8544 | 0.6667 |
| split_conditional_consistency_v1 | 0.8462 | 0.9223 | 0.5556 |
| conditional_consistency_v2 | 0.8154 | 0.8738 | 0.5926 |
| tradeoff_repair_v2 | 0.8000 | 0.7961 | 0.8148 |
| NumPy baseline | 0.7846 | 0.7961 | 0.9259 |
| frozen prompt baseline | 0.3615 | 0.2718 | 0.5926 |

## Verdict

### Did overall answer improve relative to `highrank_v1`?

Yes.

- `highrank_v1` overall: `0.8538`
- `preserve_hybrid_v1` overall: `0.8615`
- delta: `+0.0077`

### Did overturn remain competitive?

Yes, strongly.

- `preserve_hybrid_v1` overturn: `0.9223`
- `highrank_v1` overturn: `0.9029`
- delta: `+0.0194`

This matches the best aggressive overturn level seen in the split consistency
line.

### Did no-overturn improve relative to `highrank_v1`?

No.

- `highrank_v1` no-overturn: `0.6667`
- `preserve_hybrid_v1` no-overturn: `0.6296`
- delta: `-0.0371`

So the hybrid preserve design did not preserve the maintain-side gain from the
pure preserve-margin runs.

### Did the hybrid preserve loss improve frontier quality?

Partially.

What improved:

- overall answer became the strongest seen so far on this consistency branch
- overturn also became stronger than `highrank_v1`

What did not improve:

- no-overturn remained below `highrank_v1`

So this run created a stronger aggressive frontier point, not a repaired
balanced frontier point.

## Interpretation

This result clarifies what the hybrid preserve design is doing.

Adding CE back into preserve-side supervision:

- stabilizes the training trajectory
- restores aggressive answer performance
- but also reduces the strength of the preserve-side correction

Compared with the pure preserve-margin runs:

- the hybrid design moves sharply back toward overturn
- it loses the extra maintain-side protection that pure margin created

That means the small preserve margin is currently being dominated by the direct
CE pressure plus the rest of the training objective.

## Training Note

The dev trajectory was smooth and strong:

- epoch 1 dev answer: `0.2126`
- epoch 2 dev answer: `0.4646`
- epoch 3 dev answer: `0.6614`
- epoch 4 dev answer: `0.6614`
- epoch 5 dev answer: `0.6929`
- epoch 6 dev answer: `0.7795`

So this run does not look noisy or unstable in optimization terms.

The issue is still objective balance, not training failure.

## Bottom Line

`preserve_hybrid_conditional_consistency_v1` is not the final repair, but it
is informative.

- It beat `highrank_v1` on overall answer.
- It beat `highrank_v1` on overturn.
- It failed to beat `highrank_v1` on no-overturn.

So the hybrid preserve loss is better understood as an aggressive-performance
recovery design than a maintain-side repair design.

The key lesson is:

- pure preserve margin can lift no-overturn
- hybrid preserve loss can recover aggressive performance
- but the current small hybrid margin is still too weak to hold the
  maintain-side gain
