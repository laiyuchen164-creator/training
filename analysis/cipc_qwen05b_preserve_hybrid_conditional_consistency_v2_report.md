# Qwen 0.5B Preserve Hybrid Conditional Consistency V2 Report

## Scope

This run was the minimal controlled follow-up to
`preserve_hybrid_conditional_consistency_v1`.

Kept fixed:

- model family
- split
- seed
- LoRA rank/config
- epoch count
- sampler
- `answer_loss_weight`
- split preserve/replace aggregation
- hybrid preserve-side form
- replace-side CE plus anti-early margin
- `lambda_pres = 0.20`
- `lambda_rep = 0.08`
- `preserve_margin_m = 0.3`
- `beta_replace_margin = 0.05`
- `margin_m = 0.5`

Changed:

- increased only:
  - `beta_preserve_margin: 0.1 -> 0.2`

## Exact Config

- config:
  [configs/train_cipc_belief_r_qwen05b_lora_preserve_hybrid_conditional_consistency_v2.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_preserve_hybrid_conditional_consistency_v2.json)
- run:
  `runs/cipc_belief_r_qwen05b_lora_preserve_hybrid_conditional_consistency_v2`
- command:
  `.venv/bin/python training/train_commitment_control_hf.py --config configs/train_cipc_belief_r_qwen05b_lora_preserve_hybrid_conditional_consistency_v2.json`

## Result

Test-set answer metrics:

- overall answer: `0.8615`
- overturn answer: `0.9515`
- no-overturn answer: `0.5185`

## Comparison

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| preserve_hybrid_conditional_consistency_v2 | 0.8615 | 0.9515 | 0.5185 |
| preserve_hybrid_conditional_consistency_v1 | 0.8615 | 0.9223 | 0.6296 |
| highrank_v1 | 0.8538 | 0.9029 | 0.6667 |
| split_conditional_consistency_v1 | 0.8462 | 0.9223 | 0.5556 |
| conditional_consistency_v1 | 0.8923 | 0.9806 | 0.5556 |
| conditional_consistency_v2 | 0.8154 | 0.8738 | 0.5926 |
| tradeoff_repair_v2 | 0.8000 | 0.7961 | 0.8148 |
| NumPy baseline | 0.7846 | 0.7961 | 0.9259 |

## Verdict

### Did increasing `beta_preserve_margin` improve no-overturn?

No. It sharply worsened it.

- preserve-hybrid `v1` no-overturn: `0.6296`
- preserve-hybrid `v2` no-overturn: `0.5185`
- delta: `-0.1111`

This is well below `highrank_v1` and even below `split_conditional_consistency_v1`.

### Did overturn improve?

Yes, strongly.

- preserve-hybrid `v1` overturn: `0.9223`
- preserve-hybrid `v2` overturn: `0.9515`
- delta: `+0.0292`

This is now one of the strongest aggressive overturn points in the entire
consistency branch.

### Did overall answer improve?

No meaningful change in aggregate overall:

- preserve-hybrid `v1` overall: `0.8615`
- preserve-hybrid `v2` overall: `0.8615`

So the extra preserve-margin weight mostly reshaped the frontier internally:

- more overturn
- much less no-overturn
- same overall aggregate

## Interpretation

This result is important because it falsifies a tempting next-step hypothesis.

The hypothesis was:

- hybrid preserve is directionally right
- the margin term inside it is just too weak
- increasing `beta_preserve_margin` should recover no-overturn

What actually happened:

- increasing `beta_preserve_margin` pushed the system harder toward aggressive
  overturn behavior instead of preserving maintain cases

So in the current hybrid formulation, the preserve-margin term is not acting as
the kind of maintain-side correction we wanted.

The most plausible reading is:

- once CE-to-early is already present, increasing the current margin term does
  not simply "help preserve"
- instead, the combined objective still couples into a more globally
  aggressive answer geometry

That means the issue is no longer just scalar strength.

The hybrid preserve form itself is likely not aligned enough with the desired
maintain-side behavior.

## Training Note

The dev trajectory remained smooth:

- epoch 1 dev answer: `0.2126`
- epoch 2 dev answer: `0.4094`
- epoch 3 dev answer: `0.6457`
- epoch 4 dev answer: `0.7323`
- epoch 5 dev answer: `0.6772`
- epoch 6 dev answer: `0.8189`

So this is not a training-instability failure.

The failure is objective-shape failure:

- the hybrid preserve design still routes added pressure into overturn-heavy
  behavior

## Bottom Line

`preserve_hybrid_conditional_consistency_v2` is not the repair.

- It beat `highrank_v1` on overall answer.
- It strongly beat `highrank_v1` on overturn.
- It badly failed on no-overturn.

The key new conclusion is:

- in the current hybrid preserve formulation, strengthening the margin term is
  not enough and may even push the objective further away from maintain-side
  repair
