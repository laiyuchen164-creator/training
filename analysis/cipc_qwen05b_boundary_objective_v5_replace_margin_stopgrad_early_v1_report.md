# Qwen 0.5B Boundary Objective V5 Replace-Margin Stopgrad-Early V1 Report

## Scope

This run continued the current main Belief-R HF/LoRA CIPC line and tested one
minimal propagation-internal structural follow-up after the earlier boundary
variants.

Kept fixed:

- dataset and split
- seed
- backbone
- LoRA rank/config
- shared encoder + control head + answer head structure
- answer CE path from the clean baseline
- preserve-side boundary objective
- replace-side CE term
- sampler
- epoch count

Changed:

- replace-side boundary margin only
- in the replace margin term, `s_early` was replaced by `stopgrad(s_early)`

This isolates whether the direct anti-early gradient in the replace margin is
the main propagation-internal source of aggressive overturn coupling.

## Theory Motivation

The current clean baseline remains `boundary_objective_v1`.

The recent results established:

- increasing `beta_pres` pushes the model further into an aggressive overturn
  regime
- reducing `beta_rep` pulls the system back somewhat, but does not beat
  `boundary_objective_v1`
- conditionally masking answer CE strongly worsens no-overturn, which rules
  out the simplest global-answer-CE contamination hypothesis

That shifts attention inside the propagation objective itself.

The most asymmetric propagation component in the current objective is the
replace-side boundary term:

- preserve-side margin mainly pushes `s_early` up relative to alternatives
- replace-side margin pushes `s_gold` up and `s_early` down at the same time

The minimal structural hypothesis tested here was:

- perhaps the direct anti-early gradient from the replace margin is what
  over-couples replace behavior into the shared answer geometry
- if so, keeping the forward boundary but removing replace-margin gradient flow
  into `s_early` should reduce aggressive overturn pressure

## Implemented Propagation Change

All score semantics remained:

- `s_y = log p(y | x)`

Preserve-side loss remained unchanged:

- `L_preserve = CE(answer_dist, y_early) + beta_pres * max(0, m_pres - (s_early - max_non_early_score))`

Baseline replace-side loss was:

- `L_replace_base = CE(answer_dist, y_gold) + beta_rep * max(0, m_rep - (s_gold - s_early))`

This run changed only the replace margin to:

- `L_replace_stopgrad = CE(answer_dist, y_gold) + beta_rep * max(0, m_rep - (s_gold - stopgrad(s_early)))`

So:

- forward values stay the same
- only the gradient path changes
- the replace margin still rewards larger `s_gold - s_early`
- but it no longer directly pushes `s_early` downward through the margin term

Total loss remained:

- `L = L_ctrl + lambda_ans * L_ans + lambda_pres * L_preserve + lambda_rep * L_replace_stopgrad`

## Implementation

Updated code path:

- [src/models/hf_commitment_control_model.py](/workspace/training/src/models/hf_commitment_control_model.py:185)
- [tests/test_hf_commitment_control_model.py](/workspace/training/tests/test_hf_commitment_control_model.py:1)

New config:

- [configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v5_replace_margin_stopgrad_early_v1.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v5_replace_margin_stopgrad_early_v1.json)

Exact training command:

```bash
.venv/bin/python training/train_commitment_control_hf.py --config configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v5_replace_margin_stopgrad_early_v1.json
```

## Exact Config Used

Key delta relative to `boundary_objective_v1`:

- `propagation_variant = "boundary_objective_v5_replace_margin_stopgrad_early_v1"`

Everything else stayed fixed:

- `answer_loss_weight = 1.0`
- `lambda_pres = 0.20`
- `lambda_rep = 0.08`
- `beta_pres = 0.10`
- `beta_rep = 0.05`
- `m_pres = 0.30`
- `m_rep = 0.50`

Run directory:

- `runs/cipc_belief_r_qwen05b_lora_boundary_objective_v5_replace_margin_stopgrad_early_v1`

## Result

Test-set answer metrics:

- overall answer: `0.8615`
- overturn answer: `0.9709`
- no-overturn answer: `0.4444`

Supporting test summary:

- control decision accuracy: `0.8846`
- joint accuracy: `0.8538`
- early commitment persistence: `0.0291`
- late evidence takeover: `0.9709`

## Comparison

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| boundary_objective_v5_replace_margin_stopgrad_early_v1 | 0.8615 | 0.9709 | 0.4444 |
| boundary_objective_v1 | 0.8615 | 0.9223 | 0.6296 |
| boundary_objective_v2_beta_pres_only | 0.8615 | 0.9515 | 0.5185 |
| boundary_objective_v3_beta_rep_only | 0.8385 | 0.9029 | 0.5926 |
| boundary_objective_v4_conditional_masked_answer_loss_v1 | 0.8308 | 0.9709 | 0.2963 |
| highrank_v1 | 0.8538 | 0.9029 | 0.6667 |
| preserve_hybrid_v1 | 0.8615 | 0.9223 | 0.6296 |
| tradeoff_repair_v2 | 0.8000 | 0.7961 | 0.8148 |

## Readout

### 1. Did removing the replace-margin anti-early gradient improve the preserve-versus-revise frontier?

No.

Relative to `boundary_objective_v1`:

- overall stayed flat
  - `0.8615 -> 0.8615`
- overturn increased strongly
  - `0.9223 -> 0.9709`
- no-overturn dropped sharply
  - `0.6296 -> 0.4444`

So the direct anti-early gradient was not the main cause of aggressive
overturn behavior in the way hypothesized.

### 2. What does this imply about propagation-internal coupling?

It suggests the coupling is not primarily coming from the replace margin's
gradient on `s_early`.

The tested hypothesis was:

- replace margin may be too aggressive because it explicitly pushes `s_early`
  down

What actually happened:

- stopping that gradient did not protect preserve-side behavior
- the model still became more overturn-heavy

The clearest interpretation is:

- the replace-side CE plus the rest of the objective already supply enough
  pressure toward aggressive revise behavior
- the anti-early gradient inside the replace margin is not the unique or
  dominant structural culprit

### 3. Most important empirical fact

This run matched `boundary_objective_v1` on overall answer while strongly
increasing overturn and strongly reducing no-overturn.

That means the stopgrad change is active and not neutral, but the direction is
again wrong for the active project goal.

It also tells us something sharper than the earlier coefficient runs:

- even when the replace margin no longer backpropagates through `s_early`,
  the system still shifts toward a highly aggressive revise regime

So the preserve/replace bottleneck is deeper than one gradient path on the
replace margin.

## Bottom Line

`boundary_objective_v5_replace_margin_stopgrad_early_v1` is not the repair.

- It did not improve the balanced frontier relative to `boundary_objective_v1`.
- It increased overturn substantially.
- It degraded no-overturn substantially.

This is a negative but informative structural result.

It rules out the simple story that the replace-margin anti-early gradient is
the main propagation-internal source of the current trade-off failure.

## Most Principled Next Step

Return to `boundary_objective_v1` as the clean baseline.

The most principled next step is:

- keep answer CE unchanged
- keep the current backbone and head structure
- stop testing single-gradient-path removals
- test the next minimal structural change that alters how preserve-side and
  replace-side propagation terms are balanced relative to each other at the
  objective level, not just inside one replace subterm

In practical terms:

- do not reopen broad sweeps
- do not change the dataset
- the next structural test should target propagation-term interaction or
  propagation-term balancing more globally than a single stopgrad edge
