# 2026-05-06 Boundary Objective Experiment Summary

## Context

Date of work: `2026-05-06` (UTC)

Branch and remote at report time:

- branch: `main`
- remote: `origin -> git@github.com:laiyuchen164-creator/training.git`

This report consolidates the full experiment block completed today on the
Belief-R HF/LoRA CIPC training line using `Qwen/Qwen2.5-0.5B-Instruct`.

The day focused on one question:

- can a cleaner theory-guided boundary objective repair the
  preserve-vs-revise trade-off better than the current aggressive frontier?

The work produced:

- 5 new training configs under [configs](/workspace/training/configs)
- 5 completed run directories under [runs](/workspace/training/runs)
- 5 per-run reports under [analysis](/workspace/training/analysis)
- local model/trainer/test updates enabling new objective variants

## Files Added Or Updated Today

Core implementation:

- [src/models/hf_commitment_control_model.py](/workspace/training/src/models/hf_commitment_control_model.py:108)
- [training/train_commitment_control_hf.py](/workspace/training/training/train_commitment_control_hf.py:356)
- [tests/test_hf_commitment_control_model.py](/workspace/training/tests/test_hf_commitment_control_model.py:1)

New configs:

- [configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v1.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v1.json)
- [configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v2_beta_pres_only.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v2_beta_pres_only.json)
- [configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v3_beta_rep_only.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v3_beta_rep_only.json)
- [configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v4_conditional_masked_answer_loss_v1.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v4_conditional_masked_answer_loss_v1.json)
- [configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v5_replace_margin_stopgrad_early_v1.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v5_replace_margin_stopgrad_early_v1.json)

Per-run reports:

- [analysis/cipc_qwen05b_boundary_objective_v1_report.md](/workspace/training/analysis/cipc_qwen05b_boundary_objective_v1_report.md)
- [analysis/cipc_qwen05b_boundary_objective_v2_beta_pres_only_report.md](/workspace/training/analysis/cipc_qwen05b_boundary_objective_v2_beta_pres_only_report.md)
- [analysis/cipc_qwen05b_boundary_objective_v3_beta_rep_only_report.md](/workspace/training/analysis/cipc_qwen05b_boundary_objective_v3_beta_rep_only_report.md)
- [analysis/cipc_qwen05b_boundary_objective_v4_conditional_masked_answer_loss_v1_report.md](/workspace/training/analysis/cipc_qwen05b_boundary_objective_v4_conditional_masked_answer_loss_v1_report.md)
- [analysis/cipc_qwen05b_boundary_objective_v5_replace_margin_stopgrad_early_v1_report.md](/workspace/training/analysis/cipc_qwen05b_boundary_objective_v5_replace_margin_stopgrad_early_v1_report.md)

## Implementation Summary

Today’s code changes introduced three new capabilities.

### 1. Boundary-style propagation loss

Added `compute_boundary_propagation_loss(...)` in
[src/models/hf_commitment_control_model.py](/workspace/training/src/models/hf_commitment_control_model.py:111).

This objective rewrites preserve and replace supervision directly in
log-probability score space:

- preserve boundary:
  `s_early - max_non_early_score >= m_pres`
- replace boundary:
  `s_gold - s_early >= m_rep`

with:

- preserve loss:
  `CE(y_early) + beta_pres * relu(m_pres - preserve_gap)`
- replace loss:
  `CE(y_gold) + beta_rep * relu(m_rep - replace_gap)`

This is the clean theoretical baseline for the day’s block.

### 2. Conditional answer-loss variant

Added `compute_conditionally_masked_answer_loss(...)` in
[src/models/hf_commitment_control_model.py](/workspace/training/src/models/hf_commitment_control_model.py:254).

This switches answer CE targets by control type:

- preserve rows supervise toward `y_early`
- replace rows supervise toward `y_gold`
- weaken rows supervise toward `y_gold`

and averages the non-empty group means.

### 3. Replace-margin stopgrad variant

Added `compute_boundary_propagation_loss_replace_margin_stopgrad_early(...)`
in [src/models/hf_commitment_control_model.py](/workspace/training/src/models/hf_commitment_control_model.py:184).

This keeps the forward replace margin unchanged but replaces `s_early` with
`stopgrad(s_early)` inside the replace-margin term, removing the direct
anti-early gradient path from that subterm.

### 4. Trainer plumbing and tests

[training/train_commitment_control_hf.py](/workspace/training/training/train_commitment_control_hf.py:356)
was extended so configs can select:

- `answer_loss_variant`
- `propagation_variant`
- `beta_pres`
- `beta_rep`
- `m_pres`
- `m_rep`

[tests/test_hf_commitment_control_model.py](/workspace/training/tests/test_hf_commitment_control_model.py:1)
was expanded to verify:

- boundary terms use log-probability scores
- weaken rows stay neutral
- conditional answer targets follow control labels
- groupwise masked answer-loss averaging is correct
- stopgrad variant matches the baseline forward values

## Shared Training Setup

Unless stated otherwise, all five runs kept the same backbone and training
setup:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`
- seed: `7`
- epochs: `6`
- answer loss weight: `1.0`
- oversample control labels: `true`
- same dataset and split
- same shared encoder + control head + answer head structure

## Experiment Matrix

| run | main change | overall | overturn | no-overturn | control acc | joint acc |
|---|---|---:|---:|---:|---:|---:|
| `boundary_objective_v1` | clean boundary baseline | 0.8615 | 0.9223 | 0.6296 | 0.8615 | 0.8462 |
| `boundary_objective_v2_beta_pres_only` | `beta_pres 0.10 -> 0.20` | 0.8615 | 0.9515 | 0.5185 | 0.8692 | 0.8538 |
| `boundary_objective_v3_beta_rep_only` | `beta_rep 0.05 -> 0.02` | 0.8385 | 0.9029 | 0.5926 | 0.8308 | 0.8000 |
| `boundary_objective_v4_conditional_masked_answer_loss_v1` | conditional masked answer CE | 0.8308 | 0.9709 | 0.2963 | 0.8385 | 0.8231 |
| `boundary_objective_v5_replace_margin_stopgrad_early_v1` | stopgrad on replace margin `s_early` | 0.8615 | 0.9709 | 0.4444 | 0.8846 | 0.8538 |

## Training Dynamics

Final logged dev metrics from `train_log.jsonl`:

| run | epoch 6 train loss | dev control acc | dev answer acc | dev joint acc |
|---|---:|---:|---:|---:|
| `boundary_objective_v1` | 0.1345 | 0.8189 | 0.7795 | 0.7717 |
| `boundary_objective_v2_beta_pres_only` | 0.1528 | 0.8110 | 0.8189 | 0.7953 |
| `boundary_objective_v3_beta_rep_only` | 0.1336 | 0.8031 | 0.7717 | 0.7402 |
| `boundary_objective_v4_conditional_masked_answer_loss_v1` | 0.1618 | 0.8031 | 0.7953 | 0.7717 |
| `boundary_objective_v5_replace_margin_stopgrad_early_v1` | 0.1675 | 0.8031 | 0.8189 | 0.7795 |

Important pattern:

- dev aggregate metrics looked competitive for `v2`, `v4`, and `v5`
- but test no-overturn degraded badly for those same runs
- this means aggregate dev metrics are not sufficient to track the actual
  preserve-vs-revise objective
- the split metrics remain the correct decision signal

## Detailed Run Notes

### 1. `boundary_objective_v1`

Report:
[analysis/cipc_qwen05b_boundary_objective_v1_report.md](/workspace/training/analysis/cipc_qwen05b_boundary_objective_v1_report.md)

Run directory:
`runs/cipc_belief_r_qwen05b_lora_boundary_objective_v1`

Training command:

```bash
.venv/bin/python training/train_commitment_control_hf.py --config configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v1.json
```

Exact key settings:

- `lambda_pres = 0.20`
- `lambda_rep = 0.08`
- `beta_pres = 0.10`
- `beta_rep = 0.05`
- `m_pres = 0.30`
- `m_rep = 0.50`
- `propagation_variant = "boundary_objective_v1"`

Interpretation:

- this is the clean theoretical rewrite of the earlier preserve-hybrid idea
- empirically it did not beat the aggressive frontier
- it exactly matched `preserve_hybrid_v1` on the reported test metrics
- therefore the rewrite improved conceptual cleanliness more than empirical
  performance

Why it matters:

- it became the clean baseline for the rest of the day
- it showed that theory cleanup alone does not repair no-overturn

### 2. `boundary_objective_v2_beta_pres_only`

Report:
[analysis/cipc_qwen05b_boundary_objective_v2_beta_pres_only_report.md](/workspace/training/analysis/cipc_qwen05b_boundary_objective_v2_beta_pres_only_report.md)

Run directory:
`runs/cipc_belief_r_qwen05b_lora_boundary_objective_v2_beta_pres_only`

Only delta from `v1`:

- `beta_pres: 0.10 -> 0.20`

Observed effect:

- overall stayed flat at `0.8615`
- overturn increased from `0.9223` to `0.9515`
- no-overturn dropped from `0.6296` to `0.5185`

Interpretation:

- preserve-side boundary strength is an active knob
- but under current coupling it behaves in the wrong direction
- stronger nominal preserve pressure does not produce preserve recovery
- instead it drives the model into a more aggressive revise regime

Why it matters:

- this was the clearest evidence today that scalar pressure on the preserve
  side is not inert
- but the geometry it induces is not the geometry we want

### 3. `boundary_objective_v3_beta_rep_only`

Report:
[analysis/cipc_qwen05b_boundary_objective_v3_beta_rep_only_report.md](/workspace/training/analysis/cipc_qwen05b_boundary_objective_v3_beta_rep_only_report.md)

Run directory:
`runs/cipc_belief_r_qwen05b_lora_boundary_objective_v3_beta_rep_only`

Only delta from `v1`:

- `beta_rep: 0.05 -> 0.02`

Observed effect:

- overall fell to `0.8385`
- overturn fell to `0.9029`
- no-overturn recovered relative to `v2`, reaching `0.5926`

Interpretation:

- replace-side pressure is also a real control axis
- weakening it reduces the extreme overturn bias seen in `v2`
- but this alone is not enough to create a better frontier point than `v1`

Why it matters:

- it confirms the boundary family is directionally interpretable
- preserve and replace knobs do move the system in predictable directions
- but neither knob alone repairs the trade-off

### 4. `boundary_objective_v4_conditional_masked_answer_loss_v1`

Report:
[analysis/cipc_qwen05b_boundary_objective_v4_conditional_masked_answer_loss_v1_report.md](/workspace/training/analysis/cipc_qwen05b_boundary_objective_v4_conditional_masked_answer_loss_v1_report.md)

Run directory:
`runs/cipc_belief_r_qwen05b_lora_boundary_objective_v4_conditional_masked_answer_loss_v1`

Only structural delta from `v1`:

- `answer_loss_variant = "conditional_masked_v1"`

Observed effect:

- overall dropped to `0.8308`
- overturn surged to `0.9709`
- no-overturn collapsed to `0.2963`

Interpretation:

- the simplest version of the "global gold-answer CE is contaminating
  preserve behavior" hypothesis is falsified
- global answer CE was not just noise or leakage
- at this operating point it was acting as a stabilizer

Why it matters:

- this is a negative result, but a strong one
- it rules out one simple structural story and narrows the next search space

### 5. `boundary_objective_v5_replace_margin_stopgrad_early_v1`

Report:
[analysis/cipc_qwen05b_boundary_objective_v5_replace_margin_stopgrad_early_v1_report.md](/workspace/training/analysis/cipc_qwen05b_boundary_objective_v5_replace_margin_stopgrad_early_v1_report.md)

Run directory:
`runs/cipc_belief_r_qwen05b_lora_boundary_objective_v5_replace_margin_stopgrad_early_v1`

Only structural delta from `v1`:

- `propagation_variant = "boundary_objective_v5_replace_margin_stopgrad_early_v1"`

Observed effect:

- overall stayed at `0.8615`
- overturn rose to `0.9709`
- no-overturn dropped to `0.4444`

Interpretation:

- the direct anti-early gradient inside the replace margin is not the main
  culprit behind the aggressive revise bias
- removing that one gradient path does not protect preserve-side behavior

Why it matters:

- this rules out another clean, local structural explanation
- the bottleneck is deeper than one replace-margin edge

## Cross-Run Conclusions

### 1. Best clean baseline from today

`boundary_objective_v1` remains the best clean boundary-style baseline.

Reason:

- it matches the strongest overall answer level reached today
- it avoids the worst no-overturn collapses
- it is conceptually cleaner than the earlier empirical preserve-hybrid path

### 2. What was learned about the objective geometry

The day established four concrete facts:

1. The boundary family is active.
   Small coefficient changes produced large and interpretable frontier moves.
2. Preserve-side pressure is not a reliable preserve repair.
   Increasing `beta_pres` made revise behavior more aggressive.
3. Replace-side pressure matters, but weakening it is insufficient.
   Lower `beta_rep` partially restored balance but did not beat the baseline.
4. Two simple structural explanations were ruled out.
   Conditional masked answer CE failed badly, and stopgrad on the replace
   margin early score also failed.

### 3. Strongest empirical signal from the day

The main project metric is still the preserve-vs-revise split, not aggregate
dev answer accuracy.

Evidence:

- `v2`, `v4`, and `v5` each reached strong dev aggregate numbers
- all three degraded no-overturn meaningfully on the held-out test set
- the frontier can therefore look better under aggregate dev logging while
  getting worse on the actual preserve objective

### 4. Practical ranking for today’s five runs

Ordered by usefulness for the active research line:

1. `boundary_objective_v1`
   Best clean baseline and reference point.
2. `boundary_objective_v3_beta_rep_only`
   Useful directional diagnosis for replace-side pressure.
3. `boundary_objective_v2_beta_pres_only`
   Strong evidence that preserve-side pressure behaves counterintuitively.
4. `boundary_objective_v5_replace_margin_stopgrad_early_v1`
   Negative structural result that rules out one simple explanation.
5. `boundary_objective_v4_conditional_masked_answer_loss_v1`
   Strong negative result; useful diagnostically, but worst preserve behavior.

## Recommended Next Step

The most principled next move is to keep `boundary_objective_v1` as the base
and stop testing isolated single-edge or single-coefficient repairs.

Recommended direction:

- keep answer CE in its original global form
- keep the clean boundary formulation as the anchor objective
- focus the next experiment on higher-level interaction between preserve and
  replace propagation terms
- evaluate using split metrics first, not aggregate dev answer accuracy

Concretely, today’s results suggest the next useful change should alter
preserve/replace interaction at the objective level rather than:

- simply increasing one boundary weight
- masking answer targets by condition
- or removing one gradient path inside the replace margin

## Bottom Line

Today did not produce a new best balanced frontier point.

It did produce a cleaner and tighter diagnosis:

- the boundary objective family is real and controllable
- the preserve/revise trade-off is not repaired by simple coefficient nudges
- the simplest answer-loss contamination hypothesis is wrong
- the simplest replace-margin anti-early-gradient hypothesis is also wrong

That makes the day successful as a research narrowing step even though it did
not produce a direct performance win.
