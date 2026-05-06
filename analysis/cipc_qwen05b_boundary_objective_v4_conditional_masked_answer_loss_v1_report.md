# Qwen 0.5B Boundary Objective V4 Conditional Masked Answer Loss V1 Report

## Scope

This run continued the current main Belief-R HF/LoRA CIPC line and tested one
minimal structural follow-up after the coefficient-only boundary runs.

Kept fixed:

- dataset and split
- seed
- backbone
- LoRA rank/config
- shared encoder + control head + answer head structure
- preserve/replace boundary formulation
- preserve-side boundary loss
- replace-side boundary loss
- sampler
- epoch count

Changed:

- answer-loss computation only
- the global answer CE path was replaced by a conditionally masked answer-loss
  path while keeping a single scalar `lambda_ans`

This run therefore isolates whether shared-head coupling is primarily caused by
global answer CE supervision.

## Theory Motivation

The previous boundary runs established:

- `boundary_objective_v1` is the clean theoretical baseline
- `boundary_objective_v2_beta_pres_only` showed that increasing preserve-side
  boundary strength pushes the system further into an aggressive overturn
  regime
- `boundary_objective_v3_beta_rep_only` showed that weakening replace-side
  pressure partially pulls the system back, but still does not beat
  `boundary_objective_v1`

That narrowed the bottleneck:

- the issue is no longer just a single coefficient choice
- the issue appears to be coupling between preserve-side and replace-side
  supervision under the shared answer head

The minimal structural hypothesis tested here was:

- perhaps the main coupling is that the same `answer_head` receives
  global gold-answer CE on all examples
  while also receiving preserve-side and replace-side boundary supervision
- if so, conditionally masking answer supervision by control type should
  reduce that coupling and improve the preserve-versus-revise frontier

## Implemented Answer-Loss Design

This run did not change the boundary propagation objective.

It changed only the answer CE path.

### Conditional Targets

For preserve examples:

- `L_ans_preserve = CE(answer_dist, y_early)`

For replace examples:

- `L_ans_replace = CE(answer_dist, y_gold)`

For weaken examples:

- `L_ans_weaken = CE(answer_dist, y_gold)`

### Conditional Batch Aggregation

Within a batch:

- compute the mean CE inside each non-empty condition group
- then average across the non-empty groups

If:

- `A_pres = mean CE over preserve rows`
- `A_rep = mean CE over replace rows`
- `A_weak = mean CE over weaken rows`

then:

- `L_ans_cond = mean({A_pres, A_rep, A_weak} over non-empty groups)`

The total loss remained:

- `L = L_ctrl + lambda_ans * L_ans_cond + lambda_pres * L_preserve + lambda_rep * L_replace`

with a single scalar:

- `lambda_ans = answer_loss_weight = 1.0`

## How This Differs From The Previous Path

The previous path used:

- `L_ans_global = mean_i CE(answer_dist_i, y_gold_i)`

on the full batch with no conditioning.

The new path changed two things:

1. preserve examples no longer use `y_gold` as the answer CE target
2. answer CE is aggregated conditionally across non-empty control groups

So this is the smallest structural test of whether global answer CE is the
dominant source of preserve/replace coupling under the shared answer head.

## Implementation

Updated code path:

- [src/models/hf_commitment_control_model.py](/workspace/training/src/models/hf_commitment_control_model.py:185)
- [training/train_commitment_control_hf.py](/workspace/training/training/train_commitment_control_hf.py:351)
- [tests/test_hf_commitment_control_model.py](/workspace/training/tests/test_hf_commitment_control_model.py:1)

New config:

- [configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v4_conditional_masked_answer_loss_v1.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v4_conditional_masked_answer_loss_v1.json)

Exact training command:

```bash
.venv/bin/python training/train_commitment_control_hf.py --config configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v4_conditional_masked_answer_loss_v1.json
```

## Exact Config Used

Key delta relative to `boundary_objective_v1`:

- `answer_loss_variant = "conditional_masked_v1"`

Everything else stayed fixed:

- `answer_loss_weight = 1.0`
- `propagation_variant = "boundary_objective_v1"`
- `lambda_pres = 0.20`
- `lambda_rep = 0.08`
- `beta_pres = 0.10`
- `beta_rep = 0.05`
- `m_pres = 0.30`
- `m_rep = 0.50`

Run directory:

- `runs/cipc_belief_r_qwen05b_lora_boundary_objective_v4_conditional_masked_answer_loss_v1`

## Result

Test-set answer metrics:

- overall answer: `0.8308`
- overturn answer: `0.9709`
- no-overturn answer: `0.2963`

Supporting test summary:

- control decision accuracy: `0.8385`
- joint accuracy: `0.8231`
- early commitment persistence: `0.0291`
- late evidence takeover: `0.9709`

## Comparison

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| boundary_objective_v4_conditional_masked_answer_loss_v1 | 0.8308 | 0.9709 | 0.2963 |
| boundary_objective_v1 | 0.8615 | 0.9223 | 0.6296 |
| boundary_objective_v2_beta_pres_only | 0.8615 | 0.9515 | 0.5185 |
| boundary_objective_v3_beta_rep_only | 0.8385 | 0.9029 | 0.5926 |
| highrank_v1 | 0.8538 | 0.9029 | 0.6667 |
| preserve_hybrid_v1 | 0.8615 | 0.9223 | 0.6296 |
| tradeoff_repair_v2 | 0.8000 | 0.7961 | 0.8148 |

## Readout

### 1. Did conditional masking of answer supervision reduce preserve/replace coupling in the desired way?

No.

It moved the system strongly in the wrong direction.

Relative to `boundary_objective_v1`:

- overall dropped
  - `0.8615 -> 0.8308`
- overturn rose sharply
  - `0.9223 -> 0.9709`
- no-overturn collapsed
  - `0.6296 -> 0.2963`

So the structural change was not neutral.
It produced a large and highly interpretable frontier movement, but not toward
balanced preserve behavior.

### 2. What does this imply about the shared-head coupling hypothesis?

It falsifies the simplest version of the hypothesis.

The tested hypothesis was:

- global gold-answer CE may be the main source of preserve-side contamination
- replacing it with conditional masking may relieve preserve/replace coupling

What actually happened:

- removing global gold-answer CE on preserve rows did not help preserve cases
- preserve behavior became much worse
- overturn became even more dominant

So global answer CE was not merely a preserve-side pollutant.
At this operating point, it was acting as a stabilizing force that prevented
the answer head from collapsing too far into an overturn-heavy regime.

### 3. Most important empirical fact

This run is the clearest evidence so far that the global answer CE term is
not the primary culprit behind the frontier bottleneck.

If global gold-answer CE were the main harmful coupling source, conditional
masking should have at least directionally helped no-overturn.

Instead:

- no-overturn fell catastrophically to `0.2963`
- overturn rose to `0.9709`

That means the previous coupling diagnosis has to be refined.

The bottleneck is not simply:

- "global answer CE contaminates preserve"

It is more likely:

- the full interaction among answer CE, preserve boundary, and replace boundary
  shapes a shared answer geometry
- and the global answer CE term may actually provide useful anchoring that
  keeps the system from drifting into an even more aggressive overturn regime

## Bottom Line

`boundary_objective_v4_conditional_masked_answer_loss_v1` is not the repair.

- It is worse than `boundary_objective_v1` on overall answer.
- It is much worse on no-overturn.
- It is more aggressive than every earlier boundary variant except in overall
  aggregate.

So the minimal structural masking test was informative, but negative.

It shows that conditionally masking answer supervision does not solve the
shared-head bottleneck and in fact destabilizes the preserve-versus-revise
trade-off in the wrong direction.

## Most Principled Next Step

Keep the boundary formulation and keep the global answer CE path available.

The most principled next step after this run is:

- do not continue removing or conditionally masking answer CE
- treat global answer CE as a useful stabilizer rather than a simple nuisance
- return the clean baseline to `boundary_objective_v1`
- test the next minimal structural change inside the propagation objective
  itself, rather than inside answer supervision

In practical terms:

- no sweep
- no broader architecture rewrite
- no dataset change
- the next structural test should target how preserve-side and replace-side
  propagation terms interact, not how answer CE is masked
