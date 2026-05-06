# Qwen 0.5B Boundary Objective V1 Report

## Scope

This run continued the current main Belief-R HF/LoRA CIPC line and tested one
new theory-guided boundary-based propagation objective.

Kept fixed:

- dataset and split
- seed
- backbone
- LoRA rank/config
- shared encoder + control head + answer head structure
- preserve/replace split aggregation
- sampler
- epoch count
- `answer_loss_weight`

Changed:

- added a new propagation-loss variant:
  `boundary_objective_v1`
- kept score semantics aligned with the existing margin line by defining
  every score as a log-probability:
  `s_y = log p(y | x)`

## Theoretical Motivation

The current project bottleneck is preserve-vs-revise trade-off repair.

Earlier runs established:

- replace-side behavior is easier to learn
- preserve-side behavior is the main difficulty
- scalar weight changes can move the trade-off
- but they do not by themselves yield a principled preserve-region repair

This run therefore moved from empirical redesign toward a cleaner
theory-guided objective.

The idea is to define explicit answer-score regions:

1. Preserve region:
   the early answer should beat every alternative by a margin
2. Replace region:
   the gold final answer should beat the early answer by a margin

The propagation losses then act as margin-based surrogates for those two
regions.

## Implemented Formulas

All answer scores use log-softmax semantics:

- `s_early = log p(early_answer | x)`
- `s_gold = log p(gold_final_answer | x)`
- `max_non_early_score = max_{y != y_early} log p(y | x)`

Preserve boundary:

- `s_early - max_non_early_score >= m_pres`

Replace boundary:

- `s_gold - s_early >= m_rep`

Implemented preserve-side loss:

- `L_preserve = CE(final_answer_dist, early_answer) + beta_pres * max(0, m_pres - (s_early - max_non_early_score))`

Implemented replace-side loss:

- `L_replace = CE(final_answer_dist, gold_final_answer) + beta_rep * max(0, m_rep - (s_gold - s_early))`

Implemented total loss:

- `L = L_ctrl + lambda_ans * L_ans + lambda_pres * L_preserve + lambda_rep * L_replace`

This was implemented as a new loss path without overwriting the previous
conditional-consistency variants.

## Implementation

Updated code path:

- [src/models/hf_commitment_control_model.py](/workspace/training/src/models/hf_commitment_control_model.py:111)
- [training/train_commitment_control_hf.py](/workspace/training/training/train_commitment_control_hf.py:351)
- [tests/test_hf_commitment_control_model.py](/workspace/training/tests/test_hf_commitment_control_model.py:1)

New config:

- [configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v1.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v1.json)

Exact training command:

```bash
.venv/bin/python training/train_commitment_control_hf.py --config configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v1.json
```

## Exact Config Used

- `answer_loss_weight = 1.0`
- `lambda_pres = 0.20`
- `lambda_rep = 0.08`
- `beta_pres = 0.10`
- `beta_rep = 0.05`
- `m_pres = 0.30`
- `m_rep = 0.50`

Other fixed settings remained identical to the closest preserve-hybrid parent:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- `lora_r = 16`
- `lora_alpha = 32`
- `lora_dropout = 0.05`
- `num_epochs = 6`
- oversample control labels: `true`
- seed: `7`

Run directory:

- `runs/cipc_belief_r_qwen05b_lora_boundary_objective_v1`

## Result

Test-set answer metrics:

- overall answer: `0.8615`
- overturn answer: `0.9223`
- no-overturn answer: `0.6296`

Supporting test summary:

- control decision accuracy: `0.8615`
- joint accuracy: `0.8462`

## Comparison

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| boundary_objective_v1 | 0.8615 | 0.9223 | 0.6296 |
| highrank_v1 | 0.8538 | 0.9029 | 0.6667 |
| tradeoff_repair_v2 | 0.8000 | 0.7961 | 0.8148 |
| conditional_consistency_v2 | 0.8154 | 0.8738 | 0.5926 |
| preserve_margin_v1 | 0.8077 | 0.8350 | 0.7037 |
| preserve_margin_v2 | 0.8154 | 0.8544 | 0.6667 |
| preserve_hybrid_v1 | 0.8615 | 0.9223 | 0.6296 |

## Readout

### 1. Did the new boundary objective improve the current frontier?

No.

It did beat several earlier preserve-side repair attempts:

- better overall than `tradeoff_repair_v2`
- better overall than `conditional_consistency_v2`
- better overall than `preserve_margin_v1`
- better overall than `preserve_margin_v2`

It also beat `highrank_v1` on:

- overall answer
- overturn answer

But it did not beat `highrank_v1` on no-overturn:

- `0.6296 < 0.6667`

So it is still an aggressive frontier point rather than a repaired balanced
frontier point.

### 2. Did the result match the theory prediction?

Only partially.

The theory prediction was:

- preserve-side boundary supervision should define a cleaner preserve region
- replace-side boundary supervision should retain strong overturn behavior
- observed trade-off movement should therefore become more interpretable

What happened:

- overturn remained strong
- the objective stayed on a clean aggressive point
- but preserve-side accuracy did not recover beyond the current aggressive
  frontier anchor

So the new objective matched the directional expectation on replace-side
stability, but not the hoped-for preserve-side gain.

### 3. What is the most important empirical fact?

`boundary_objective_v1` exactly matched `preserve_hybrid_v1` on the reported
test metrics:

- overall: `0.8615`
- overturn: `0.9223`
- no-overturn: `0.6296`

That means the new theory-guided rewrite did not create a new observable
frontier movement at this initial setting.

The cleanest interpretation is:

- the old preserve-hybrid design was already functionally close to this
  boundary formulation under these weights
- the extra theoretical clarity did not by itself increase preserve-side
  control strength

## Is This Cleaner Than The Previous Empirical Redesigns?

Yes in formulation, no in empirical outcome.

Cleaner in formulation:

- it states preserve and replace objectives directly as score-space boundaries
- it uses one consistent score semantics with the prior margin line
- it is easier to reason about than the earlier sequence of empirical tweaks

Not cleaner in empirical outcome:

- at the tested setting, it did not separate itself from
  `preserve_hybrid_v1`
- it therefore did not yet provide a stronger practical explanation of the
  preserve-side trade-off

So this run is still useful because it makes the objective family more
principled, even though it does not yet improve the frontier.

## Bottom Line

`boundary_objective_v1` is a cleaner objective statement of the current
preserve/replace theory, but it is not yet the preserve-side repair.

- It improved overall and overturn relative to `highrank_v1`.
- It failed to improve no-overturn relative to `highrank_v1`.
- It exactly matched `preserve_hybrid_v1` on the reported test metrics.

So the first boundary-based run does not falsify the theory-guided direction,
but it does show that theory-guided reformulation alone is insufficient at this
setting.

## Most Principled Next Step

Keep the boundary formulation, but change the preserve-side pressure within the
same theory-guided family rather than returning to empirical loss tweaking.

The most principled next step is:

- keep the score-space boundary view
- keep the same split, seed, backbone, LoRA rank, sampler, and epoch count
- test a preserve-stronger boundary follow-up that changes only preserve-side
  strength or preserve-side weighting
- verify whether no-overturn moves upward in the direction predicted by the
  preserve-region theory

In other words:

- do not abandon the boundary objective
- do not reopen broad heuristic sweeps
- use this run as the clean theoretical baseline for the next preserve-side
  boundary-strength test
