# Qwen 0.5B Boundary Objective V3 Beta-Rep Only Report

## Scope

This run continued the current main Belief-R HF/LoRA CIPC line and executed
exactly one follow-up experiment after the two earlier boundary-objective runs.

Kept fixed:

- dataset and split
- seed
- backbone
- LoRA rank/config
- shared encoder + control head + answer head structure
- preserve/replace split aggregation
- sampler
- epoch count
- score semantics: `s_y = log p(y | x)`
- boundary-based preserve/replace formulation
- preserve-side logic

Changed:

- replace-side boundary strength only:
  `beta_rep: 0.05 -> 0.02`

This run therefore isolates the effect of reducing replace-side pressure
within the same theory-guided boundary family.

## Theory Motivation

The two earlier boundary runs established:

- `boundary_objective_v1` was theoretically clean but still sat on an
  aggressive frontier point
- `boundary_objective_v2_beta_pres_only` showed that increasing
  `beta_pres` moved the frontier in the wrong direction:
  overturn increased and no-overturn collapsed

That result suggested:

- preserve-side pressure is an active control knob
- but under the current coupling it does not act as a maintain-side repair
  mechanism

So the next clean test was not to further increase preserve-side pressure.
The more principled alternative was to weaken replace-side pressure and test
whether excessive replace-side dominance is what keeps the system trapped in
an overly aggressive overturn regime.

## Exact Config Delta Relative To Boundary Objective V1

Parent config:

- [configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v1.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v1.json)

New config:

- [configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v3_beta_rep_only.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v3_beta_rep_only.json)

Single delta:

- `beta_rep: 0.05 -> 0.02`

Everything else stayed fixed:

- `answer_loss_weight = 1.0`
- `lambda_pres = 0.20`
- `lambda_rep = 0.08`
- `beta_pres = 0.10`
- `m_pres = 0.30`
- `m_rep = 0.50`
- `propagation_variant = "boundary_objective_v1"`

## Implemented Boundary Objective

All answer scores use log-probability semantics:

- `s_early = log p(early_answer | x)`
- `s_gold = log p(gold_final_answer | x)`
- `max_non_early_score = max_{y != y_early} log p(y | x)`

Preserve boundary:

- `s_early - max_non_early_score >= m_pres`

Replace boundary:

- `s_gold - s_early >= m_rep`

Preserve-side loss:

- `L_preserve = CE(final_answer_dist, early_answer) + beta_pres * max(0, m_pres - (s_early - max_non_early_score))`

Replace-side loss:

- `L_replace = CE(final_answer_dist, gold_final_answer) + beta_rep * max(0, m_rep - (s_gold - s_early))`

Total loss:

- `L = L_ctrl + lambda_ans * L_ans + lambda_pres * L_preserve + lambda_rep * L_replace`

## Exact Config Used

Run directory:

- `runs/cipc_belief_r_qwen05b_lora_boundary_objective_v3_beta_rep_only`

Exact training command:

```bash
.venv/bin/python training/train_commitment_control_hf.py --config configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v3_beta_rep_only.json
```

Key settings:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- `lora_r = 16`
- `lora_alpha = 32`
- `lora_dropout = 0.05`
- `num_epochs = 6`
- seed: `7`
- oversample control labels: `true`
- `beta_rep = 0.02`

## Result

Test-set answer metrics:

- overall answer: `0.8385`
- overturn answer: `0.9029`
- no-overturn answer: `0.5926`

Supporting test summary:

- control decision accuracy: `0.8308`
- joint accuracy: `0.8000`
- early commitment persistence: `0.0971`
- late evidence takeover: `0.9029`

## Comparison

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| boundary_objective_v3_beta_rep_only | 0.8385 | 0.9029 | 0.5926 |
| highrank_v1 | 0.8538 | 0.9029 | 0.6667 |
| tradeoff_repair_v2 | 0.8000 | 0.7961 | 0.8148 |
| preserve_hybrid_v1 | 0.8615 | 0.9223 | 0.6296 |
| boundary_objective_v1 | 0.8615 | 0.9223 | 0.6296 |
| boundary_objective_v2_beta_pres_only | 0.8615 | 0.9515 | 0.5185 |

## Readout

### 1. Did lowering replace-side pressure move the frontier toward a more balanced region?

Yes directionally, but not enough to produce a better frontier point.

Relative to `boundary_objective_v2_beta_pres_only`:

- overall dropped
  - `0.8615 -> 0.8385`
- overturn dropped
  - `0.9515 -> 0.9029`
- no-overturn improved
  - `0.5185 -> 0.5926`

So reducing replace-side pressure clearly moved the system away from the most
aggressive overturn regime and partially recovered maintain-side behavior.

Relative to `boundary_objective_v1`, however:

- overall dropped
  - `0.8615 -> 0.8385`
- overturn dropped
  - `0.9223 -> 0.9029`
- no-overturn also dropped
  - `0.6296 -> 0.5926`

So although the movement was directionally more balanced than `v2`, it did not
beat the cleaner `v1` baseline.

### 2. What does this say about replace-side dominance?

It looks like a real part of the remaining bottleneck, but not the whole one.

Evidence that replace-side dominance matters:

- lowering `beta_rep` reduced the extreme overturn bias seen in
  `boundary_objective_v2_beta_pres_only`
- no-overturn recovered from `0.5185` to `0.5926`

Evidence that it is not the entire bottleneck:

- the recovered point is still worse than `boundary_objective_v1` on
  no-overturn
- overall also fell below both `boundary_objective_v1` and `highrank_v1`

So replace-side pressure is a meaningful control axis, but simply weakening it
does not yet yield the desired preserve-region repair.

### 3. Most important empirical fact

This run cleanly separates two claims:

1. Replace-side pressure contributes to overly aggressive behavior.
2. Reducing replace-side pressure alone is not enough to create a superior
   balanced frontier point.

That is a useful diagnosis.

Compared with `boundary_objective_v2_beta_pres_only`, the system became less
aggressive in exactly the expected way.
Compared with `boundary_objective_v1`, it still underperformed on both overall
and no-overturn.

So the boundary family is now showing interpretable directional behavior on
both preserve-side and replace-side knobs, but neither scalar change alone is
the repair.

## Bottom Line

`boundary_objective_v3_beta_rep_only` is not the repair, but it is
diagnostically useful.

- It lowered overturn relative to the over-aggressive `v2` point.
- It partially recovered no-overturn relative to `v2`.
- It did not recover enough to beat `boundary_objective_v1`.

This means the frontier did move toward a more balanced region relative to the
most aggressive point, but not toward a better balanced frontier than the
original boundary baseline.

## Is Replace-Side Dominance The Main Remaining Bottleneck?

It appears to be one major bottleneck, but not the sole one.

The current evidence suggests:

- stronger preserve-side pressure can also backfire
- stronger replace-side pressure can over-aggressively favor overturn
- weaker replace-side pressure can relieve some of that bias
- but the preserve-side region still does not stabilize enough to beat the
  `boundary_objective_v1` anchor

So the bottleneck is not just "too much replace."
It is the coupled geometry between preserve-region enforcement and
replace-region enforcement under the shared answer head.

## Most Principled Next Step

Keep the boundary-based formulation and stop treating either scalar knob by
itself as the repair.

The most principled next step is:

- keep split, seed, backbone, LoRA rank, sampler, epoch count, and score
  semantics fixed
- keep the preserve/replace boundary formulation itself
- move from single-scalar pressure changes to a cleaner redesign of how the
  preserve and replace terms are balanced structurally

In practical terms:

- do not reopen heuristic sweeps
- do not abandon the boundary family
- use `boundary_objective_v1` as the clean baseline
- next test should target the coupling structure between preserve and replace,
  rather than only increasing or decreasing one boundary coefficient
