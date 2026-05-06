# Qwen 0.5B Boundary Objective V2 Beta-Pres Only Report

## Scope

This run continued the current main Belief-R HF/LoRA CIPC line and executed
exactly one follow-up experiment after `boundary_objective_v1`.

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
- replace-side logic

Changed:

- preserve-side boundary strength only:
  `beta_pres: 0.10 -> 0.20`

This run therefore isolates the effect of preserve-side boundary strength
within the same theory-guided objective family.

## Theory Motivation

`boundary_objective_v1` showed that the boundary formulation is theoretically
cleaner than the earlier empirical redesigns, but it landed at exactly the
same operating point as `preserve_hybrid_v1`:

- overall answer: `0.8615`
- overturn answer: `0.9223`
- no-overturn answer: `0.6296`

The working hypothesis for the next clean test was:

- the boundary formulation itself is valid
- the current preserve boundary is too weak to move the model out of the
  aggressive region

Under that hypothesis, the most principled follow-up is not to alter the score
space, not to alter the replace side, and not to reopen heuristic sweeps.
It is to increase preserve-side boundary strength only and observe whether the
frontier moves toward a stronger preserve region.

## Exact Config Delta Relative To Boundary Objective V1

Parent config:

- [configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v1.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v1.json)

New config:

- [configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v2_beta_pres_only.json](/workspace/training/configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v2_beta_pres_only.json)

Single delta:

- `beta_pres: 0.10 -> 0.20`

Everything else stayed fixed:

- `answer_loss_weight = 1.0`
- `lambda_pres = 0.20`
- `lambda_rep = 0.08`
- `beta_rep = 0.05`
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

- `runs/cipc_belief_r_qwen05b_lora_boundary_objective_v2_beta_pres_only`

Exact training command:

```bash
.venv/bin/python training/train_commitment_control_hf.py --config configs/train_cipc_belief_r_qwen05b_lora_boundary_objective_v2_beta_pres_only.json
```

Key settings:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- `lora_r = 16`
- `lora_alpha = 32`
- `lora_dropout = 0.05`
- `num_epochs = 6`
- seed: `7`
- oversample control labels: `true`
- `beta_pres = 0.20`

## Result

Test-set answer metrics:

- overall answer: `0.8615`
- overturn answer: `0.9515`
- no-overturn answer: `0.5185`

Supporting test summary:

- control decision accuracy: `0.8692`
- joint accuracy: `0.8538`
- early commitment persistence: `0.0485`
- late evidence takeover: `0.9515`

## Comparison

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| boundary_objective_v2_beta_pres_only | 0.8615 | 0.9515 | 0.5185 |
| boundary_objective_v1 | 0.8615 | 0.9223 | 0.6296 |
| highrank_v1 | 0.8538 | 0.9029 | 0.6667 |
| tradeoff_repair_v2 | 0.8000 | 0.7961 | 0.8148 |
| preserve_margin_v1 | 0.8077 | 0.8350 | 0.7037 |
| preserve_margin_v2 | 0.8154 | 0.8544 | 0.6667 |
| preserve_hybrid_v1 | 0.8615 | 0.9223 | 0.6296 |

## Readout

### 1. Did increasing `beta_pres` move the frontier in the predicted direction?

Yes, but not in the hoped-for balanced direction.

Relative to `boundary_objective_v1`:

- overall stayed flat
  - `0.8615 -> 0.8615`
- overturn increased strongly
  - `0.9223 -> 0.9515`
- no-overturn dropped sharply
  - `0.6296 -> 0.5185`

So changing only `beta_pres` clearly moved the frontier.
The effect is real and large.

But the movement was toward a more aggressive overturn-heavy point, not toward
maintain-side recovery.

### 2. What does this say about the preserve-side boundary hypothesis?

It supports one part of the hypothesis and falsifies another.

Supported:

- preserve-side boundary strength is not inert
- it is a real control knob inside the boundary formulation

Falsified at this setting:

- increasing preserve-side boundary strength does not automatically improve
  preserve-side outcome quality
- in the current coupled training geometry, stronger preserve-side boundary
  pressure can still route the model toward a more aggressive operating point

This matches the earlier preserve-hybrid `v2` lesson:

- stronger nominal preserve-side pressure can paradoxically increase overturn
  rather than recover maintain behavior

### 3. Is preserve-side boundary strength the main remaining control knob?

It appears to be a major control knob, but not the only missing ingredient.

Why it looks major:

- one scalar change produced a large frontier movement
- the result moved much more than many earlier weight-only repairs

Why it is not sufficient on its own:

- the movement again had the wrong preserve-side sign
- no-overturn fell to `0.5185`, far below `boundary_objective_v1`
  and `highrank_v1`

So preserve-side boundary strength is a genuine lever, but the current
objective coupling does not make that lever behave like a clean maintain-side
repair mechanism.

### 4. Most important empirical fact

This run produced a cleaner diagnosis than `boundary_objective_v1`.

`boundary_objective_v1` could be read as "the new theory is valid but too weak
to matter."

`boundary_objective_v2_beta_pres_only` rules that out.

The stronger statement after this run is:

- the boundary formulation is active
- `beta_pres` does materially control behavior
- but stronger preserve-side boundary pressure alone pushes the system toward
  a more aggressive region rather than a more balanced preserve region

## Bottom Line

`boundary_objective_v2_beta_pres_only` is not the repair.

- It kept overall answer at `0.8615`.
- It improved overturn to `0.9515`.
- It badly hurt no-overturn to `0.5185`.

So the frontier moved, but it moved in the wrong direction for the active
project goal.

This is still a useful result because it shows:

- preserve-side boundary strength is a real and powerful knob
- but it does not currently act as a maintain-side correction knob

## Most Principled Next Step

Keep the boundary-based formulation, but stop treating stronger preserve-side
boundary weight as the direct fix.

The most principled next step after this run is:

- keep the same score semantics and boundary formulation
- keep split, seed, backbone, LoRA rank, sampler, and epoch count fixed
- hold `beta_pres` fixed at the cleaner baseline rather than increasing it
- change the preserve-side boundary shape or preserve-side weighting structure
  in a way that disentangles preserve pressure from the current aggressive
  overturn coupling

In practical terms:

- do not reopen broad sweeps
- do not abandon the boundary family
- treat this run as evidence that scalar preserve-strength increase alone is
  not the repair
- the next theory-guided step should redesign how preserve-region pressure is
  injected, not just how much of it is applied
