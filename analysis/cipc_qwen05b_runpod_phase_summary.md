# Qwen 0.5B Runpod Phase Summary

## Scope

This memo summarizes the current Runpod phase on the active Belief-R HF/LoRA
line.

The phase stayed on the intended main line:

- Belief-R only
- training-based CIPC only
- HF/LoRA only
- fixed tracked split and seed
- no prompt-line return

## Environment

- preflight: passed
- torch: `2.4.1+cu124`
- gpu: `NVIDIA H100 80GB HBM3`
- tracked Belief-R commitment-control assets: present

## Reference Points

Tracked local references:

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| local control_focused_v1 | 0.7923 | 0.7670 | 0.8889 |
| local highrank_v1 | 0.8462 | 0.8641 | 0.7778 |
| NumPy CIPC baseline | 0.7846 | 0.7961 | 0.9259 |
| frozen prompt baseline | 0.3615 | 0.2718 | 0.5926 |

## Runpod Results

| run | change class | overall answer | overturn answer | no-overturn answer |
|---|---|---:|---:|---:|
| control_focused_v1 | reproduction | 0.8154 | 0.8932 | 0.5185 |
| highrank_v1 | reproduction | 0.8538 | 0.9029 | 0.6667 |
| tradeoff_repair_v1 | `answer_loss_weight=0.7` | 0.8538 | 0.9320 | 0.5556 |
| tradeoff_repair_v2 | `answer_loss_weight=1.3` | 0.8000 | 0.7961 | 0.8148 |
| tradeoff_repair_v3 | `answer_loss_weight=1.1` | 0.8154 | 0.8835 | 0.5556 |
| tradeoff_repair_v4 | `answer_loss_weight=1.2` | 0.7846 | 0.8058 | 0.7037 |
| highrank_select_tradeoff_v1 | selection-only rerun | 0.8538 | 0.9029 | 0.6667 |
| highrank_condition_sampling_v1 | `incremental_no_overturn` 4x sampler | 0.7538 | 0.8058 | 0.5556 |
| highrank_consistency_v1 | `consistency_loss_weight=0.2` | 0.7846 | 0.7670 | 0.8519 |
| highrank_consistency_v2 | `consistency_loss_weight=0.1` | 0.7923 | 0.9126 | 0.3333 |

## Main Findings

### 1. The Runpod frontier does not match the local frontier shape

The local memo suggested:

- `control_focused_v1` is the more balanced point
- `highrank_v1` is the more aggressive point

That did not reproduce on this server.

- Runpod `control_focused_v1` was already very aggressive.
- Runpod `highrank_v1` dominated it on both overall answer accuracy and
  no-overturn answer accuracy.

So the active server-side frontier anchor became:

- aggressive/raw point: Runpod `highrank_v1`

not the local `control_focused_v1`.

### 2. `answer_loss_weight` is the cleanest and most stable trade-off control

Across the `tradeoff_repair_v{1,2,3,4}` runs, changing only
`answer_loss_weight` consistently moved the behavior:

- lower values pushed further toward overturn
- higher values pulled back toward maintain behavior

This was the most stable and interpretable control lever found in the phase.

But no tested value beat the Runpod `highrank_v1` frontier point overall.

### 3. Checkpoint selection was not the main problem

`highrank_select_tradeoff_v1` changed only checkpoint selection.

It reproduced the same result as `highrank_v1`.

That means the current trade-off is not mainly caused by picking the wrong
epoch under the present training trajectory.

### 4. Sampling-only repair failed

`highrank_condition_sampling_v1` changed only the sampler by giving
`incremental_no_overturn` a 4x condition multiplier.

This did not improve no-overturn.

- overall dropped
- overturn dropped
- no-overturn also dropped

So the current trade-off is not fixed by a simple condition-sampling patch.

### 5. Minimal consistency loss is promising in direction but unstable in form

The first minimal consistency design did what it was supposed to do in one
setting:

- `highrank_consistency_v1` recovered no-overturn strongly
- but it gave up too much overturn

Reducing the consistency weight did not produce a smooth frontier:

- `highrank_consistency_v2` swung back to a much more aggressive point
- overturn became even stronger than `highrank_v1`
- no-overturn collapsed badly

So consistency is a real control axis, but the current formulation is not yet
stable enough to serve as the main repair mechanism.

## Current Verdict

The phase successfully narrowed the explanation space.

What now looks true:

- the trade-off is primarily driven by the training objective and shared
  representation geometry
- it is not mainly a checkpoint-selection artifact
- it is not repaired by a simple condition-sampling boost

The best current raw Runpod point remains:

- `highrank_v1`

The strongest conservative anchor remains:

- `tradeoff_repair_v2`

but it is not a better frontier point overall.

## Recommendation

Do not keep sweeping the same three axes:

- `answer_loss_weight`
- simple checkpoint selection
- simple condition sampling

The next main-line step should be:

- redesign the consistency term itself
- then test that redesigned consistency term on top of the current Runpod
  `highrank_v1` base

See:

- `analysis/cipc_qwen05b_consistency_redesign_plan.md`

