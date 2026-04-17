# Frozen Qwen2.5-0.5B Multiple-Choice Baseline

Date: 2026-04-17 UTC

## Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Mode: frozen zero-shot inference
- Input format: early context + early commitment + late evidence + 3-way answer options
- Output format: score candidate continuations `" a"`, `" b"`, `" c"` and choose the highest-scoring option
- Control decision mapping:
  - predict `preserve` if predicted answer equals the early commitment label
  - otherwise predict `replace`
- Script:
  - `analysis/evaluate_frozen_qwen_mc_baseline.py`
- Run artifacts:
  - `runs/frozen_qwen25_05b_mc_baseline_v1/`

This is an external frozen-model baseline, not a trained `CIPC` run and not one
of the repository's earlier prompt-policy baselines.

## Main Result

### Split Summary

| split | n | overall answer | early_commitment_persistence | late_evidence_takeover |
|---|---:|---:|---:|---:|
| train | 2050 | 0.7200 | 0.1167 | 0.8833 |
| dev | 254 | 0.6929 | 0.1584 | 0.8416 |
| test | 260 | 0.7308 | 0.1165 | 0.8835 |

### Test by Condition

| condition | n | answer accuracy |
|---|---:|---:|
| `full_info` | 130 | 0.7308 |
| `incremental_no_overturn` | 27 | 0.1481 |
| `incremental_overturn_reasoning` | 103 | 0.8835 |

## Interpretation

- The baseline is strongly biased toward aggressive revision.
- On the test split it predicts:
  - `replace`: `228 / 260`
  - `preserve`: `32 / 260`
- Predicted final answers are also heavily concentrated:
  - `c`: `228`
  - `a`: `30`
  - `b`: `2`

So the model gets respectable aggregate accuracy mainly by doing well on
`incremental_overturn_reasoning`, but it almost collapses on
`incremental_no_overturn`.

This means the baseline is useful as an outside reference point, but it is not
a competitive solution to the project's actual objective, which requires a good
overturn / no-overturn trade-off rather than a nearly always-revise policy.

## Comparison to Existing Reference Points

Using existing internal reports:

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| frozen prompt baseline | 0.3615 | 0.2718 | 0.5926 |
| frozen Qwen2.5-0.5B MC baseline | 0.7308 | 0.8835 | 0.1481 |
| NumPy CIPC baseline | 0.7846 | 0.7961 | 0.9259 |
| HF highrank v1 | 0.8462 | 0.8641 | 0.7778 |

## Takeaway

- This frozen external baseline is much stronger than the old frozen prompt
  baseline in overall accuracy and overturn behavior.
- But it is still clearly worse than the training-based lines because it fails
  to preserve maintain cases.
- The result reinforces the current project conclusion: raw frozen reasoning can
  find the overturn side, but the real difficulty is calibrated preservation.
