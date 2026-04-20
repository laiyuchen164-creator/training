# CIPC Qwen0.5B LoRA Gold-Gated Preserve v1

Date: 2026-04-20 UTC

## Purpose

This run tests a preserve-side-only repair after `gated_propagation_v1` showed
that predicted soft gates were not sufficient.

The hypothesis was:

- do not use the model's predicted preserve confidence as the gate
- apply preserve-side answer anchoring only on gold preserve examples
- avoid adding more replace-side pressure, since the aggressive frontier is
  already strong on overturn

## Configuration

Run:

- `runs/cipc_belief_r_qwen05b_lora_gold_gated_preserve_v1/`

Config:

- `configs/train_cipc_belief_r_qwen05b_lora_gold_gated_preserve_v1.json`

Base settings match `highrank_v1`:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- LoRA rank: `16`
- LoRA alpha: `32`
- max length: `256`
- epochs: `6`
- learning rate: `0.00012`
- answer loss weight: `1.0`

New setting:

- `gold_preserve_lambda = 0.2`

## Objective Change

The new objective adds preserve-side answer anchoring:

`L_gold_preserve = beta * I[gold_control = preserve] * CE(answer, early_answer)`

No replace-side propagation term or replace margin is added in this run.

## Results

### Test Summary

- n: `260`
- control decision accuracy: `0.8538`
- final answer accuracy: `0.8538`
- joint accuracy: `0.8231`
- early commitment persistence: `0.0874`
- late evidence takeover: `0.9126`

### By Condition

| condition | n | control acc | answer acc | joint acc |
|---|---:|---:|---:|---:|
| full_info | 130 | 0.8538 | 0.8538 | 0.8231 |
| incremental_no_overturn | 27 | 0.6667 | 0.6296 | 0.5926 |
| incremental_overturn_reasoning | 103 | 0.9029 | 0.9126 | 0.8835 |

## Comparison

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| `highrank_v1` | 0.8538 | 0.9029 | 0.6667 |
| `tradeoff_repair_v2` | 0.8000 | 0.7961 | 0.8148 |
| `gated_propagation_v1` | 0.8077 | 0.8641 | 0.5926 |
| `gold_gated_preserve_v1` | 0.8538 | 0.9126 | 0.6296 |

## Interpretation

`gold_gated_preserve_v1` is not the balanced point we wanted.

It matches `highrank_v1` on overall answer accuracy and slightly improves
overturn answer accuracy:

- overturn: `0.9029 -> 0.9126`

However, it worsens no-overturn answer accuracy:

- no-overturn: `0.6667 -> 0.6296`

So this run is better interpreted as another aggressive variant, not as a
preserve-side repair.

The result suggests that CE-to-early on gold preserve examples is still too
weak as a structural constraint. This is consistent with earlier findings:

- plain preserve CE did not protect no-overturn well
- pure preserve margin was the first design to improve no-overturn
- hybrid CE + margin recovered aggressive behavior but did not preserve the
  maintain-side gain

## Next Implication

The next balanced attempt should strengthen preserve-side structure more
directly, but avoid excessive replace pressure.

Most plausible next variants:

1. **Gold-gated preserve margin**
   - apply margin only on gold preserve examples
   - require early answer to outrank the strongest alternative
   - do not add replace-side margin

2. **Gold-gated preserve CE + margin**
   - preserve examples get CE-to-early plus a small preserve margin
   - replace examples keep only the base answer loss

3. **Preserve-upsample plus highrank objective**
   - revisit sampling, but only as a targeted preserve-side intervention
   - do not combine it with replace-side propagation changes initially

The immediate next candidate should be `gold_gated_preserve_margin_v1`.
