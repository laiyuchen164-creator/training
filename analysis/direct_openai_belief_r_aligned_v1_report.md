# Direct OpenAI Belief-R Aligned Baseline v1

Date: 2026-04-20 UTC

## Purpose

This run tests whether a more task-aligned direct OpenAI prompt improves over
the naive direct API baseline on the fixed Belief-R commitment-control test
split.

Unlike the naive prompt, this aligned prompt explicitly states that the task
uses suppression-task / belief-revision semantics and describes when option
`c` should be used.

## Setup

- Dataset: `data/processed/belief_r_commitment_control_test.jsonl`
- Size: `260`
- Model: OpenAI `gpt-5.4-mini`
- Run directory: `runs/direct_openai_gpt54mini_belief_r_aligned_v1/`
- Script: `analysis/evaluate_api_mc_baseline_aligned.py`

## Results

Overall:

- control decision accuracy: `0.2462`
- final answer accuracy: `0.2462`
- joint accuracy: `0.2462`
- early commitment persistence on replace cases: `0.9515`
- late evidence takeover on replace cases: `0.0485`

By condition:

| condition | n | answer acc |
|---|---:|---:|
| full_info | 130 | 0.2462 |
| incremental_no_overturn | 27 | 1.0000 |
| incremental_overturn_reasoning | 103 | 0.0485 |

Prediction distribution:

- `a`: `137`
- `b`: `113`
- `c`: `10`

On `incremental_overturn_reasoning`, the model predicted:

- `a`: `56`
- `b`: `42`
- `c`: `5`

## Comparison

| method | overall | overturn | no-overturn |
|---|---:|---:|---:|
| naive direct OpenAI `gpt-5.4-mini` | 0.2077 | 0.0000 | 1.0000 |
| aligned direct OpenAI `gpt-5.4-mini` | 0.2462 | 0.0485 | 1.0000 |
| CIPC `highrank_v1` | 0.8538 | 0.9029 | 0.6667 |
| CIPC `tradeoff_repair_v2` | 0.8000 | 0.7961 | 0.8148 |

## Interpretation

Prompt alignment helps slightly but does not solve the task.

The aligned prompt increases direct OpenAI performance from `0.2077` to
`0.2462` overall and from `0.0000` to `0.0485` on overturn cases. However, the
model still overwhelmingly preserves the early commitment and predicts option
`c` only `10 / 260` times.

This supports a more precise claim:

- the naive direct prompt was under-aligned
- a more aligned direct prompt improves the baseline a little
- but direct prompting still does not learn stable commitment-control behavior
  on Belief-R

The trained CIPC checkpoints remain far stronger on the actual preserve-vs-
revise control task.
