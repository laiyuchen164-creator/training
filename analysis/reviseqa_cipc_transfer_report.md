# ReviseQA Full-Set Transfer Evaluation for Trained CIPC

Date: 2026-04-20 UTC

## Purpose

This report evaluates the trained CIPC HF/LoRA checkpoints on the full
ReviseQA incremental dataset, after converting ReviseQA into the same
commitment-control schema used by the Belief-R CIPC pipeline.

This is different from the earlier prompt-based `source_revision` line. The
models evaluated here are trained CIPC checkpoints.

## Data

- Source dataset: official ReviseQA natural-language edits
- Converted file: `data/processed/reviseqa_commitment_control_full.jsonl`
- Total examples: `22302`
- Conditions:
  - `full_info`: `11151`
  - `incremental_no_overturn`: `5457`
  - `incremental_overturn_reasoning`: `5694`
- Control labels:
  - `replace`: `11388`
  - `preserve`: `10914`
- Final answer labels:
  - `a`: `11312`
  - `b`: `10990`
  - `c`: `0`

## Evaluated CIPC Checkpoints

### `highrank_v1`

Checkpoint:

- `runs/cipc_belief_r_qwen05b_lora_highrank_v1/`

Run output:

- `runs/reviseqa_cipc_highrank_v1_full_eval/`

Overall:

- n: `22302`
- control decision accuracy: `0.4899`
- final answer accuracy: `0.4448`
- joint accuracy: `0.2193`
- early commitment persistence on replace cases: `0.4257`
- late evidence takeover on replace cases: `0.4468`

By condition:

| condition | n | control acc | answer acc | joint acc | early persistence | late takeover |
|---|---:|---:|---:|---:|---:|---:|
| full_info | 11151 | 0.4899 | 0.4448 | 0.2193 | 0.4257 | 0.4468 |
| incremental_no_overturn | 5457 | 0.7643 | 0.4427 | 0.3517 | 0.0000 | 0.0000 |
| incremental_overturn_reasoning | 5694 | 0.2269 | 0.4468 | 0.0924 | 0.4257 | 0.4468 |

### `preserve_hybrid_v1`

Checkpoint:

- `runs/cipc_belief_r_qwen05b_lora_preserve_hybrid_conditional_consistency_v1/`

Run output:

- `runs/reviseqa_cipc_preserve_hybrid_v1_full_eval/`

Overall:

- n: `22302`
- control decision accuracy: `0.4895`
- final answer accuracy: `0.4220`
- joint accuracy: `0.2120`
- early commitment persistence on replace cases: `0.4139`
- late evidence takeover on replace cases: `0.4204`

By condition:

| condition | n | control acc | answer acc | joint acc | early persistence | late takeover |
|---|---:|---:|---:|---:|---:|---:|
| full_info | 11151 | 0.4895 | 0.4220 | 0.2120 | 0.4139 | 0.4204 |
| incremental_no_overturn | 5457 | 0.7706 | 0.4237 | 0.3405 | 0.0000 | 0.0000 |
| incremental_overturn_reasoning | 5694 | 0.2201 | 0.4204 | 0.0889 | 0.4139 | 0.4204 |

## Interpretation

On full ReviseQA, `highrank_v1` is the better CIPC transfer point among the two
tested checkpoints. It has higher overall and condition-level final-answer
accuracy than `preserve_hybrid_v1`.

The absolute numbers are much lower than the Belief-R in-domain results. This
is expected to be a difficult cross-dataset transfer setting:

- CIPC was trained on Belief-R commitment-control data, not ReviseQA.
- ReviseQA has more formal logical edits and a different answer-label
  distribution.
- The converted ReviseQA full set has no gold `c` final answers, unlike
  Belief-R where uncertainty is central.
- The CIPC input renderer truncates to the Belief-R training max length
  (`max_length = 256` for `highrank_v1`), while ReviseQA examples often contain
  much longer premise sets.

## Current Recommendation

For comparison against external GPT API baselines, use:

1. `highrank_v1` as the main trained CIPC checkpoint.
2. `preserve_hybrid_v1` only as an additional newer consistency variant.

The direct GPT API full-set baseline is still pending. It should be run with a
fresh API key supplied through `OPENAI_API_KEY`, because the previously pasted
key appeared in chat and should be treated as exposed.
