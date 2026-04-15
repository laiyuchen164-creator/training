# Belief-R Final Targeted Fix Report

## Scope

This round follows the constraints in
`C:/Users/laiyu/Downloads/belief_r_final_targeted_fix_instructions.docx`.
It does not touch ReviseQA, ATOMIC, the general pipeline, or the baseline
paper candidate. The only intervention is a targeted edit to the
`source_revision` follow-up prompt.

Targeted residual problems:

- false `replace` on `incremental_no_overturn`
- label-semantic confusion on missed `incremental_overturn_reasoning`

## What Changed

Frozen reference point:

- `configs/belief_r_source_revision_holdout_v2.json`

Pattern mining artifacts:

- `analysis/belief_r_false_replace_pattern_memo.md`
- `analysis/belief_r_missed_overturn_pattern_memo.md`

Single prompt intervention:

- added `prompts/llm_followup_source_revision_belief_r_v4.txt`
- added `configs/belief_r_source_revision_holdout_v4.json`
- extended `src/prompting.py` to route
  `belief_r_source_revision_prompt_version = "v4"`

The `v4` prompt tries to stop default `replace` decisions for added detail,
supporting actors, separate triggers, and background conditions. It asks for
`replace` only when the earlier commitment can no longer stand exactly as
before.

## Validation Setup

Held-out validation uses the same slice and seed as the previous comparison:

- seed: `41`
- systems: `source_revision` only
- conditions: `full_info`, `incremental_no_overturn`,
  `incremental_overturn_reasoning`
- sample size: `15` per condition

Reference runs:

- frozen baseline: `runs/belief_r_source_revision_holdout_v2/`
- rejected conservative tweak: `runs/belief_r_source_revision_holdout_v3/`
- new targeted fix: `runs/belief_r_source_revision_holdout_v4/`

## Result

| run | full_info | no_overturn | overturn | assistant_assumption_survival | correction_uptake |
| --- | --- | --- | --- | --- | --- |
| `v2` | `0.60` | `0.80` | `0.4667` | `0.5333` | `0.4667` |
| `v3` | `0.60` | `1.00` | `0.00` | `1.00` | `0.00` |
| `v4` | `0.60` | `0.8667` | `0.00` | `1.00` | `0.00` |

Direct readout:

- `v4` improves `incremental_no_overturn` over `v2`:
  `0.80 -> 0.8667`
- `v4` does not improve `full_info`:
  `0.60 -> 0.60`
- `v4` collapses `incremental_overturn_reasoning`:
  `0.4667 -> 0.00`

## Error Structure

### No-overturn

`v4` reduces but does not eliminate the original false-`replace` problem.

- `v2` relation counts on `incremental_no_overturn`:
  `replace = 10`, `confirm = 4`, `unrelated = 1`
- `v4` relation counts on `incremental_no_overturn`:
  `replace = 6`, `contradict = 3`, `confirm = 4`, `unrelated = 2`

Wrong predictions:

- `v2`: `3` errors, all `replace`
- `v4`: `2` errors, both still `replace`

So the targeted prompt repair helps on the exact problem it aimed at, but only
partially.

### Overturn

`v4` fails the acceptance criterion because it destroys recovery on the
incremental overturn slice.

`v2` on `incremental_overturn_reasoning`:

- `confirm + alternative_pathway + changed_answer = False`: `8`
- `replace + extra_requirement + changed_answer = True`: `7`

`v4` on `incremental_overturn_reasoning`:

- `confirm + alternative_pathway + changed_answer = False`: `9`
- `replace + extra_requirement + changed_answer = False`: `3`
- `confirm + extra_requirement + changed_answer = False`: `1`
- `elaborate + alternative_pathway + changed_answer = False`: `1`
- `replace + contradiction + changed_answer = False`: `1`

The key failure is not only relation labeling. Even when `v4` emits
`replace`, it no longer actually revises the answer on the held-out overturn
slice. That produces:

- `assistant_assumption_survival = 1.00`
- `correction_uptake = 0.00`

This is effectively the same failure mode as `v3`, but with a slightly weaker
conservative bias.

## Decision

`v4` is rejected.

Reason:

- it improves `incremental_no_overturn` only modestly
- it causes a material collapse on `incremental_overturn_reasoning`
- it therefore fails the acceptance rule in the instruction document

The frozen repaired+gated baseline remains the default paper candidate:

- large-scale reference:
  `runs/belief_r_openai_gated_stable_v1_scale50/`
- held-out prompt reference:
  `runs/belief_r_source_revision_holdout_v2/`

## Practical Takeaway

The remaining Belief-R problem is no longer "make `replace` rarer" in general.
That direction has now failed twice (`v3`, `v4`) on the same held-out slice.
The next successful fix, if attempted, has to preserve explicit revision on
true extra-requirement overturn cases while reducing false `replace` only in
the narrow tollens-style no-overturn subgroup.
