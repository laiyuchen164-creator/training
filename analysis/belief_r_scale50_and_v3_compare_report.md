# Belief-R Scale-50 Validation and Minimal Prompt-Tweak Comparison

## 1. Frozen Repaired+Gated Validation

- Stable config:
  `configs/belief_r_openai_gated_stable_v1_scale50.json`
- Run:
  `runs/belief_r_openai_gated_stable_v1_scale50/`
- Sample size:
  50 examples per condition
  (`full_info`, `incremental_no_overturn`, `incremental_overturn_reasoning`)
- Systems:
  `raw_history`, `running_summary`, `structured_no_source`,
  `source_no_revision`, `source_revision`

### Main readout

- `full_info`
  - `raw_history`: `0.52`
  - `running_summary`: `0.50`
  - `structured_no_source`: `0.50`
  - `source_no_revision`: `0.52`
  - `source_revision`: `0.50`
- `incremental_no_overturn`
  - `raw_history`: `0.82`
  - `running_summary`: `0.60`
  - `structured_no_source`: `0.88`
  - `source_no_revision`: `0.86`
  - `source_revision`: `0.60`
- `incremental_overturn_reasoning`
  - `raw_history`: `0.22`
  - `running_summary`: `0.46`
  - `structured_no_source`: `0.12`
  - `source_no_revision`: `0.18`
  - `source_revision`: `0.40`

### Interpretation

- The repaired+gated Belief-R line survives scale-up.
- `source_revision` remains clearly above
  `structured_no_source` and `source_no_revision` on overturn.
- The update-versus-maintain trade-off also remains:
  `source_revision` is still much weaker than the conservative baselines on
  no-overturn.
- `full_info` is no longer broken, but it is not yet strong enough to be treated
  as a solved reference condition.

## 2. Expanded Error Analysis

Source:

- `analysis/belief_r_openai_gated_stable_v1_scale50_analysis.md`
- `analysis/belief_r_openai_gated_stable_v1_scale50_error_summary.csv`

### Source-Revision no-overturn failures

- Total failures: `20`
- Bucket counts:
  - `wrong_relation_to_prior`: `20`

Interpretation:

- The dominant remaining problem is not generic parsing instability.
- The model is still over-calling `replace` on cases that should preserve the
  earlier commitment.

### Source-Revision missed overturns

- Total missed overturns: `30`
- Bucket counts:
  - `wrong_relation_to_prior`: `17`
  - `label_semantic_confusion`: `13`

Interpretation:

- The remaining overturn loss comes from two sources:
  1. relation-to-prior is still too conservative on many true revision cases
  2. even when the relation is aggressive enough, Belief-R label semantics are
     not always mapped correctly

## 3. Minimal Prompt-Tweak Test

Following the scale-50 diagnosis, one minimal prompt tweak was tested only on
the `source_revision` follow-up prompt.

- Frozen baseline:
  `configs/belief_r_source_revision_holdout_v2.json`
- Minimal tweak:
  `configs/belief_r_source_revision_holdout_v3.json`
- Same held-out seed:
  `41`
- Same held-out sample size:
  15 per condition

### Held-out comparison

- `full_info`
  - `v2`: `0.60`
  - `v3`: `0.60`
- `incremental_no_overturn`
  - `v2`: `0.80`
  - `v3`: `1.00`
- `incremental_overturn_reasoning`
  - `v2`: `0.4667`
  - `v3`: `0.00`

Mechanism metrics on overturn:

- `assistant_assumption_survival`
  - `v2`: `0.5333`
  - `v3`: `1.00`
- `correction_uptake`
  - `v2`: `0.4667`
  - `v3`: `0.00`

### Interpretation

- The minimal prompt tweak improved conservatism too much.
- It fixed no-overturn on this held-out slice, but it destroyed overturn
  recovery.
- The tweak should therefore be rejected as the new default.

## 4. Decision

- Keep the frozen repaired+gated configuration as the main Belief-R candidate.
- Do not promote the `v3` prompt tweak.
- The next useful move should not be a broad rewrite.
- If another prompt tweak is attempted later, it should target the specific
  subset of no-overturn false `replace` cases without pushing overturn cases
  into blanket `confirm` behavior.
