# Claim Sheet

## Claim 1

Source metadata alone does not explain the main gains on implicit revision
 cases.

Supported by:

- `belief_r_api_pilot_openai_medium_ablation`
- On `incremental_overturn_reasoning`,
  `structured_no_source = 0.20`,
  `source_no_revision = 0.20`,
  `source_revision = 0.90`

Interpretation:

- The main gain requires source-conditioned persistence control, not just source
  tagging.

## Claim 2

The method trades belief-update strength against belief-maintain conservatism.

Supported by:

- `belief_r_api_pilot_openai_medium_ablation`
- On `incremental_no_overturn`,
  `source_revision = 0.30`
- On `incremental_overturn_reasoning`,
  `source_revision = 0.90`

Interpretation:

- The method should be framed as a targeted intervention for stale-assumption
  persistence rather than a universally better reasoning policy.

## Claim 3

The proposed mechanism matters more when revision cues are implicit than when
they are explicitly stated.

Supported by:

- `belief_r_api_pilot_openai_medium_ablation`
- `atomic_explicit_openai_small`

Interpretation:

- On Belief-R, which relies on suppression-style implicit revision cues, system
  differences are large.
- On the ATOMIC explicit transform, where the correction is directly written
  into the premise, system differences largely collapse.

## Claim 4

The repository now supports reproducible end-to-end experiments across three
dataset pipelines with the same evaluation stack.

Supported by:

- `belief_r` transform and pilots
- `atomic_explicit_revision` transform and pilots
- `reviseqa_incremental` official transform and pilots
- Shared outputs:
  predictions, traces, summaries, progress reports, and paper assets
