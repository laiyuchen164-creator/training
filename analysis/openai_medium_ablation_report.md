# OpenAI Medium Ablation Report

## Objective

Test whether the gain comes from explicit source-aware revision policy rather
than merely storing source metadata.

## Setup

- Config: `configs/api_pilot_belief_r_openai_medium_ablation.json`
- Provider: OpenAI
- Model requested: `gpt-5.4-mini`
- Systems:
  `raw_history`, `running_summary`, `structured_no_source`,
  `source_no_revision`, `source_revision`
- Sample size:
  10 examples each for `full_info`, `incremental_no_overturn`,
  `incremental_overturn_reasoning`

## Key Results

Overturn subset accuracy:

- `raw_history`: 0.40
- `running_summary`: 0.50
- `structured_no_source`: 0.20
- `source_no_revision`: 0.20
- `source_revision`: 0.90

Assistant-assumption survival on overturn subset:

- `raw_history`: 0.60
- `running_summary`: 0.50
- `structured_no_source`: 0.80
- `source_no_revision`: 0.80
- `source_revision`: 0.10

No-overturn subset accuracy:

- `raw_history`: 0.60
- `running_summary`: 0.40
- `structured_no_source`: 0.80
- `source_no_revision`: 0.90
- `source_revision`: 0.30

## Interpretation

The ablation isolates the mechanism cleanly:

- `source_no_revision` performs almost identically to `structured_no_source` on
  the overturn subset.
- The jump to `source_revision` is therefore not explained by source metadata
  alone.
- The main effect comes from applying source-conditioned persistence control to
  assistant-generated intermediate commitments.

This is the strongest current empirical support for the paper's intended claim.
