# OpenAI Medium Pilot Report

## Objective

Move beyond smoke validation and run a first analyzable real-API pilot on
Belief-R using the same four-system comparison.

## Setup

- Provider: OpenAI
- Model requested: `gpt-5.4-mini`
- Config: `configs/api_pilot_belief_r_openai_medium.json`
- Sample size:
  10 examples each for `full_info`, `incremental_no_overturn`,
  `incremental_overturn_reasoning`
- Total evaluated records:
  120
- Total API calls:
  200
- Total prompt tokens:
  93,141

## Results

- `raw_history`
  - `full_info`: 0.40
  - `incremental_no_overturn`: 0.60
  - `incremental_overturn_reasoning`: 0.50
- `running_summary`
  - `full_info`: 0.20
  - `incremental_no_overturn`: 0.40
  - `incremental_overturn_reasoning`: 0.70
- `structured_no_source`
  - `full_info`: 0.20
  - `incremental_no_overturn`: 0.70
  - `incremental_overturn_reasoning`: 0.20
- `source_revision`
  - `full_info`: 0.20
  - `incremental_no_overturn`: 0.30
  - `incremental_overturn_reasoning`: 1.00

## Interpretation

- The source-aware revision condition produces the strongest belief-update
  behavior on the overturn subset.
- The expected trade-off remains: aggressive revision helps update-heavy cases
  but hurts belief-maintain cases.
- `full_info` remains weak across systems, which suggests prompt semantics for
  Belief-R's suppression-style judgments still need refinement.

## Output Files

- `runs/belief_r_api_pilot_openai_medium/summary.csv`
- `runs/belief_r_api_pilot_openai_medium/predictions.jsonl`
- `runs/belief_r_api_pilot_openai_medium/traces/`
- `paper_assets/belief_r_api_openai_medium_summary.csv`
