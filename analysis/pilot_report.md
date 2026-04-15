# Belief-R Pilot Report

## Objective

Run the first end-to-end pilot for the EMNLP project on a real benchmark and
check whether source-aware revision reduces wrong-turn persistence.

## Setup

- Dataset: Belief-R strong paired subset
- Conditions:
  `full_info`, `incremental_no_overturn`,
  `incremental_overturn_reasoning`
- Sample size:
  40 examples per condition, 120 total
- Systems:
  `raw_history`, `running_summary`, `structured_no_source`,
  `source_revision`

## Key Results

- `full_info` accuracy is 0.75 for all systems.
- `incremental_no_overturn` favors conservative systems:
  `raw_history` 0.875 and `running_summary` 0.800.
- `incremental_overturn_reasoning` favors revision-aware systems:
  `raw_history` 0.525,
  `running_summary` 0.600,
  `structured_no_source` 0.650,
  `source_revision` 0.700.
- Mechanism metrics move in the expected direction on overturn cases:
  `stale_belief_persistence` and `assistant_assumption_survival`
  both fall from 0.475 in `raw_history` to 0.300 in `source_revision`.

## Interpretation

The pipeline is now capable of producing the exact artifacts the paper needs at
pilot stage: transformed data, per-example traces, final predictions, aggregate
tables, and mechanism metrics. The current signal is aligned with the paper
claim, but the backend is heuristic. The next milestone is to keep the same
interfaces and replace the backend with an API-driven LLM implementation.

## Output Files

- Run summary: `runs/belief_r_pilot_v1/summary.csv`
- Prediction log: `runs/belief_r_pilot_v1/predictions.jsonl`
- Traces: `runs/belief_r_pilot_v1/traces/`
- Paper table copy: `paper_assets/belief_r_pilot_summary.csv`
