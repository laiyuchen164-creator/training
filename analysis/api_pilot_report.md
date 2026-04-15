# OpenAI API Pilot Report

## Objective

Validate that the Belief-R pilot can run with real model calls instead of the
heuristic backend while preserving the same experiment interface and outputs.

## Setup

- Provider: OpenAI
- Config: `configs/api_pilot_belief_r_openai_small.json`
- Requested model: `gpt-5.4-mini`
- Fallbacks configured:
  `gpt-5-mini`, `gpt-4.1-mini`
- Sample size:
  2 examples per condition, 6 total
- Systems:
  `raw_history`, `running_summary`, `structured_no_source`,
  `source_revision`

## Outcome

The run completed successfully and wrote:

- `runs/belief_r_api_pilot_openai_small/predictions.jsonl`
- `runs/belief_r_api_pilot_openai_small/summary.csv`
- `runs/belief_r_api_pilot_openai_small/traces/`

The traces confirm:

- real API responses were received
- the model returned JSON labels that parsed cleanly
- usage tokens were logged
- turn-by-turn memory state and final ledger snapshots were saved

## Interpretation

This run is an infrastructure validation step, not a paper-quality result. The
sample is too small to interpret as evidence. The important result is that the
project can now execute real API-based experiments under the same evaluation
stack used by the heuristic pilot.
