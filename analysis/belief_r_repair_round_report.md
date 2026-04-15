# Belief-R Repair Round Report

## Scope

This round followed the debugging order in
`codex_debug_execution_order.docx`:

- P0: repair Belief-R full-info prompting
- P1: add a conservative revision gate
- do not broaden the method or add new execution layers

## What Changed

### 1. Full-info prompt repair

- Added a Belief-R-specific system prompt:
  `prompts/llm_answer_system_belief_r_v2.txt`
- Added a dedicated full-info prompt with mini calibration examples:
  `prompts/llm_full_info_belief_r_v2.txt`
- Added explicit structured fields:
  `premise_role` and `relation_to_prior`
- Added a confusion-analysis script:
  `analysis/analyze_belief_r_full_info_confusions.py`

### 2. Conservative revision gate

- Added `relation_to_prior` parsing to the API client path
- Logged parsed structured outputs in turn traces
- Added gated source revision logic so only
  `contradict` / `replace` can aggressively demote the prior commitment
- Added the gate unit test:
  `tests/test_systems.py`

## Error Analysis Before Repair

The previous Belief-R ablation showed a strong collapse on full-info examples.

From `analysis/belief_r_full_info_confusion_report.md`:

- full-info predictions:
  50 total
- confusion counts:
  `gold b -> pred b = 5`
  `gold c -> pred a = 14`
  `gold c -> pred b = 28`
  `gold c -> pred c = 3`

Interpretation:

- The model was systematically misreading `gold = c` cases as either preserved
  ponens or preserved tollens.
- The dominant mistake was treating contextual enabling conditions as
  independent alternative pathways.

## Test 1: Full-Info Repair

- Config:
  `configs/belief_r_openai_full_info_repair_v1.json`
- Run:
  `runs/belief_r_openai_full_info_repair_v1/`
- Sample:
  24 full-info examples

Result:

- `source_revision` full-info accuracy:
  `0.375`

Additional readout:

- gold labels:
  `c = 20`, `a = 3`, `b = 1`
- predicted labels:
  `a = 12`, `c = 9`, `b = 3`

Interpretation:

- This is materially better than the old `source_revision` full-info result
  (`0.10` in the previous medium ablation).
- The prompt is no longer completely collapsing to the old answer modes.
- However, the model still over-predicts `alternative_pathway`, so this repair
  is an improvement, not a final solution.

## Test 2: Gated Belief-R Ablation

- Config:
  `configs/belief_r_openai_gated_ablation_v1.json`
- Run:
  `runs/belief_r_openai_gated_ablation_v1/`
- Sample:
  8 examples per condition, 5 systems

Summary:

- `full_info`
  - all five systems: `0.75`
- `incremental_no_overturn`
  - `raw_history`: `0.625`
  - `running_summary`: `0.50`
  - `structured_no_source`: `0.50`
  - `source_no_revision`: `0.75`
  - `source_revision`: `0.50`
- `incremental_overturn_reasoning`
  - `raw_history`: `0.25`
  - `running_summary`: `0.50`
  - `structured_no_source`: `0.125`
  - `source_no_revision`: `0.25`
  - `source_revision`: `0.625`

Comparison to the older medium ablation:

- `source_revision full_info`:
  `0.10 -> 0.75`
- `source_revision incremental_no_overturn`:
  `0.30 -> 0.50`
- `source_revision incremental_overturn_reasoning`:
  `0.90 -> 0.625`
- `source_revision assistant_assumption_survival` on overturn:
  `0.10 -> 0.375`

Interpretation:

- The reference condition problem is substantially improved.
- The over-revision problem is partially improved:
  `source_revision` no-overturn accuracy rose materially.
- The main mechanism signal still survives:
  `source_revision` remains clearly above `structured_no_source` and
  `source_no_revision` on overturn recovery.
- But the repair is not free:
  overturn performance dropped from the earlier peak, so this is a
  trade-off-reduction step, not a dominant win on every metric.

## Gate Behavior Notes

In this small rerun:

- `source_revision` produced:
  `relation_to_prior = replace` on 11 incremental cases
  `relation_to_prior = confirm` on 5 incremental cases
- explicit gate overrides did not fire in this sample:
  `gate_preserved_prior = 0`

Interpretation:

- The new prompt mostly changed the model's own relation judgment before the
  gate had to force an override.
- The gate logic is installed and visible in traces, but this sample suggests
  the main gains came from semantic alignment plus the relation-classification
  step, not from frequent hard overrides.

## Bottom Line

- P0 is largely successful:
  Belief-R full-info is no longer obviously broken.
- P1 is partially successful:
  no-overturn conservatism improved, while overturn strength remains above the
  key baselines but below the earlier peak.
- ReviseQA is still unresolved and should remain a stress-test line for now.
