# ReviseQA Official Integration and Small OpenAI Pilot

## Dataset Integration

- Source: official `ReviseQA` GitHub release
- Processed transform: `reviseqa_incremental`
- Original examples: 1593
- Edited pairs: 11151
- `full_info`: 11151
- `incremental_no_overturn`: 5457
- `incremental_overturn_reasoning`: 5694
- Modification mix:
  `FLIP` 5810,
  `INVARIANT` 5341

## Prompting Update

- Added a generic-logic prompt family separate from the Belief-R
  suppression-task prompt family.
- Added a second small pilot (`v2`) after tightening the answer semantics so
  the model judges the exact conclusion sentence as written, including negated
  conclusions.

## Small-Pilot Readout

### v1

- Run:
  `runs/reviseqa_openai_small/`
- Outcome:
  the third data line ran end to end, but overturn performance was 0.00 for all
  systems.

### v2

- Run:
  `runs/reviseqa_openai_small_v2/`
- `full_info`:
  `raw_history` 0.25,
  `running_summary` 0.25,
  `structured_no_source` 0.25,
  `source_no_revision` 0.25,
  `source_revision` 0.50
- `incremental_no_overturn`:
  `raw_history` 0.50,
  `running_summary` 0.50,
  `structured_no_source` 0.50,
  `source_no_revision` 0.50,
  `source_revision` 0.25
- `incremental_overturn_reasoning`:
  all five systems = 0.00

## Interpretation

- The official-data integration problem is solved.
- The current blocker is prompt alignment, not infrastructure.
- The dataset appears materially harder than the current Belief-R and ATOMIC
  setups for this direct prompting scheme.
- The small traces suggest the model often defaults to uncertainty on edited
  logical conclusions, especially when the conclusion itself is conditional or
  negated.

## What This Means for the Project

- `ReviseQA` is now a valid third data line in the repo.
- It is not yet ready to be used as supportive evidence for the main method
  claim.
- It is currently best framed as a stress-test dataset that exposes a prompt /
  representation gap.

## Immediate Next Step

- Move `ReviseQA` to a more structured evaluation path:
  either theorem-style prompting, or a prompt that explicitly rewrites the
  conclusion and its negation before judging truth status.
