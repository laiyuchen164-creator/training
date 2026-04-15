# ATOMIC Explicit Revision Report

## Objective

Create a second dataset path that does not depend on Belief-R alone and test the
same experiment stack on a more explicit revision task.

## Dataset

- Source: official ATOMIC release
- Transform type: synthetic explicit revision
- Seed examples used: 300
- Derived records:
  - `full_info`: 600
  - `incremental_no_overturn`: 300
  - `incremental_overturn_reasoning`: 300

Each seed produces:

- a maintain example with an explicit alternative pathway to the same outcome
- an update example with an explicit correction that makes the earlier rule more
  restrictive

## OpenAI Small Pilot

- Config: `configs/atomic_explicit_openai_small.json`
- Sample size: 4 examples per condition
- Systems:
  `raw_history`, `running_summary`, `structured_no_source`,
  `source_no_revision`, `source_revision`

## Main Observation

On the explicit synthetic dataset, all systems reached 1.00 accuracy on both
incremental conditions in the small pilot. The main differences only remained on
the `full_info` subset.

## Interpretation

This is a useful complement to Belief-R:

- On Belief-R, revision is often implicit and commonsense-heavy, and the
  source-aware persistence policy matters a lot.
- On the ATOMIC explicit transform, the correction signal is directly written
  into the premise, so the task becomes much easier and the system differences
  mostly collapse.

That contrast helps delimit the paper's mechanism claim. The proposed method is
most useful when stale intermediate commitments persist despite weak or implicit
revision cues.
