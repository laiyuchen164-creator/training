# CIPC vs Frozen Prompt Baseline

Split compared: `test`

## Overall

| method | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |
|---|---:|---:|---:|---:|---:|---:|
| CIPC | 0.6769 | 0.6462 | 0.6462 | 0.6462 | 0.4272 | 0.5728 |
| frozen_prompt_source_revision | 0.3615 | 0.3615 | 0.3615 | 0.3615 | 0.6893 | 0.3107 |

## By Condition

| condition | method | control_acc | answer_acc | joint_acc | early_persistence | late_takeover |
|---|---|---:|---:|---:|---:|---:|
| full_info | CIPC | 0.6769 | 0.6462 | 0.6462 | 0.4272 | 0.5728 |
| full_info | frozen_prompt_source_revision | 0.3846 | 0.3846 | 0.3846 | 0.6505 | 0.3495 |
| incremental_no_overturn | CIPC | 0.9259 | 0.9259 | 0.9259 | 0.0 | 0.0 |
| incremental_no_overturn | frozen_prompt_source_revision | 0.5926 | 0.5926 | 0.5926 | 0.0 | 0.0 |
| incremental_overturn_reasoning | CIPC | 0.6117 | 0.5728 | 0.5728 | 0.4272 | 0.5728 |
| incremental_overturn_reasoning | frozen_prompt_source_revision | 0.2718 | 0.2718 | 0.2718 | 0.7282 | 0.2718 |

## Interpretation

- The prompt baseline control score is a binary proxy derived from whether
  the final answer preserves the gold early commitment or switches away
  from it.
- This comparison is therefore valid for the current Belief-R binary
  `preserve/replace` setup, but it is not yet the final comparison design
  for a future 3-way `preserve/weaken/replace` setting.
