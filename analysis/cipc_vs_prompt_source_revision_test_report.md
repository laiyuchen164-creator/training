# CIPC vs Frozen Prompt Baseline

Split compared: `test`

## Overall

| method | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |
|---|---:|---:|---:|---:|---:|---:|
| CIPC | 0.8923 | 0.7846 | 0.7808 | 0.7846 | 0.2524 | 0.7476 |
| frozen_prompt_source_revision | 0.3615 | 0.3615 | 0.3615 | 0.3615 | 0.6893 | 0.3107 |

## By Condition

| condition | method | control_acc | answer_acc | joint_acc | early_persistence | late_takeover |
|---|---|---:|---:|---:|---:|---:|
| full_info | CIPC | 0.8154 | 0.7462 | 0.7385 | 0.301 | 0.699 |
| full_info | frozen_prompt_source_revision | 0.3846 | 0.3846 | 0.3846 | 0.6505 | 0.3495 |
| incremental_no_overturn | CIPC | 0.9259 | 0.9259 | 0.9259 | 0.0 | 0.0 |
| incremental_no_overturn | frozen_prompt_source_revision | 0.5926 | 0.5926 | 0.5926 | 0.0 | 0.0 |
| incremental_overturn_reasoning | CIPC | 0.9806 | 0.7961 | 0.7961 | 0.2039 | 0.7961 |
| incremental_overturn_reasoning | frozen_prompt_source_revision | 0.2718 | 0.2718 | 0.2718 | 0.7282 | 0.2718 |

## Interpretation

- The prompt baseline control score is a binary proxy derived from whether
  the final answer preserves the gold early commitment or switches away
  from it.
- This comparison is therefore valid for the current Belief-R binary
  `preserve/replace` setup, but it is not yet the final comparison design
  for a future 3-way `preserve/weaken/replace` setting.
