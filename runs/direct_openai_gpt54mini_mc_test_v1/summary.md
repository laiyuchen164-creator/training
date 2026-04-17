# Commitment-Control Evaluation

## Split Summary

### test

- n: `260`
- control_decision_accuracy: `0.2077`
- final_answer_accuracy: `0.2077`
- joint_accuracy: `0.2077`
- consistency_rate_gold: `0.2077`
- early_commitment_persistence: `1.0`
- late_evidence_takeover: `0.0`

## By Condition

| split | condition | n | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| test | full_info | 130 | 0.2077 | 0.2077 | 0.2077 | 0.2077 | 1.0 | 0.0 |
| test | incremental_no_overturn | 27 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| test | incremental_overturn_reasoning | 103 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
