# Commitment-Control Evaluation

## Split Summary

### test

- n: `260`
- control_decision_accuracy: `0.2192`
- final_answer_accuracy: `0.2192`
- joint_accuracy: `0.2192`
- consistency_rate_gold: `0.2192`
- early_commitment_persistence: `0.9854`
- late_evidence_takeover: `0.0146`

## By Condition

| split | condition | n | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| test | full_info | 130 | 0.2231 | 0.2231 | 0.2231 | 0.2231 | 0.9806 | 0.0194 |
| test | incremental_no_overturn | 27 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| test | incremental_overturn_reasoning | 103 | 0.0097 | 0.0097 | 0.0097 | 0.0097 | 0.9903 | 0.0097 |
