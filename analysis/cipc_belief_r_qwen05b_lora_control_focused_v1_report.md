# Commitment-Control Evaluation

## Split Summary

### train

- n: `2050`
- control_decision_accuracy: `0.9005`
- final_answer_accuracy: `0.9161`
- joint_accuracy: `0.8927`
- consistency_rate_gold: `0.9161`
- early_commitment_persistence: `0.1057`
- late_evidence_takeover: `0.8943`

### dev

- n: `254`
- control_decision_accuracy: `0.7402`
- final_answer_accuracy: `0.7402`
- joint_accuracy: `0.7323`
- consistency_rate_gold: `0.748`
- early_commitment_persistence: `0.2574`
- late_evidence_takeover: `0.7327`

### test

- n: `260`
- control_decision_accuracy: `0.7692`
- final_answer_accuracy: `0.7923`
- joint_accuracy: `0.7615`
- consistency_rate_gold: `0.7923`
- early_commitment_persistence: `0.233`
- late_evidence_takeover: `0.767`

## By Condition

| split | condition | n | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | full_info | 1025 | 0.9005 | 0.9161 | 0.8927 | 0.9161 | 0.1057 | 0.8943 |
| train | incremental_no_overturn | 211 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| train | incremental_overturn_reasoning | 814 | 0.8747 | 0.8943 | 0.8649 | 0.8943 | 0.1057 | 0.8943 |
| dev | full_info | 127 | 0.7402 | 0.7402 | 0.7323 | 0.748 | 0.2574 | 0.7327 |
| dev | incremental_no_overturn | 26 | 0.7692 | 0.7692 | 0.7308 | 0.7692 | 0.0 | 0.0 |
| dev | incremental_overturn_reasoning | 101 | 0.7327 | 0.7327 | 0.7327 | 0.7426 | 0.2574 | 0.7327 |
| test | full_info | 130 | 0.7692 | 0.7923 | 0.7615 | 0.7923 | 0.233 | 0.767 |
| test | incremental_no_overturn | 27 | 0.9259 | 0.8889 | 0.8889 | 0.8889 | 0.0 | 0.0 |
| test | incremental_overturn_reasoning | 103 | 0.7282 | 0.767 | 0.7282 | 0.767 | 0.233 | 0.767 |
