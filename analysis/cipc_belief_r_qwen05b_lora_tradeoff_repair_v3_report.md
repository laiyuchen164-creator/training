# Commitment-Control Evaluation

## Split Summary

### train

- n: `2050`
- control_decision_accuracy: `0.9678`
- final_answer_accuracy: `0.9639`
- joint_accuracy: `0.959`
- consistency_rate_gold: `0.9639`
- early_commitment_persistence: `0.043`
- late_evidence_takeover: `0.957`

### dev

- n: `254`
- control_decision_accuracy: `0.7638`
- final_answer_accuracy: `0.7559`
- joint_accuracy: `0.7165`
- consistency_rate_gold: `0.7559`
- early_commitment_persistence: `0.1683`
- late_evidence_takeover: `0.8317`

### test

- n: `260`
- control_decision_accuracy: `0.8308`
- final_answer_accuracy: `0.8154`
- joint_accuracy: `0.8`
- consistency_rate_gold: `0.8154`
- early_commitment_persistence: `0.1165`
- late_evidence_takeover: `0.8835`

## By Condition

| split | condition | n | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | full_info | 1025 | 0.9678 | 0.9639 | 0.959 | 0.9639 | 0.043 | 0.957 |
| train | incremental_no_overturn | 211 | 0.981 | 0.9905 | 0.981 | 0.9905 | 0.0 | 0.0 |
| train | incremental_overturn_reasoning | 814 | 0.9644 | 0.957 | 0.9533 | 0.957 | 0.043 | 0.957 |
| dev | full_info | 127 | 0.7638 | 0.7559 | 0.7165 | 0.7559 | 0.1683 | 0.8317 |
| dev | incremental_no_overturn | 26 | 0.4615 | 0.4615 | 0.3846 | 0.4615 | 0.0 | 0.0 |
| dev | incremental_overturn_reasoning | 101 | 0.8416 | 0.8317 | 0.802 | 0.8317 | 0.1683 | 0.8317 |
| test | full_info | 130 | 0.8308 | 0.8154 | 0.8 | 0.8154 | 0.1165 | 0.8835 |
| test | incremental_no_overturn | 27 | 0.5556 | 0.5556 | 0.5556 | 0.5556 | 0.0 | 0.0 |
| test | incremental_overturn_reasoning | 103 | 0.9029 | 0.8835 | 0.8641 | 0.8835 | 0.1165 | 0.8835 |
