# Commitment-Control Evaluation

## Split Summary

### train

- n: `2050`
- control_decision_accuracy: `0.9873`
- final_answer_accuracy: `0.9854`
- joint_accuracy: `0.9824`
- consistency_rate_gold: `0.9854`
- early_commitment_persistence: `0.016`
- late_evidence_takeover: `0.984`

### dev

- n: `254`
- control_decision_accuracy: `0.7717`
- final_answer_accuracy: `0.7638`
- joint_accuracy: `0.7323`
- consistency_rate_gold: `0.7638`
- early_commitment_persistence: `0.1584`
- late_evidence_takeover: `0.8416`

### test

- n: `260`
- control_decision_accuracy: `0.8538`
- final_answer_accuracy: `0.8538`
- joint_accuracy: `0.8308`
- consistency_rate_gold: `0.8538`
- early_commitment_persistence: `0.068`
- late_evidence_takeover: `0.932`

## By Condition

| split | condition | n | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | full_info | 1025 | 0.9873 | 0.9854 | 0.9824 | 0.9854 | 0.016 | 0.984 |
| train | incremental_no_overturn | 211 | 0.9858 | 0.9905 | 0.9858 | 0.9905 | 0.0 | 0.0 |
| train | incremental_overturn_reasoning | 814 | 0.9877 | 0.984 | 0.9816 | 0.984 | 0.016 | 0.984 |
| dev | full_info | 127 | 0.7717 | 0.7638 | 0.7323 | 0.7638 | 0.1584 | 0.8416 |
| dev | incremental_no_overturn | 26 | 0.4615 | 0.4615 | 0.3846 | 0.4615 | 0.0 | 0.0 |
| dev | incremental_overturn_reasoning | 101 | 0.8515 | 0.8416 | 0.8218 | 0.8416 | 0.1584 | 0.8416 |
| test | full_info | 130 | 0.8538 | 0.8538 | 0.8308 | 0.8538 | 0.068 | 0.932 |
| test | incremental_no_overturn | 27 | 0.4815 | 0.5556 | 0.4815 | 0.5556 | 0.0 | 0.0 |
| test | incremental_overturn_reasoning | 103 | 0.9515 | 0.932 | 0.9223 | 0.932 | 0.068 | 0.932 |
