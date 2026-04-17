# Commitment-Control Evaluation

## Split Summary

### train

- n: `2050`
- control_decision_accuracy: `0.9317`
- final_answer_accuracy: `0.9385`
- joint_accuracy: `0.9239`
- consistency_rate_gold: `0.9385`
- early_commitment_persistence: `0.0762`
- late_evidence_takeover: `0.9238`

### dev

- n: `254`
- control_decision_accuracy: `0.7638`
- final_answer_accuracy: `0.7402`
- joint_accuracy: `0.7087`
- consistency_rate_gold: `0.7402`
- early_commitment_persistence: `0.2178`
- late_evidence_takeover: `0.7822`

### test

- n: `260`
- control_decision_accuracy: `0.7846`
- final_answer_accuracy: `0.8`
- joint_accuracy: `0.7462`
- consistency_rate_gold: `0.8`
- early_commitment_persistence: `0.2039`
- late_evidence_takeover: `0.7961`

## By Condition

| split | condition | n | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | full_info | 1025 | 0.9317 | 0.9385 | 0.9239 | 0.9385 | 0.0762 | 0.9238 |
| train | incremental_no_overturn | 211 | 1.0 | 0.9953 | 0.9953 | 0.9953 | 0.0 | 0.0 |
| train | incremental_overturn_reasoning | 814 | 0.914 | 0.9238 | 0.9054 | 0.9238 | 0.0762 | 0.9238 |
| dev | full_info | 127 | 0.7638 | 0.7402 | 0.7087 | 0.7402 | 0.2178 | 0.7822 |
| dev | incremental_no_overturn | 26 | 0.6538 | 0.5769 | 0.5385 | 0.5769 | 0.0 | 0.0 |
| dev | incremental_overturn_reasoning | 101 | 0.7921 | 0.7822 | 0.7525 | 0.7822 | 0.2178 | 0.7822 |
| test | full_info | 130 | 0.7846 | 0.8 | 0.7462 | 0.8 | 0.2039 | 0.7961 |
| test | incremental_no_overturn | 27 | 0.7407 | 0.8148 | 0.7037 | 0.8148 | 0.0 | 0.0 |
| test | incremental_overturn_reasoning | 103 | 0.7961 | 0.7961 | 0.7573 | 0.7961 | 0.2039 | 0.7961 |
