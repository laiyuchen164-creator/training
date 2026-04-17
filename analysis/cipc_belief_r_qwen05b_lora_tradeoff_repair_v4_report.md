# Commitment-Control Evaluation

## Split Summary

### train

- n: `2050`
- control_decision_accuracy: `0.9541`
- final_answer_accuracy: `0.9385`
- joint_accuracy: `0.9307`
- consistency_rate_gold: `0.9385`
- early_commitment_persistence: `0.0762`
- late_evidence_takeover: `0.9238`

### dev

- n: `254`
- control_decision_accuracy: `0.7874`
- final_answer_accuracy: `0.7795`
- joint_accuracy: `0.748`
- consistency_rate_gold: `0.7795`
- early_commitment_persistence: `0.198`
- late_evidence_takeover: `0.802`

### test

- n: `260`
- control_decision_accuracy: `0.8231`
- final_answer_accuracy: `0.7846`
- joint_accuracy: `0.7769`
- consistency_rate_gold: `0.7846`
- early_commitment_persistence: `0.1942`
- late_evidence_takeover: `0.8058`

## By Condition

| split | condition | n | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | full_info | 1025 | 0.9541 | 0.9385 | 0.9307 | 0.9385 | 0.0762 | 0.9238 |
| train | incremental_no_overturn | 211 | 0.9905 | 0.9953 | 0.9905 | 0.9953 | 0.0 | 0.0 |
| train | incremental_overturn_reasoning | 814 | 0.9447 | 0.9238 | 0.9152 | 0.9238 | 0.0762 | 0.9238 |
| dev | full_info | 127 | 0.7874 | 0.7795 | 0.748 | 0.7795 | 0.198 | 0.802 |
| dev | incremental_no_overturn | 26 | 0.6154 | 0.6923 | 0.5769 | 0.6923 | 0.0 | 0.0 |
| dev | incremental_overturn_reasoning | 101 | 0.8317 | 0.802 | 0.7921 | 0.802 | 0.198 | 0.802 |
| test | full_info | 130 | 0.8231 | 0.7846 | 0.7769 | 0.7846 | 0.1942 | 0.8058 |
| test | incremental_no_overturn | 27 | 0.7037 | 0.7037 | 0.6667 | 0.7037 | 0.0 | 0.0 |
| test | incremental_overturn_reasoning | 103 | 0.8544 | 0.8058 | 0.8058 | 0.8058 | 0.1942 | 0.8058 |
