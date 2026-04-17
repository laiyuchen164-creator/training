# Commitment-Control Evaluation

## Split Summary

### train

- n: `2050`
- control_decision_accuracy: `0.9873`
- final_answer_accuracy: `0.9824`
- joint_accuracy: `0.9795`
- consistency_rate_gold: `0.9824`
- early_commitment_persistence: `0.0221`
- late_evidence_takeover: `0.9779`

### dev

- n: `254`
- control_decision_accuracy: `0.7953`
- final_answer_accuracy: `0.7874`
- joint_accuracy: `0.7638`
- consistency_rate_gold: `0.7874`
- early_commitment_persistence: `0.1782`
- late_evidence_takeover: `0.8218`

### test

- n: `260`
- control_decision_accuracy: `0.8538`
- final_answer_accuracy: `0.8538`
- joint_accuracy: `0.8308`
- consistency_rate_gold: `0.8538`
- early_commitment_persistence: `0.0971`
- late_evidence_takeover: `0.9029`

## By Condition

| split | condition | n | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | full_info | 1025 | 0.9873 | 0.9824 | 0.9795 | 0.9824 | 0.0221 | 0.9779 |
| train | incremental_no_overturn | 211 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| train | incremental_overturn_reasoning | 814 | 0.984 | 0.9779 | 0.9742 | 0.9779 | 0.0221 | 0.9779 |
| dev | full_info | 127 | 0.7953 | 0.7874 | 0.7638 | 0.7874 | 0.1782 | 0.8218 |
| dev | incremental_no_overturn | 26 | 0.6154 | 0.6538 | 0.5769 | 0.6538 | 0.0 | 0.0 |
| dev | incremental_overturn_reasoning | 101 | 0.8416 | 0.8218 | 0.8119 | 0.8218 | 0.1782 | 0.8218 |
| test | full_info | 130 | 0.8538 | 0.8538 | 0.8308 | 0.8538 | 0.0971 | 0.9029 |
| test | incremental_no_overturn | 27 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.0 | 0.0 |
| test | incremental_overturn_reasoning | 103 | 0.9029 | 0.9029 | 0.8738 | 0.9029 | 0.0971 | 0.9029 |
