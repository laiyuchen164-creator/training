# Commitment-Control Evaluation

## Split Summary

### train

- n: `2050`
- control_decision_accuracy: `0.9239`
- final_answer_accuracy: `0.9229`
- joint_accuracy: `0.9102`
- consistency_rate_gold: `0.9229`
- early_commitment_persistence: `0.0971`
- late_evidence_takeover: `0.9029`

### dev

- n: `254`
- control_decision_accuracy: `0.7244`
- final_answer_accuracy: `0.7559`
- joint_accuracy: `0.7165`
- consistency_rate_gold: `0.7559`
- early_commitment_persistence: `0.2574`
- late_evidence_takeover: `0.7426`

### test

- n: `260`
- control_decision_accuracy: `0.7923`
- final_answer_accuracy: `0.7846`
- joint_accuracy: `0.7615`
- consistency_rate_gold: `0.7846`
- early_commitment_persistence: `0.233`
- late_evidence_takeover: `0.767`

## By Condition

| split | condition | n | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | full_info | 1025 | 0.9239 | 0.9229 | 0.9102 | 0.9229 | 0.0971 | 0.9029 |
| train | incremental_no_overturn | 211 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| train | incremental_overturn_reasoning | 814 | 0.9042 | 0.9029 | 0.887 | 0.9029 | 0.0971 | 0.9029 |
| dev | full_info | 127 | 0.7244 | 0.7559 | 0.7165 | 0.7559 | 0.2574 | 0.7426 |
| dev | incremental_no_overturn | 26 | 0.6923 | 0.8077 | 0.6923 | 0.8077 | 0.0 | 0.0 |
| dev | incremental_overturn_reasoning | 101 | 0.7327 | 0.7426 | 0.7228 | 0.7426 | 0.2574 | 0.7426 |
| test | full_info | 130 | 0.7923 | 0.7846 | 0.7615 | 0.7846 | 0.233 | 0.767 |
| test | incremental_no_overturn | 27 | 0.8148 | 0.8519 | 0.8148 | 0.8519 | 0.0 | 0.0 |
| test | incremental_overturn_reasoning | 103 | 0.7864 | 0.767 | 0.7476 | 0.767 | 0.233 | 0.767 |
