# Commitment-Control Evaluation

## Split Summary

### train

- n: `2050`
- control_decision_accuracy: `0.9785`
- final_answer_accuracy: `0.9717`
- joint_accuracy: `0.9707`
- consistency_rate_gold: `0.9717`
- early_commitment_persistence: `0.0356`
- late_evidence_takeover: `0.9644`

### dev

- n: `254`
- control_decision_accuracy: `0.7717`
- final_answer_accuracy: `0.7559`
- joint_accuracy: `0.7244`
- consistency_rate_gold: `0.7559`
- early_commitment_persistence: `0.2178`
- late_evidence_takeover: `0.7822`

### test

- n: `260`
- control_decision_accuracy: `0.8615`
- final_answer_accuracy: `0.8462`
- joint_accuracy: `0.8231`
- consistency_rate_gold: `0.8462`
- early_commitment_persistence: `0.1359`
- late_evidence_takeover: `0.8641`

## By Condition

| split | condition | n | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | full_info | 1025 | 0.9785 | 0.9717 | 0.9707 | 0.9717 | 0.0356 | 0.9644 |
| train | incremental_no_overturn | 211 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| train | incremental_overturn_reasoning | 814 | 0.973 | 0.9644 | 0.9631 | 0.9644 | 0.0356 | 0.9644 |
| dev | full_info | 127 | 0.7717 | 0.7559 | 0.7244 | 0.7559 | 0.2178 | 0.7822 |
| dev | incremental_no_overturn | 26 | 0.6538 | 0.6538 | 0.5769 | 0.6538 | 0.0 | 0.0 |
| dev | incremental_overturn_reasoning | 101 | 0.802 | 0.7822 | 0.7624 | 0.7822 | 0.2178 | 0.7822 |
| test | full_info | 130 | 0.8615 | 0.8462 | 0.8231 | 0.8462 | 0.1359 | 0.8641 |
| test | incremental_no_overturn | 27 | 0.7778 | 0.7778 | 0.7778 | 0.7778 | 0.0 | 0.0 |
| test | incremental_overturn_reasoning | 103 | 0.8835 | 0.8641 | 0.835 | 0.8641 | 0.1359 | 0.8641 |
