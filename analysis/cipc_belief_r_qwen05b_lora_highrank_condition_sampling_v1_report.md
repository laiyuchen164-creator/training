# Commitment-Control Evaluation

## Split Summary

### train

- n: `2050`
- control_decision_accuracy: `0.9493`
- final_answer_accuracy: `0.9161`
- joint_accuracy: `0.9122`
- consistency_rate_gold: `0.9171`
- early_commitment_persistence: `0.1044`
- late_evidence_takeover: `0.8943`

### dev

- n: `254`
- control_decision_accuracy: `0.7795`
- final_answer_accuracy: `0.7559`
- joint_accuracy: `0.7323`
- consistency_rate_gold: `0.7559`
- early_commitment_persistence: `0.2079`
- late_evidence_takeover: `0.7921`

### test

- n: `260`
- control_decision_accuracy: `0.8154`
- final_answer_accuracy: `0.7538`
- joint_accuracy: `0.7231`
- consistency_rate_gold: `0.7615`
- early_commitment_persistence: `0.1845`
- late_evidence_takeover: `0.8058`

## By Condition

| split | condition | n | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | full_info | 1025 | 0.9493 | 0.9161 | 0.9122 | 0.9171 | 0.1044 | 0.8943 |
| train | incremental_no_overturn | 211 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| train | incremental_overturn_reasoning | 814 | 0.9361 | 0.8943 | 0.8894 | 0.8956 | 0.1044 | 0.8943 |
| dev | full_info | 127 | 0.7795 | 0.7559 | 0.7323 | 0.7559 | 0.2079 | 0.7921 |
| dev | incremental_no_overturn | 26 | 0.5 | 0.6154 | 0.5 | 0.6154 | 0.0 | 0.0 |
| dev | incremental_overturn_reasoning | 101 | 0.8515 | 0.7921 | 0.7921 | 0.7921 | 0.2079 | 0.7921 |
| test | full_info | 130 | 0.8154 | 0.7538 | 0.7231 | 0.7615 | 0.1845 | 0.8058 |
| test | incremental_no_overturn | 27 | 0.5185 | 0.5556 | 0.4444 | 0.5556 | 0.0 | 0.0 |
| test | incremental_overturn_reasoning | 103 | 0.8932 | 0.8058 | 0.7961 | 0.8155 | 0.1845 | 0.8058 |
