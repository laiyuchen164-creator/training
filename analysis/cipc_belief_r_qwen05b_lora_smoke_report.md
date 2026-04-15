# Commitment-Control Evaluation

## Split Summary

### train

- n: `512`
- control_decision_accuracy: `0.5879`
- final_answer_accuracy: `0.5293`
- joint_accuracy: `0.332`
- consistency_rate_gold: `0.541`
- early_commitment_persistence: `0.284`
- late_evidence_takeover: `0.692`

### dev

- n: `128`
- control_decision_accuracy: `0.5469`
- final_answer_accuracy: `0.5625`
- joint_accuracy: `0.3438`
- consistency_rate_gold: `0.5625`
- early_commitment_persistence: `0.2821`
- late_evidence_takeover: `0.7179`

### test

- n: `128`
- control_decision_accuracy: `0.5312`
- final_answer_accuracy: `0.5469`
- joint_accuracy: `0.3281`
- consistency_rate_gold: `0.5469`
- early_commitment_persistence: `0.3`
- late_evidence_takeover: `0.7`

## By Condition

| split | condition | n | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | full_info | 256 | 0.5859 | 0.5312 | 0.3359 | 0.543 | 0.272 | 0.704 |
| train | incremental_no_overturn | 131 | 0.4962 | 0.3817 | 0.2137 | 0.3817 | 0.0 | 0.0 |
| train | incremental_overturn_reasoning | 125 | 0.688 | 0.68 | 0.448 | 0.704 | 0.296 | 0.68 |
| dev | full_info | 64 | 0.5469 | 0.5625 | 0.3438 | 0.5625 | 0.2821 | 0.7179 |
| dev | incremental_no_overturn | 25 | 0.44 | 0.32 | 0.2 | 0.32 | 0.0 | 0.0 |
| dev | incremental_overturn_reasoning | 39 | 0.6154 | 0.7179 | 0.4359 | 0.7179 | 0.2821 | 0.7179 |
| test | full_info | 64 | 0.5312 | 0.5469 | 0.3281 | 0.5469 | 0.3 | 0.7 |
| test | incremental_no_overturn | 24 | 0.4583 | 0.2917 | 0.125 | 0.2917 | 0.0 | 0.0 |
| test | incremental_overturn_reasoning | 40 | 0.575 | 0.7 | 0.45 | 0.7 | 0.3 | 0.7 |
