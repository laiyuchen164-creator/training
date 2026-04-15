# Commitment-Control Evaluation

## Split Summary

### train

- n: `1536`
- control_decision_accuracy: `0.6641`
- final_answer_accuracy: `0.6374`
- joint_accuracy: `0.6335`
- consistency_rate_gold: `0.6426`
- early_commitment_persistence: `0.447`
- late_evidence_takeover: `0.5464`

### dev

- n: `254`
- control_decision_accuracy: `0.6535`
- final_answer_accuracy: `0.6378`
- joint_accuracy: `0.622`
- consistency_rate_gold: `0.6535`
- early_commitment_persistence: `0.4257`
- late_evidence_takeover: `0.5545`

### test

- n: `260`
- control_decision_accuracy: `0.6692`
- final_answer_accuracy: `0.6308`
- joint_accuracy: `0.6308`
- consistency_rate_gold: `0.6308`
- early_commitment_persistence: `0.4466`
- late_evidence_takeover: `0.5534`

## By Condition

| split | condition | n | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | full_info | 795 | 0.6654 | 0.6377 | 0.634 | 0.6428 | 0.4503 | 0.5433 |
| train | incremental_no_overturn | 148 | 0.9865 | 0.9865 | 0.9865 | 0.9865 | 0.0 | 0.0 |
| train | incremental_overturn_reasoning | 593 | 0.5818 | 0.5497 | 0.5447 | 0.5565 | 0.4435 | 0.5497 |
| dev | full_info | 127 | 0.6535 | 0.6378 | 0.622 | 0.6535 | 0.4257 | 0.5545 |
| dev | incremental_no_overturn | 26 | 0.9615 | 0.9615 | 0.9615 | 0.9615 | 0.0 | 0.0 |
| dev | incremental_overturn_reasoning | 101 | 0.5743 | 0.5545 | 0.5347 | 0.5743 | 0.4257 | 0.5545 |
| test | full_info | 130 | 0.6692 | 0.6308 | 0.6308 | 0.6308 | 0.4466 | 0.5534 |
| test | incremental_no_overturn | 27 | 0.9259 | 0.9259 | 0.9259 | 0.9259 | 0.0 | 0.0 |
| test | incremental_overturn_reasoning | 103 | 0.6019 | 0.5534 | 0.5534 | 0.5534 | 0.4466 | 0.5534 |
