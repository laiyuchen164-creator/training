# Commitment-Control Evaluation

## Split Summary

### train

- n: `2050`
- control_decision_accuracy: `0.9551`
- final_answer_accuracy: `0.9502`
- joint_accuracy: `0.9327`
- consistency_rate_gold: `0.9522`
- early_commitment_persistence: `0.016`
- late_evidence_takeover: `0.9816`

### dev

- n: `254`
- control_decision_accuracy: `0.748`
- final_answer_accuracy: `0.7795`
- joint_accuracy: `0.7087`
- consistency_rate_gold: `0.7795`
- early_commitment_persistence: `0.1089`
- late_evidence_takeover: `0.8911`

### test

- n: `260`
- control_decision_accuracy: `0.7769`
- final_answer_accuracy: `0.7923`
- joint_accuracy: `0.7462`
- consistency_rate_gold: `0.7923`
- early_commitment_persistence: `0.0874`
- late_evidence_takeover: `0.9126`

## By Condition

| split | condition | n | control_acc | answer_acc | joint_acc | consistency_gold | early_persistence | late_takeover |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | full_info | 1025 | 0.9551 | 0.9502 | 0.9327 | 0.9522 | 0.016 | 0.9816 |
| train | incremental_no_overturn | 211 | 0.91 | 0.8294 | 0.8152 | 0.8294 | 0.0 | 0.0 |
| train | incremental_overturn_reasoning | 814 | 0.9668 | 0.9816 | 0.9631 | 0.984 | 0.016 | 0.9816 |
| dev | full_info | 127 | 0.748 | 0.7795 | 0.7087 | 0.7795 | 0.1089 | 0.8911 |
| dev | incremental_no_overturn | 26 | 0.5 | 0.3462 | 0.3462 | 0.3462 | 0.0 | 0.0 |
| dev | incremental_overturn_reasoning | 101 | 0.8119 | 0.8911 | 0.802 | 0.8911 | 0.1089 | 0.8911 |
| test | full_info | 130 | 0.7769 | 0.7923 | 0.7462 | 0.7923 | 0.0874 | 0.9126 |
| test | incremental_no_overturn | 27 | 0.4444 | 0.3333 | 0.2963 | 0.3333 | 0.0 | 0.0 |
| test | incremental_overturn_reasoning | 103 | 0.8641 | 0.9126 | 0.8641 | 0.9126 | 0.0874 | 0.9126 |
