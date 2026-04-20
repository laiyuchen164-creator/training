# Daily Progress Summary: 2026-04-20

## Overall Status

Today we moved the project from a stage-report preparation phase into a more
systematic comparison and model-design phase.

The current main direction is still the trained CIPC line, not the earlier
prompt-only source-revision line. The main research question is now more
precise:

> Can we train a source-aware commitment-control model that balances early
> belief preservation and late evidence revision, instead of simply maximizing
> one side?

Current evidence supports strong in-domain gains on Belief-R, while
cross-dataset generalization to ReviseQA remains limited and should be treated
as a future improvement target.

## Completed Work

### 1. Advisor Stage Report Deck

Created a stage-report presentation for the advisor meeting:

- `presentations/advisor_stage_report_2026-04-20.pptx`
- generation script: `scripts/create_advisor_stage_report_ppt.py`

The deck covers project background, motivation, related work, main
contribution, experiment design, expected outcomes, and current experimental
status.

Commit:

- `c03da33 Add advisor stage report deck`

### 2. ReviseQA Full-Set Conversion and CIPC Transfer Evaluation

Converted the full ReviseQA dataset into the commitment-control schema used by
the Belief-R CIPC pipeline.

Converted dataset:

- `data/processed/reviseqa_commitment_control_full.jsonl`

Dataset size:

| split condition | count |
|---|---:|
| full_info | 11151 |
| incremental_no_overturn | 5457 |
| incremental_overturn_reasoning | 5694 |
| total | 22302 |

Evaluated trained CIPC checkpoints on full ReviseQA:

| checkpoint | overall answer | no-overturn | overturn | control |
|---|---:|---:|---:|---:|
| `highrank_v1` | 0.4448 | 0.4427 | 0.4468 | 0.4899 |
| `preserve_hybrid_v1` | 0.4220 | 0.4237 | 0.4204 | 0.4895 |

Interpretation:

- `highrank_v1` transfers slightly better than `preserve_hybrid_v1`.
- Absolute transfer performance is low.
- This suggests that the current method has strong in-domain generalization but
  limited cross-dataset generalization.
- ReviseQA differs from Belief-R in format, answer distribution, edit style,
  and sequence length.

Related report:

- `analysis/reviseqa_cipc_transfer_report.md`

Commits:

- `2761ef2 Add resumable ReviseQA evaluation scripts`
- `80b3fe7 Add CIPC ReviseQA transfer evaluation`

### 3. Direct OpenAI API Baseline on Belief-R with Aligned Prompt

Built and ran an aligned direct OpenAI multiple-choice baseline on the fixed
Belief-R test split.

The aligned prompt explicitly describes the belief-revision setting and the
meaning of the uncertainty option.

Results:

| method | overall | overturn | no-overturn |
|---|---:|---:|---:|
| naive direct OpenAI `gpt-5.4-mini` | 0.2077 | 0.0000 | 1.0000 |
| aligned direct OpenAI `gpt-5.4-mini` | 0.2462 | 0.0485 | 1.0000 |
| CIPC `highrank_v1` | 0.8538 | 0.9029 | 0.6667 |
| CIPC `tradeoff_repair_v2` | 0.8000 | 0.7961 | 0.8148 |

Interpretation:

- Prompt alignment helps slightly.
- Direct prompting still largely preserves the early belief and rarely revises
  correctly on overturn cases.
- The CIPC training-based method remains much stronger on the actual
  commitment-control task.

Related report:

- `analysis/direct_openai_belief_r_aligned_v1_report.md`

Commit:

- `8fafda2 Add aligned OpenAI Belief-R baseline`

### 4. Gated Propagation CIPC Experiment

Implemented and evaluated `gated_propagation_v1`.

Objective:

- use detached control-head probabilities as gates
- encourage the answer head to preserve when control predicts preserve
- encourage the answer head to revise when control predicts replace
- avoid directly backpropagating the propagation loss into the control head

Results:

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| `highrank_v1` | 0.8538 | 0.9029 | 0.6667 |
| `gated_propagation_v1` | 0.8077 | 0.8641 | 0.5926 |

Interpretation:

- This was not a new frontier point.
- The detached soft gate did not repair the preserve-side weakness.
- It likely followed and amplified the model's existing aggressive bias.

Related report:

- `analysis/cipc_qwen05b_gated_propagation_v1_report.md`

Commit:

- `902ce24 Add gated propagation CIPC experiment`

### 5. Gold-Gated Preserve CIPC Experiment

Implemented and evaluated `gold_gated_preserve_v1`.

Objective:

- apply preserve-side answer anchoring only on gold preserve examples
- avoid adding extra replace-side pressure
- test whether gold labels provide a cleaner preserve repair signal than
  predicted soft gates

Results:

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| `highrank_v1` | 0.8538 | 0.9029 | 0.6667 |
| `tradeoff_repair_v2` | 0.8000 | 0.7961 | 0.8148 |
| `gated_propagation_v1` | 0.8077 | 0.8641 | 0.5926 |
| `gold_gated_preserve_v1` | 0.8538 | 0.9126 | 0.6296 |

Interpretation:

- `gold_gated_preserve_v1` is still aggressive.
- It slightly improves overturn accuracy over `highrank_v1`.
- It worsens no-overturn accuracy, so it is not the balanced point.
- Preserve-side CE alone is too weak as a structural constraint.

Related report:

- `analysis/cipc_qwen05b_gold_gated_preserve_v1_report.md`

Commit:

- `95a19b5 Add gold-gated preserve CIPC experiment`

## Current Research Conclusion

The project currently has three clear findings.

First, direct prompting is not enough. Even with a more aligned OpenAI prompt,
the API baseline performs poorly on the Belief-R commitment-control task,
especially on overturn cases.

Second, trained CIPC has strong in-domain value. The best Belief-R checkpoints
substantially outperform direct OpenAI prompting on the fixed test split.

Third, the current CIPC method does not yet have strong cross-dataset
generalization. ReviseQA transfer performance is low, which suggests that
format alignment, answer-space differences, and data distribution shift remain
important open problems.

## Current Frontier

The useful frontier is now between two anchors:

| checkpoint | overall | overturn | no-overturn | role |
|---|---:|---:|---:|---|
| `highrank_v1` | 0.8538 | 0.9029 | 0.6667 | aggressive anchor |
| `tradeoff_repair_v2` | 0.8000 | 0.7961 | 0.8148 | conservative anchor |

The goal is not necessarily to make every metric best at once. The next stage
should search for a balanced model that preserves most of the overturn strength
while recovering more no-overturn accuracy.

Target range for a useful balanced point:

| metric | target |
|---|---:|
| overall answer | 0.82-0.85 |
| overturn answer | 0.84-0.88+ |
| no-overturn answer | 0.74-0.80 |

## Recommended Next Step

The immediate next model should be:

`gold_gated_preserve_margin_v1`

Proposed objective:

```text
L = L_control + L_answer
    + beta * I[gold_control = preserve]
      * max(0, margin - (log p(early_answer) - max_non_early_log_p))
```

Rationale:

- CE-to-early on preserve examples was too weak.
- A preserve-side margin directly requires the early answer to outrank
  alternatives on gold preserve examples.
- No additional replace-side pressure should be added initially, because the
  aggressive checkpoints are already strong on overturn cases.

This is the most direct next attempt to move from the aggressive anchor toward
a better balanced model.
