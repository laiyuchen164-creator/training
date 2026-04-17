# 2026-04-17 Research Session Summary

## Session Scope

This session stayed strictly on the active EMNLP Belief-R HF/LoRA main line:

- Belief-R only
- training-based CIPC only
- HF/LoRA only
- no prompt-line return
- no ReviseQA
- no dataset expansion
- no split change
- no seed change

The session goal was to continue the consistency-side repair path on top of the
Runpod `highrank_v1` frontier and test whether loss-shape changes could reduce
the overturn / no-overturn trade-off.

## Starting Point

At the start of the session, the active diagnosis was:

- simple weight reduction inside the previous conditional propagation loss moved
  behavior in the expected direction
- but did not beat `highrank_v1`
- the likely problem was that the propagation loss remained too
  replace-dominated
- the next repair should strengthen preserve-side influence and/or separate
  preserve/replace aggregation

Active reference points before today:

| run | overall | overturn | no-overturn |
|---|---:|---:|---:|
| `highrank_v1` | 0.8538 | 0.9029 | 0.6667 |
| `conditional_consistency_v2` | 0.8154 | 0.8738 | 0.5926 |
| `tradeoff_repair_v2` | 0.8000 | 0.7961 | 0.8148 |
| NumPy baseline | 0.7846 | 0.7961 | 0.9259 |
| frozen prompt baseline | 0.3615 | 0.2718 | 0.5926 |

## Infrastructure / Repo Work

Today we also set up persistent research memory and GitHub sync:

- created [PROJECT_MEMORY.md](/workspace/training/PROJECT_MEMORY.md:1) as the
  durable cross-server handoff file
- checked local vs GitHub state
- configured SSH auth for GitHub pushes on this machine
- pushed local commits so the repo and memory file were backed up remotely

## Code Paths Modified

All method work stayed in the existing HF training path:

- [src/models/hf_commitment_control_model.py](/workspace/training/src/models/hf_commitment_control_model.py:1)
- [training/train_commitment_control_hf.py](/workspace/training/training/train_commitment_control_hf.py:1)
- [tests/test_hf_commitment_control_model.py](/workspace/training/tests/test_hf_commitment_control_model.py:1)

No dataset files, evaluation code, split definitions, or sampler logic were
changed.

## Experiment 1: Split Conditional Consistency V1

Report:
- [analysis/cipc_qwen05b_split_conditional_consistency_v1_report.md](/workspace/training/analysis/cipc_qwen05b_split_conditional_consistency_v1_report.md:1)

### Motivation

The prior conditional propagation loss pooled preserve and replace examples into
one average. The hypothesis was that replace examples were dominating the pooled
signal and weakening preserve-side control.

### Loss Redesign

Kept:

- `L = L_ctrl + lambda_ans * L_ans + L_prop`
- `L_ctrl =` control-decision cross-entropy
- `L_ans =` final-answer cross-entropy

Changed:

- preserve and replace examples were aggregated separately
- preserve-side:
  `L_preserve = mean CE(final_answer_dist, early_implied_answer)` over preserve
  examples only
- replace-side:
  `L_replace = mean(CE(final_answer_dist, gold_final_answer) + beta_replace_margin * anti_early_margin)` over replace examples only
- combined with:
  `L_prop = lambda_pres * L_preserve + lambda_rep * L_replace`

Config:

- `lambda_pres = 0.20`
- `lambda_rep = 0.08`
- `beta_replace_margin = 0.05`
- `margin_m = 0.5`

### Result

- overall: `0.8462`
- overturn: `0.9223`
- no-overturn: `0.5556`

### Interpretation

- overturn improved over `highrank_v1`
- no-overturn got worse than `highrank_v1`
- separate aggregation alone did not repair the frontier
- conclusion:
  preserve-side CE remained too weak as a structural maintain-side constraint

## Experiment 2: Preserve Margin Conditional Consistency V1

Report:
- [analysis/cipc_qwen05b_preserve_margin_conditional_consistency_v1_report.md](/workspace/training/analysis/cipc_qwen05b_preserve_margin_conditional_consistency_v1_report.md:1)

### Motivation

After split aggregation failed, the next hypothesis was:

- the preserve side needs a stronger structural constraint than plain CE
- preserve examples should not just "prefer" the early answer
- they should require the early answer to beat alternatives by a margin

### Loss Redesign

Kept split preserve/replace aggregation and the same replace-side objective.

Changed preserve-side from CE to pure margin:

- for preserve examples:
  `L_preserve = mean max(0, preserve_margin_m - (log p(early) - max_non_early_log_prob))`

Config:

- `lambda_pres = 0.20`
- `lambda_rep = 0.08`
- `preserve_margin_m = 0.5`
- `beta_replace_margin = 0.05`
- `margin_m = 0.5`

### Result

- overall: `0.8077`
- overturn: `0.8350`
- no-overturn: `0.7037`

### Interpretation

- first consistency-side run to beat `highrank_v1` on no-overturn
- but overturn dropped too much
- conclusion:
  preserve-side structural constraints are real control levers, but pure
  preserve margin at this strength over-corrects

## Experiment 3: Preserve Margin Conditional Consistency V2

Report:
- [analysis/cipc_qwen05b_preserve_margin_conditional_consistency_v2_report.md](/workspace/training/analysis/cipc_qwen05b_preserve_margin_conditional_consistency_v2_report.md:1)

### Motivation

`preserve_margin_v1` proved preserve margin matters, but `preserve_margin_m=0.5`
looked too strong. The next controlled test was whether weakening only the
margin strength could recover overturn while keeping some maintain-side gain.

### Change

Only one scalar changed:

- `preserve_margin_m: 0.5 -> 0.3`

### Result

- overall: `0.8154`
- overturn: `0.8544`
- no-overturn: `0.6667`

### Interpretation

- overturn recovered relative to margin `v1`
- no-overturn fell back to exactly `highrank_v1`
- conclusion:
  preserve margin is a smooth trade-off control axis, but the simple pure-margin
  form still did not dominate `highrank_v1`

## Experiment 4: Preserve Hybrid Conditional Consistency V1

Report:
- [analysis/cipc_qwen05b_preserve_hybrid_conditional_consistency_v1_report.md](/workspace/training/analysis/cipc_qwen05b_preserve_hybrid_conditional_consistency_v1_report.md:1)

### Motivation

The pure margin runs suggested:

- pure margin helps no-overturn
- but may over-correct

So the next idea was to combine:

- direct CE-to-early supervision for stability
- plus a small preserve margin for extra maintain-side structure

### Loss Redesign

Preserve side became a hybrid:

- `preserve_ce_loss = mean CE(final_answer_dist, early_implied_answer)`
- `preserve_margin_loss = mean max(0, preserve_margin_m - (log p(early) - max_non_early_log_prob))`
- `L_preserve = preserve_ce_loss + beta_preserve_margin * preserve_margin_loss`

Overall propagation remained:

- `L_prop = lambda_pres * L_preserve + lambda_rep * L_replace`

Config:

- `lambda_pres = 0.20`
- `lambda_rep = 0.08`
- `preserve_margin_m = 0.3`
- `beta_preserve_margin = 0.1`
- `beta_replace_margin = 0.05`
- `margin_m = 0.5`

### Result

- overall: `0.8615`
- overturn: `0.9223`
- no-overturn: `0.6296`

### Interpretation

- beat `highrank_v1` on overall
- beat `highrank_v1` on overturn
- still below `highrank_v1` on no-overturn
- conclusion:
  hybrid preserve recovered aggressive performance but did not preserve the
  maintain-side gain from pure margin

## Experiment 5: Preserve Hybrid Conditional Consistency V2

Report:
- [analysis/cipc_qwen05b_preserve_hybrid_conditional_consistency_v2_report.md](/workspace/training/analysis/cipc_qwen05b_preserve_hybrid_conditional_consistency_v2_report.md:1)

### Motivation

One remaining hypothesis was:

- hybrid preserve is directionally right
- the preserve-margin term inside the hybrid may just be too weak

So the controlled follow-up was to strengthen only the hybrid preserve margin
coefficient.

### Change

Only one scalar changed:

- `beta_preserve_margin: 0.1 -> 0.2`

All else stayed fixed.

### Result

- overall: `0.8615`
- overturn: `0.9515`
- no-overturn: `0.5185`

### Interpretation

- overall stayed high
- overturn became very strong
- no-overturn collapsed badly
- conclusion:
  in the current hybrid preserve formulation, strengthening the margin term
  does not repair maintain cases; it pushes the system further toward an
  aggressive overturn geometry

## High-Level Findings From Today

### 1. Separate aggregation was not enough

Separating preserve and replace pools was a necessary attribution-clean change,
but by itself it did not solve the maintain-side failure.

### 2. Preserve-side structure definitely matters

The pure preserve-margin experiments established that strengthen-the-preserve
side is the right causal direction. That was the first route that actually
improved no-overturn above `highrank_v1`.

### 3. Pure preserve margin is a real, smooth control axis

Changing `preserve_margin_m` from `0.5` to `0.3` produced the expected
frontier movement:

- weaker margin -> more overturn
- weaker margin -> less no-overturn protection

So the effect is not noise.

### 4. The simple hybrid preserve form is not the repair

Hybrid preserve recovered aggressive performance very effectively, but it did
not hold onto maintain-side gains. Increasing its margin component made the
model even more overturn-heavy.

### 5. The bottleneck is now objective form, not just scalar strength

By the end of the session, the evidence no longer supports the idea that we
just need "more" or "less" of the current hybrid preserve term. The current
hybrid form seems to route extra pressure into aggressive behavior rather than
stable maintain-side repair.

## Best Reading Of The Frontier After Today

The frontier is now better understood as:

- `highrank_v1`: strong aggressive reference point
- `preserve_margin_v1`: strongest maintain-side consistency result, but too
  costly on overturn
- `preserve_hybrid_v1`: strongest aggressive consistency result, beats
  `highrank_v1` on overall and overturn, but not on no-overturn

No run today cleanly dominated `highrank_v1` on all relevant axes.

## Most Important New Scientific Conclusion

Today established a concrete causal story:

- maintain-side repair requires explicit preserve-side structure
- plain preserve-side CE is too weak
- pure preserve margin can recover maintain behavior
- but the current hybrid preserve design does not combine the two effects in
  the way we want

So the next step should not be another generic weight sweep.

The next step should redesign preserve-side supervision again, using what was
learned here about:

- pure structural preserve constraints
- aggressive-performance recovery under hybrid preserve
- failure of stronger hybrid-margin weighting

## Session Artifacts

Primary artifacts created or updated today:

- [PROJECT_MEMORY.md](/workspace/training/PROJECT_MEMORY.md:1)
- [analysis/cipc_qwen05b_split_conditional_consistency_v1_report.md](/workspace/training/analysis/cipc_qwen05b_split_conditional_consistency_v1_report.md:1)
- [analysis/cipc_qwen05b_preserve_margin_conditional_consistency_v1_report.md](/workspace/training/analysis/cipc_qwen05b_preserve_margin_conditional_consistency_v1_report.md:1)
- [analysis/cipc_qwen05b_preserve_margin_conditional_consistency_v2_report.md](/workspace/training/analysis/cipc_qwen05b_preserve_margin_conditional_consistency_v2_report.md:1)
- [analysis/cipc_qwen05b_preserve_hybrid_conditional_consistency_v1_report.md](/workspace/training/analysis/cipc_qwen05b_preserve_hybrid_conditional_consistency_v1_report.md:1)
- [analysis/cipc_qwen05b_preserve_hybrid_conditional_consistency_v2_report.md](/workspace/training/analysis/cipc_qwen05b_preserve_hybrid_conditional_consistency_v2_report.md:1)
- [analysis/2026-04-17_research_session_summary.md](/workspace/training/analysis/2026-04-17_research_session_summary.md:1)
