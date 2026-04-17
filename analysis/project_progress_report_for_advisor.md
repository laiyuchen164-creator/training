# Project Progress Report for Advisor

## 1. Project Overview

### 1.1 Research Goal

This project studies **source-aware belief revision** under incremental
evidence. The core scientific question is:

- when new evidence arrives, can a system correctly decide whether to
  **maintain** the early commitment or **revise** it
- and, if revision is needed, can it correctly **propagate** that change to the
  final answer

The project has evolved into two lines:

1. a **frozen prompt-based diagnosis line**, used to understand the task and
   establish baselines
2. an **active training-based line**, called **CIPC**
   (Commitment Integration and Propagation Control), which is now the main
   method line

### 1.2 Current Main Line

The active line is now:

- Belief-R only
- training-based CIPC only
- HF/LoRA only
- fixed tracked split and seed
- no prompt-line return as the main method

The main open problem is no longer infrastructure or basic feasibility. The
main problem is the **trade-off**:

- keep strong **overturn** performance on cases where the early commitment
  should change
- while recovering strong **no-overturn** performance on cases where the early
  commitment should be preserved

## 2. Task and Data Formulation

### 2.1 Original Benchmark

The core dataset is the **Belief-R strong paired subset**. The project turns
the original paired examples into an incremental decision task with three main
conditions:

- `full_info`
- `incremental_no_overturn`
- `incremental_overturn_reasoning`

Interpretation:

- `incremental_no_overturn`: new evidence should **not** change the earlier
  commitment
- `incremental_overturn_reasoning`: new evidence **should** overturn the earlier
  commitment

### 2.2 Commitment-Control Conversion

To support training, the Belief-R data was converted into a supervised
commitment-control format:

- train: `2050`
- dev: `254`
- test: `260`

Active label space in the first Belief-R training path:

- control labels:
  - `preserve`
  - `replace`
  - `weaken` kept in the label space for compatibility, but not active in the
    first Belief-R pass
- answer labels:
  - `a`
  - `b`
  - `c`

This conversion is important because it explicitly separates two subproblems:

1. **integration**
   decide whether to preserve or replace the early commitment
2. **propagation**
   produce the correct final answer after that control decision

## 3. Phase-by-Phase Project History

## 3.1 Phase A: Prompt-Based Diagnosis

### What was built

The early phase built the repository structure and prompt-based runner with
four systems:

- `raw_history`
- `running_summary`
- `structured_no_source`
- `source_revision`

Later, an additional ablation was added:

- `source_no_revision`

### Why this phase mattered

This phase was not the final method line. Its purpose was:

- establish the task framing
- verify the mechanism claim
- measure whether source-aware revision helps on belief-overturn cases

### Pilot result

On a compact pilot, the source-aware revision system was strongest on overturn
cases, but worse on no-overturn cases.

This was the first evidence for the core trade-off:

- more aggressive revision helps overturn
- but hurts maintain-side stability

### OpenAI / API validation

The project later moved the same evaluation stack to real API models. This
validated that the same trade-off exists with real LLM execution, not just
heuristic simulation.

Most important result from the OpenAI medium pilot:

- `source_revision` was clearly strongest on
  `incremental_overturn_reasoning`
- but weaker on `incremental_no_overturn`

### Source-no-revision ablation

This was a key conceptual milestone.

Finding:

- `structured_no_source` and `source_no_revision` were both weak on overturn
- `source_revision` was much stronger

Interpretation:

- source tags alone are not enough
- the gain comes from the **revision policy**, not just source metadata

### Outcome of prompt phase

The prompt phase successfully established:

- the task is real
- the mechanism claim is meaningful
- the main empirical difficulty is the overturn / no-overturn trade-off

However, the prompt line was eventually frozen because:

- it was not the best path to a strong paper result
- the project needed a trainable model line

## 3.2 Phase B: NumPy Training Baseline (CIPC v1)

Report:
- [analysis/cipc_belief_r_lora_v1_report.md](/workspace/training/analysis/cipc_belief_r_lora_v1_report.md:1)

### Why the project moved to training

The prompt line showed the mechanism but remained limited. So the project
shifted to a training-first line:

- **CIPC: Commitment Integration and Propagation Control**

The first executable version was a NumPy multitask baseline.

### Model design

The model predicts two outputs:

1. `control_decision`
2. `final_answer`

The initial training loss was:

`L = L_ctrl + L_ans`

where:

- `L_ctrl =` cross-entropy over control labels
- `L_ans =` cross-entropy over final-answer labels

Important practical addition:

- deterministic oversampling of the minority control label in training

Reason:

- without this, the model tended to collapse toward all-`replace`

### NumPy baseline result

Test result:

- control accuracy: `0.8923`
- answer accuracy: `0.7846`
- joint accuracy: `0.7808`

Condition-level answer highlights:

- `incremental_no_overturn`: `0.9259`
- `incremental_overturn_reasoning`: `0.7961`

### Interpretation

This was a major project milestone:

- the project now had a real trainable pipeline
- no-overturn behavior was strong
- overturn behavior was still useful
- the model separated the two failure modes:
  - commitment integration was already strong
  - answer propagation still lagged

This baseline also beat the frozen prompt baseline by a large margin.

## 3.3 Phase C: HF/LoRA Upgrade

### Why this phase was necessary

The NumPy baseline was a good proof of concept, but it was not a final
architecture. The next step was to move onto a real pretrained instruction
model with HF/LoRA.

The chosen family was:

- `Qwen/Qwen2.5-0.5B-Instruct`

### First HF baseline

The first HF/LoRA `balanced_full_v1` run was weak:

- overall answer: `0.6462`
- overturn answer: `0.5728`
- no-overturn answer: `0.9259`

Interpretation:

- maintain-side behavior was strong
- overturn propagation was much too weak

### Two local frontier follow-ups

Report:
- [analysis/cipc_qwen05b_local_followup_report.md](/workspace/training/analysis/cipc_qwen05b_local_followup_report.md:1)

Two targeted local follow-ups were then run.

#### Control-focused v1

Changes:

- more epochs
- lower LR
- reduced `answer_loss_weight` from `1.0` to `0.7`
- LoRA rank unchanged

Intent:

- emphasize the control decision more strongly

Result:

- overall answer: `0.7923`
- overturn answer: `0.7670`
- no-overturn answer: `0.8889`

Interpretation:

- a large recovery over the first HF baseline
- much closer to the NumPy baseline
- still relatively balanced

#### High-rank v1

Changes:

- increased LoRA rank from `8` to `16`
- increased LoRA alpha from `16` to `32`
- slightly longer training
- lower LR
- kept `answer_loss_weight = 1.0`

Intent:

- test whether the remaining errors were partly capacity-limited

Result:

- overall answer: `0.8462`
- overturn answer: `0.8641`
- no-overturn answer: `0.7778`

Interpretation:

- strongest raw HF result locally
- beat the NumPy baseline on overall and overturn
- but reintroduced maintain-side regression

### Frontier understanding after local HF phase

At this point, the project had two useful HF frontier points:

- more balanced point: `control_focused_v1`
- more aggressive point: `highrank_v1`

The new scientific question became:

- can we keep the strong overturn behavior of `highrank_v1`
- while recovering the no-overturn conservatism of `control_focused_v1`

## 3.4 Phase D: Runpod Reproduction and Trade-Off Repair

Report:
- [analysis/cipc_qwen05b_runpod_phase_summary.md](/workspace/training/analysis/cipc_qwen05b_runpod_phase_summary.md:1)

### Why Runpod

The local frontier was promising, but the project needed stronger GPU-backed
training and cleaner scaling. Runpod was used for the main HF/LoRA phase.

### Key Runpod findings

#### Reproduction changed the frontier shape

Runpod `control_focused_v1` did not behave like the local balanced point.
Instead:

- Runpod `highrank_v1` became the active aggressive frontier anchor

Runpod `highrank_v1`:

- overall answer: `0.8538`
- overturn answer: `0.9029`
- no-overturn answer: `0.6667`

#### Answer-loss trade-off repair

Several runs changed only `answer_loss_weight`.

Main finding:

- lower values pushed toward overturn
- higher values pulled back toward maintain

Best conservative anchor:

- `tradeoff_repair_v2`
  - overall: `0.8000`
  - overturn: `0.7961`
  - no-overturn: `0.8148`

Interpretation:

- `answer_loss_weight` is a clean trade-off control
- but no setting beat `highrank_v1` overall

#### Checkpoint selection and sampler repair failed

- changing only checkpoint selection did not help
- increasing no-overturn sampling did not help

Interpretation:

- the problem is not mainly selection
- the problem is not mainly simple sampling balance

### Conclusion after trade-off repair

The problem now looked like:

- an **objective-shape** problem
- not primarily a sampler problem
- not primarily a checkpoint problem

This is what motivated the consistency-loss redesigns.

## 4. Loss Function Evolution on the Consistency / Propagation Line

This section is the most important technically, because this is where the
current research focus lies.

## 4.1 Original HF Objective

Before the consistency redesign work, the basic HF objective was:

`L = L_ctrl + lambda_ans * L_ans`

where:

- `L_ctrl =` control-decision cross-entropy
- `L_ans =` final-answer cross-entropy
- `lambda_ans` is implemented as `answer_loss_weight`

This objective already produced a real frontier, but it did not repair the
overturn / no-overturn trade-off.

## 4.2 Minimal Consistency Loss

Before today, the first consistency path tried to tie:

- the control head's preserve probability
- to the answer head's early-answer probability

That line showed:

- consistency can move the frontier
- but the formulation is unstable

Runpod results:

- `highrank_consistency_v1`: good no-overturn, weak overturn
- `highrank_consistency_v2`: very strong overturn, collapsed no-overturn

Conclusion:

- consistency matters
- the old formulation is the wrong shape

## 4.3 Conditional Propagation Loss

The next redesign was to make the propagation loss more directly semantic.

### Conditional consistency v1

Report:
- [analysis/cipc_qwen05b_conditional_consistency_v1_report.md](/workspace/training/analysis/cipc_qwen05b_conditional_consistency_v1_report.md:1)

Preserve:

- encourage final answer to match early implied answer

Replace:

- encourage final answer to match gold final answer
- add anti-early margin

Result:

- overall: `0.8923`
- overturn: `0.9806`
- no-overturn: `0.5556`

Interpretation:

- extremely aggressive
- replace-dominated
- not a repair

### Conditional consistency v2

Report:
- [analysis/cipc_qwen05b_conditional_consistency_v2_report.md](/workspace/training/analysis/cipc_qwen05b_conditional_consistency_v2_report.md:1)

Change:

- reduced the propagation weight and replace margin strength

Result:

- overall: `0.8154`
- overturn: `0.8738`
- no-overturn: `0.5926`

Interpretation:

- moved in the right direction
- but still below `highrank_v1` on both overturn and no-overturn
- confirmed the v1 failure was not random

## 4.4 Today’s Full Consistency Redesign Chain

Session summary:
- [analysis/2026-04-17_research_session_summary.md](/workspace/training/analysis/2026-04-17_research_session_summary.md:1)

### Step 1: Split Conditional Consistency

Report:
- [analysis/cipc_qwen05b_split_conditional_consistency_v1_report.md](/workspace/training/analysis/cipc_qwen05b_split_conditional_consistency_v1_report.md:1)

#### Motivation

The conditional propagation loss still averaged preserve and replace examples
together. Hypothesis:

- replace examples dominate the propagation pool

#### Loss change

Preserve and replace were separated:

- `L_preserve = mean CE(final_answer, early_answer)` over preserve examples only
- `L_replace = mean(CE(final_answer, gold_final_answer) + anti_early_margin)` over replace examples only
- `L_prop = lambda_pres * L_preserve + lambda_rep * L_replace`

#### Result

- overall: `0.8462`
- overturn: `0.9223`
- no-overturn: `0.5556`

#### Interpretation

- separate aggregation alone was not enough
- preserve-side CE remained too weak

### Step 2: Pure Preserve Margin

Reports:
- [analysis/cipc_qwen05b_preserve_margin_conditional_consistency_v1_report.md](/workspace/training/analysis/cipc_qwen05b_preserve_margin_conditional_consistency_v1_report.md:1)
- [analysis/cipc_qwen05b_preserve_margin_conditional_consistency_v2_report.md](/workspace/training/analysis/cipc_qwen05b_preserve_margin_conditional_consistency_v2_report.md:1)

#### Motivation

Hypothesis:

- preserve-side CE is too weak
- preserve examples need a real structural constraint

#### Loss change

Replace preserve-side CE with a margin:

- require `log p(early)` to beat the strongest non-early answer by at least
  `preserve_margin_m`

#### Two runs

`v1: preserve_margin_m = 0.5`

- overall: `0.8077`
- overturn: `0.8350`
- no-overturn: `0.7037`

`v2: preserve_margin_m = 0.3`

- overall: `0.8154`
- overturn: `0.8544`
- no-overturn: `0.6667`

#### Interpretation

- this was the first path to raise no-overturn above `highrank_v1`
- but stronger margin hurt overturn
- weakening the margin recovered overturn but removed the maintain-side gain

Key lesson:

- pure preserve margin is a real, smooth control axis

### Step 3: Preserve Hybrid Loss

Reports:
- [analysis/cipc_qwen05b_preserve_hybrid_conditional_consistency_v1_report.md](/workspace/training/analysis/cipc_qwen05b_preserve_hybrid_conditional_consistency_v1_report.md:1)
- [analysis/cipc_qwen05b_preserve_hybrid_conditional_consistency_v2_report.md](/workspace/training/analysis/cipc_qwen05b_preserve_hybrid_conditional_consistency_v2_report.md:1)

#### Motivation

Hypothesis:

- pure margin helps maintain-side behavior
- but may be too strong
- maybe the right design is:
  `CE_to_early + small preserve margin`

#### Loss change

Preserve-side became:

- `preserve_ce_loss = CE(final_answer, early_answer)`
- `preserve_margin_loss = margin(early vs strongest alternative)`
- `L_preserve = preserve_ce_loss + beta_preserve_margin * preserve_margin_loss`

#### Hybrid v1

Settings:

- `preserve_margin_m = 0.3`
- `beta_preserve_margin = 0.1`

Result:

- overall: `0.8615`
- overturn: `0.9223`
- no-overturn: `0.6296`

Interpretation:

- strongest aggressive result on this branch
- beat `highrank_v1` on overall and overturn
- still failed to beat `highrank_v1` on no-overturn

#### Hybrid v2

Settings:

- same as v1, except:
  - `beta_preserve_margin: 0.1 -> 0.2`

Result:

- overall: `0.8615`
- overturn: `0.9515`
- no-overturn: `0.5185`

Interpretation:

- increasing the hybrid preserve-margin weight made the model **more**
  overturn-heavy, not more maintain-stable
- this strongly suggests the current hybrid form is not the right repair

## 5. Current Project Understanding

At the end of today, the project has reached a much sharper diagnosis.

### What we now know

1. Prompt baselines were useful for diagnosis but are not the active method
   line.
2. The NumPy CIPC baseline proved the training formulation is valid.
3. HF/LoRA clearly works and can beat the prompt baseline by a large margin.
4. `highrank_v1` is the active aggressive Runpod frontier anchor.
5. The main difficulty is objective-shape trade-off, not infrastructure,
   checkpointing, or simple sampler balance.
6. Preserve-side structure matters causally.
7. The current hybrid preserve design is not the final answer.

### Best current points on the frontier

From different perspectives:

- strongest aggressive Runpod anchor:
  - `highrank_v1`
  - overall `0.8538`
  - overturn `0.9029`
  - no-overturn `0.6667`
- strongest conservative anchor:
  - `tradeoff_repair_v2`
  - overall `0.8000`
  - overturn `0.7961`
  - no-overturn `0.8148`
- strongest maintain-side consistency result:
  - `preserve_margin_v1`
  - overall `0.8077`
  - overturn `0.8350`
  - no-overturn `0.7037`
- strongest aggressive consistency result:
  - `preserve_hybrid_v2`
  - overall `0.8615`
  - overturn `0.9515`
  - no-overturn `0.5185`

No single point currently dominates the frontier on all axes.

## 6. Scientific Takeaways for Advisor Discussion

If this project is presented to an advisor, the most important message is:

- the project has moved from **problem diagnosis** to **trainable model
  design**
- the core empirical bottleneck has been isolated to a concrete, reproducible
  trade-off in the loss geometry

The strongest scientific contributions so far are:

1. a clean reformulation of Belief-R as a commitment-control training task
2. a training-based CIPC pipeline that strongly outperforms the frozen prompt
   baseline
3. a reproducible HF/LoRA frontier showing a real overturn / no-overturn
   trade-off
4. direct causal evidence that preserve-side structural constraints matter for
   maintain-side repair
5. direct causal evidence that the current hybrid preserve form is not yet the
   right objective

## 7. Recommended Next Step

The next step should **not** be another generic weight sweep.

Why:

- simple `answer_loss_weight` sweeps are already well understood
- pure margin strength sweeps are already well understood
- strengthening the current hybrid preserve margin made the model more
  aggressive, not more stable

The next step should be a **new preserve-side objective form**, not just a new
scalar value.

In other words:

- the next research contribution should come from redesigning preserve-side
  supervision again, using what has already been learned from:
  - plain preserve CE
  - pure preserve margin
  - hybrid preserve CE + margin

## 8. Files to Show in the Meeting

If only a few files are shown to the advisor, the most useful set is:

- [analysis/project_progress_report_for_advisor.md](/workspace/training/analysis/project_progress_report_for_advisor.md:1)
- [analysis/2026-04-17_research_session_summary.md](/workspace/training/analysis/2026-04-17_research_session_summary.md:1)
- [analysis/cipc_qwen05b_runpod_phase_summary.md](/workspace/training/analysis/cipc_qwen05b_runpod_phase_summary.md:1)
- [analysis/cipc_qwen05b_preserve_margin_conditional_consistency_v1_report.md](/workspace/training/analysis/cipc_qwen05b_preserve_margin_conditional_consistency_v1_report.md:1)
- [analysis/cipc_qwen05b_preserve_hybrid_conditional_consistency_v2_report.md](/workspace/training/analysis/cipc_qwen05b_preserve_hybrid_conditional_consistency_v2_report.md:1)
