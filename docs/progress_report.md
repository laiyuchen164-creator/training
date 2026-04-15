# Progress Report

## Stage 0 - Scope Lock

Status: completed

- Locked the first milestone to a real pilot on Belief-R.
- Kept scope inference-time only: no training, no fine-tuning, no branch
  execution, and no agentic tool use.
- Chosen initial condition set:
  `full_info`, `incremental_no_overturn`,
  `incremental_overturn_reasoning`.

## Stage 1 - Repository Setup

Status: completed

- Created the required project structure.
- Added a reproducible config for the Belief-R pilot.
- Added versioned prompt placeholders, a runner entrypoint, and ledger tests.

## Stage 2 - Data Pipeline

Status: completed

- Download Belief-R raw CSVs.
- Transform the strong paired subset into JSONL examples with deterministic IDs.
- Save condition statistics for the pilot.

Current data status:

- Paired examples: 1282
- `full_info`: 1282
- `incremental_no_overturn`: 264
- `incremental_overturn_reasoning`: 1018
- Skipped unpaired weak examples: 462

## Stage 3 - Core Method and Baselines

Status: completed for the pilot milestone

- Implement the belief ledger and revision status transitions.
- Implement four systems:
  `raw_history`, `running_summary`, `structured_no_source`,
  `source_revision`.
- Save turn-level traces and cost proxies for each example.

## Stage 4 - Pilot Run

Status: completed

- Run a compact pilot on 40 examples per condition.
- Export predictions, traces, and aggregate summary tables.
- Update this report with concrete numbers and current blockers.

Pilot sample:

- 120 examples total
- 40 `full_info`
- 40 `incremental_no_overturn`
- 40 `incremental_overturn_reasoning`

Pilot readout:

- `full_info` accuracy: 0.75 for all four systems.
- `incremental_no_overturn` accuracy:
  `raw_history` 0.875,
  `running_summary` 0.800,
  `structured_no_source` 0.575,
  `source_revision` 0.475.
- `incremental_overturn_reasoning` accuracy:
  `raw_history` 0.525,
  `running_summary` 0.600,
  `structured_no_source` 0.650,
  `source_revision` 0.700.
- `stale_belief_persistence` on overturn cases decreases monotonically:
  0.475 -> 0.400 -> 0.350 -> 0.300.
- `assistant_assumption_survival` on overturn cases shows the same pattern:
  0.475 -> 0.400 -> 0.350 -> 0.300.

Interpretation:

- The pilot already shows the intended mechanism signal:
  the source-aware revision system recovers more often on overturn cases than
  the history-only and summary baselines.
- The trade-off is also visible:
  aggressive revision hurts no-overturn performance.
- This is useful for the paper because it matches the framing in the execution
  plan, but the current backend is still heuristic and not publishable as final
  evidence.

Current blockers before paper-ready experiments:

- Replace the heuristic backend with an API-backed LLM backend.
- Add the missing `source_with_source_but_without_revision` ablation if kept
  distinct from `structured_no_source`.
- Extend beyond Belief-R to at least one additional dataset or transformed
  incremental task.
- Add interpretation-overturn coverage.
- Produce publication-ready figures from `paper_assets/`.

## Stage 5 - API Backend Validation

Status: completed for small-scale smoke validation

- Added chat-completions compatible API support for OpenAI and DeepSeek.
- Kept the same experiment interface, ledger, traces, and summary outputs.
- Added versioned LLM prompt templates for turn 1 and system-specific follow-up
  turns.
- Ran a real OpenAI-backed smoke pilot on 6 examples total
  (2 per condition, 4 systems).

OpenAI small-run highlights:

- Config:
  `configs/api_pilot_belief_r_openai_small.json`
- Run directory:
  `runs/belief_r_api_pilot_openai_small/`
- Model path used:
  `gpt-5.4-mini`
- JSON output parsing worked end to end.
- Prompt-token accounting now comes from API usage where available.

What this stage proves:

- The project no longer depends on heuristic outputs alone.
- We can now run real model-based comparisons while preserving the same logging
  and evaluation stack.
- The next step is scaling sample size and improving prompts, not rebuilding
  infrastructure.

Remaining blockers after API validation:

- Increase the OpenAI pilot from smoke scale to a statistically meaningful
  sample.
- Tune prompts so full-info performance is a stronger reference condition.
- Add a clean `source-without-revision` ablation if we keep it distinct.
- Add at least one second dataset before paper drafting.

## Stage 6 - OpenAI Medium Pilot

Status: completed

- Added a medium-scale OpenAI config:
  `configs/api_pilot_belief_r_openai_medium.json`
- Ran 30 sampled examples total, 10 per condition.
- Completed 120 scored records and 200 real API calls.
- Logged 93,141 prompt tokens across the run.

OpenAI medium-run highlights:

- `raw_history`
  - `full_info`: 0.40
  - `incremental_no_overturn`: 0.60
  - `incremental_overturn_reasoning`: 0.50
- `running_summary`
  - `full_info`: 0.20
  - `incremental_no_overturn`: 0.40
  - `incremental_overturn_reasoning`: 0.70
- `structured_no_source`
  - `full_info`: 0.20
  - `incremental_no_overturn`: 0.70
  - `incremental_overturn_reasoning`: 0.20
- `source_revision`
  - `full_info`: 0.20
  - `incremental_no_overturn`: 0.30
  - `incremental_overturn_reasoning`: 1.00

Interpretation:

- The source-aware system is now clearly strongest on the belief-update subset
  under real API execution.
- The belief-maintain trade-off is also very clear, which supports the paper's
  mechanism framing rather than a generic "more revision is always better"
  framing.
- `full_info` remains under-optimized and should be treated as a prompt-design
  problem, not yet as a stable empirical claim.

## Stage 7 - Second Dataset Recon

Status: completed via fallback dataset path

- Investigated `ReviseQA` as the next candidate dataset.
- Confirmed the workshop paper is available.
- The publicly exposed Hugging Face artifact under the same name does not yet
  present a clean, obviously matching benchmark interface for direct ingestion.

Fallback executed:

- Built a second dataset path from the official ATOMIC release.
- Added `atomic_explicit_revision`, a synthetic transform with explicit
  `maintain` and `update` cases.
- Generated:
  600 `full_info`,
  300 `incremental_no_overturn`,
  300 `incremental_overturn_reasoning` examples.
- Ran a small OpenAI pilot on the new dataset.

Interpretation:

- The ATOMIC explicit transform is easier than Belief-R.
- Under explicit correction cues, system differences largely collapse.
- This is useful for the paper because it shows the source-aware method is most
  valuable when the revision cue is implicit or under-specified, not when the
  correction is spelled out directly.

## Stage 8 - Source-Without-Revision Ablation

Status: completed

- Added a new ablation system:
  `source_no_revision`
- This system exposes source tags in the ledger but does not apply the
  source-conditioned revision policy.
- Ran a five-system OpenAI medium pilot with the same 30-example sample.

Most important result:

- On `incremental_overturn_reasoning`,
  `structured_no_source` = 0.20,
  `source_no_revision` = 0.20,
  `source_revision` = 0.90.
- On the same subset, `assistant_assumption_survival` drops from 0.80 in both
  non-revision structured systems to 0.10 in `source_revision`.

Interpretation:

- Source metadata by itself is not enough.
- The improvement comes from the persistence-control rule, which is exactly the
  mechanism the paper is supposed to test.

## Stage 9 - Figure Assets

Status: completed

- Added SVG figure generation for the latest OpenAI ablation run.
- Exported:
  `paper_assets/belief_r_api_openai_ablation_overturn_accuracy.svg`
- Exported:
  `paper_assets/belief_r_api_openai_ablation_no_overturn_accuracy.svg`
- Exported:
  `paper_assets/belief_r_api_openai_ablation_tradeoff.svg`

## Stage 10 - ReviseQA Official Integration

Status: completed for data integration, completed for exploratory API validation

- Resolved the earlier dataset-access blocker by integrating the official
  `ReviseQA` GitHub release directly.
- Added a native transform:
  `reviseqa_incremental`
- Added a generic-logic prompt family so `ReviseQA` is no longer forced through
  the Belief-R suppression-task prompt path.
- Generated the processed dataset from the official natural-language edit files.

Current data status:

- Original examples: 1593
- Edited pairs: 11151
- `full_info`: 11151
- `incremental_no_overturn`: 5457
- `incremental_overturn_reasoning`: 5694
- Modification types:
  `FLIP` 5810,
  `INVARIANT` 5341

Exploratory API validation:

- Added:
  `configs/reviseqa_openai_small.json`
- Added:
  `configs/reviseqa_openai_small_v2.json`
- Ran two OpenAI-backed small pilots on 12 sampled examples each
  (4 per condition, 5 systems).

Readout:

- The pipeline now runs end to end on official `ReviseQA` data.
- However, the current prompt family is still under-aligned to the task.
- In the refined `v2` run:
  `source_revision` reaches 0.50 on `full_info`,
  but `incremental_overturn_reasoning` remains 0.00 for all systems.

Interpretation:

- The remaining blocker is no longer dataset access.
- The blocker is prompt-task alignment:
  `ReviseQA` contains more formal logical semantics, including negated and
  conditional conclusions plus explicit add/remove edits.
- This dataset is currently useful as a stress test and negative result,
  but not yet as paper-grade evidence for the method claim.

Next action implied by this stage:

- Either add a more theorem-style prompt / intermediate representation for
  `ReviseQA`, or route it through a more structured reasoning interface instead
  of the current direct natural-language judgment prompt.

## Stage 11 - Belief-R Full-Info Repair

Status: completed for the first repair round

- Added a Belief-R-specific full-info prompt with mini calibration examples.
- Added structured intermediate fields:
  `premise_role` and `relation_to_prior`
- Added a confusion-analysis script for existing full-info mistakes.

Readout:

- Previous full-info confusion was dominated by `gold = c` mistakes.
- In the dedicated repair run
  `runs/belief_r_openai_full_info_repair_v1/`,
  `source_revision` full-info rose to `0.375` on 24 examples.

Interpretation:

- The prompt repair materially improved the weak reference condition.
- The model is still not fully stable on the hard `gold = c` cases, so this is
  not the final Belief-R prompt yet.

## Stage 12 - Gated Belief-R Ablation

Status: completed for the first small rerun

- Added a conservative revision gate to the `source_revision` path.
- Logged `relation_to_prior` and parsed structured outputs in traces.
- Ran a new small Belief-R ablation:
  `runs/belief_r_openai_gated_ablation_v1/`

Readout:

- `full_info` rose to `0.75` across the rerun sample.
- `source_revision` on `incremental_no_overturn` improved from `0.30` to
  `0.50`.
- `source_revision` on `incremental_overturn_reasoning` dropped from `0.90` to
  `0.625`, but remained above
  `structured_no_source = 0.125` and
  `source_no_revision = 0.25`.

Interpretation:

- The main mechanism line still survives after repair.
- The update-versus-maintain trade-off is reduced, but not eliminated.
- The next Belief-R step should be controlled scaling, not another dataset
  expansion.

## Stage 13 - Belief-R Scale-50 Validation

Status: completed

- Froze the repaired+gated Belief-R setup as the stable candidate for
  expansion.
- Ran a larger 5-system ablation with 50 examples per condition:
  `runs/belief_r_openai_gated_stable_v1_scale50/`
- Generated a detailed post-run error analysis centered on
  `relation_to_prior` and `premise_role`.

Readout:

- `source_revision`:
  - `full_info`: `0.50`
  - `incremental_no_overturn`: `0.60`
  - `incremental_overturn_reasoning`: `0.40`
- Key baselines on overturn:
  - `structured_no_source`: `0.12`
  - `source_no_revision`: `0.18`

Interpretation:

- The main mechanism claim survives scale-up.
- The remaining weakness is concentrated in relation judgment rather than
  generic infrastructure failure.
- `source_revision` still over-predicts `replace` on too many no-overturn
  cases.

## Stage 14 - Minimal Prompt-Tweak Comparison

Status: completed and rejected

- After the scale-50 analysis, tested one minimal held-out tweak only on the
  `source_revision` follow-up prompt.
- Compared frozen `v2` vs minimal-tweak `v3` on the same held-out sample.

Readout:

- `incremental_no_overturn` improved:
  `0.80 -> 1.00`
- `incremental_overturn_reasoning` collapsed:
  `0.4667 -> 0.00`

Interpretation:

- The tweak over-corrected toward conservatism.
- It should not replace the frozen repaired+gated setup.

## Stage 15 - Final Targeted Fix Validation

Status: completed and rejected

- Followed the final targeted-fix instructions strictly:
  froze the baseline, mined two error pattern memos, and changed only the
  `source_revision` follow-up prompt.
- Added:
  `analysis/belief_r_false_replace_pattern_memo.md`,
  `analysis/belief_r_missed_overturn_pattern_memo.md`,
  `prompts/llm_followup_source_revision_belief_r_v4.txt`,
  `configs/belief_r_source_revision_holdout_v4.json`
- Ran the same held-out validation slice used for the earlier `v2` vs `v3`
  comparison:
  `runs/belief_r_source_revision_holdout_v4/`

Readout:

- `full_info` stayed flat:
  `0.60`
- `incremental_no_overturn` improved modestly:
  `0.80 -> 0.8667`
- `incremental_overturn_reasoning` collapsed again:
  `0.4667 -> 0.00`
- Mechanism metrics on overturn also collapsed:
  `assistant_assumption_survival = 1.00`,
  `correction_uptake = 0.00`

Interpretation:

- The targeted repair reduced some false `replace` behavior on the no-overturn
  slice.
- But it also removed actual revision behavior on the overturn slice, which is
  the same high-level failure mode already seen in `v3`.
- This fix is rejected. The frozen repaired+gated Belief-R setup remains the
  default paper candidate.

## Stage 16 - Belief-R Commitment-Control Dataset Build

Status: completed

- Switched the project to the new training-first plan.
- Froze prompt-based `source_revision` as a baseline and started the `CIPC`
  pipeline.
- Added a deterministic Belief-R commitment-control transform and exported:
  - `data/processed/belief_r_commitment_control_train.jsonl`
  - `data/processed/belief_r_commitment_control_dev.jsonl`
  - `data/processed/belief_r_commitment_control_test.jsonl`
- Added dataset reports:
  - `analysis/belief_r_commitment_control_stats.md`
  - `analysis/belief_r_commitment_control_spotcheck.md`

Readout:

- Total pairs: `1282`
- Total examples: `2564`
- Active control labels in v1: `preserve`, `replace`
- Split sizes:
  - train: `2050`
  - dev: `254`
  - test: `260`

Interpretation:

- The repository now has a clean supervised dataset for commitment control.
- A critical conversion bug was fixed in this stage:
  splits must use `pair_id + modus`, not bare `pair_id`, because Belief-R
  reuses numeric ids across `ponens` and `tollens`.
- The project is no longer blocked on data format.

## Stage 17 - CIPC Belief-R Training Proof Of Concept

Status: completed for the first local training baseline

- Added a trainable multitask `CIPC` path:
  - `training/train_commitment_control.py`
  - `training/evaluate_commitment_control.py`
  - `src/models/commitment_control_model.py`
  - `src/eval/commitment_metrics.py`
  - `configs/train_cipc_belief_r_lora_v1.yaml`
- Because this environment lacks `torch`, `transformers`, and `peft`, the
  executable proof-of-concept is currently a NumPy multitask baseline rather
  than a true LoRA run.
- Added deterministic control-label oversampling to avoid preserve-class
  collapse.
- Ran the first end-to-end training experiment:
  `runs/cipc_belief_r_lora_v1/`

Readout:

- Test split:
  - `control_decision_accuracy = 0.8923`
  - `final_answer_accuracy = 0.7846`
  - `joint_accuracy = 0.7808`
- Test by condition:
  - `incremental_no_overturn`:
    control `0.9259`, answer `0.9259`
  - `incremental_overturn_reasoning`:
    control `0.9806`, answer `0.7961`
  - `full_info`:
    control `0.8154`, answer `0.7462`

Interpretation:

- The new training pipeline already shows the intended directional behavior:
  it prevents the no-overturn collapse while keeping strong overturn recovery.
- The strongest remaining issue is propagation lag:
  control prediction on overturn is stronger than final-answer execution.
- Same-split comparison against the frozen prompt baseline is still pending,
  and the backend still needs to be upgraded from NumPy to a true LoRA model.

## Stage 18 - Same-Split CIPC vs Frozen Prompt Baseline

Status: completed

- Built a deterministic Belief-R original-format subset aligned to the
  commitment-control `test` split:
  `data/processed/belief_r_incremental_commitment_test_subset.jsonl`
- Ran the frozen prompt-based `source_revision` baseline on the exact same
  260 examples:
  `runs/belief_r_commitment_test_prompt_source_revision/`
- Added a direct comparison script and report:
  - `analysis/compare_cipc_and_prompt_baseline.py`
  - `analysis/cipc_vs_prompt_source_revision_test_report.md`
  - `paper_assets/cipc_vs_prompt_source_revision_test_metrics.csv`

Readout:

- Overall on the same test split:
  - `CIPC`:
    control `0.8923`, answer `0.7846`, joint `0.7808`
  - frozen prompt `source_revision`:
    control proxy `0.3615`, answer `0.3615`, joint `0.3615`
- On `incremental_overturn_reasoning`:
  - `CIPC` answer `0.7961`
  - frozen prompt `source_revision` answer `0.2718`
- On `incremental_no_overturn`:
  - `CIPC` answer `0.9259`
  - frozen prompt `source_revision` answer `0.5926`

Interpretation:

- The training-based method clearly outperforms the frozen prompt baseline on
  the same Belief-R split.
- The strongest gain is on overturn handling:
  `early_commitment_persistence` drops from `0.7282` to `0.2039`,
  while `late_evidence_takeover` rises from `0.2718` to `0.7961`.
- This is the first clean same-split evidence that the project should remain
  on the training path rather than returning to prompt tuning.

## Stage 19 - HF/LoRA Backend Smoke Run

Status: completed for the first smoke test

- Installed the transformer training stack and corrected the PyTorch install to
  the CUDA build.
- Verified the local machine exposes:
  - `torch 2.11.0+cu128`
  - CUDA available on `NVIDIA GeForce RTX 4080 SUPER`
- Added a real HuggingFace + PEFT training backend:
  - `src/models/hf_commitment_control_model.py`
  - `training/train_commitment_control_hf.py`
  - `configs/train_cipc_belief_r_qwen05b_lora_smoke.json`
- Ran a first LoRA smoke experiment with
  `Qwen/Qwen2.5-0.5B-Instruct` on a reduced subset:
  `runs/cipc_belief_r_qwen05b_lora_smoke/`

Readout:

- Test split on the reduced smoke slice:
  - `control_decision_accuracy = 0.5312`
  - `final_answer_accuracy = 0.5469`
  - `joint_accuracy = 0.3281`
- `incremental_overturn_reasoning` answer accuracy on the smoke slice:
  `0.7000`
- `incremental_no_overturn` answer accuracy on the smoke slice:
  `0.2917`

Interpretation:

- The important success in this stage is infrastructural, not SOTA quality:
  the project now has a real GPU-backed HF/LoRA training path that can load a
  model, attach LoRA, train, and emit the same metrics artifacts as the NumPy
  baseline.
- The first smoke run is undertrained and not yet competitive with the stronger
  NumPy multitask baseline.
- The next training step is no longer "make HF work"; it is "scale the HF run
  to a sensible subset/full split and tune optimization so no-overturn does not
  collapse."

## Stage 20 - HF/LoRA Balanced Local Pilot

Status: completed

- Upgraded the HF trainer to handle the main class-imbalance problem:
  - deterministic subset selection instead of taking the first rows only
  - weighted sampling for `control_label`
  - answer-label class weights
- Added a more serious local config:
  `configs/train_cipc_belief_r_qwen05b_lora_balanced_full_v1.json`
- Ran a full-train local LoRA pilot with
  `Qwen/Qwen2.5-0.5B-Instruct`:
  `runs/cipc_belief_r_qwen05b_lora_balanced_full_v1/`
- Added a richer run report:
  `analysis/cipc_belief_r_qwen05b_lora_balanced_full_v1_report.md`

Readout:

- Test split:
  - `control_decision_accuracy = 0.6769`
  - `final_answer_accuracy = 0.6462`
  - `joint_accuracy = 0.6462`
- Test by condition:
  - `incremental_no_overturn`:
    control `0.9259`, answer `0.9259`
  - `incremental_overturn_reasoning`:
    control `0.6117`, answer `0.5728`
  - `full_info`:
    control `0.6769`, answer `0.6462`

Interpretation:

- The HF line has now moved beyond a pure smoke test.
- Most importantly, the earlier HF `no-overturn` collapse has been fixed.
- The remaining weakness is now concentrated where it should be:
  overturn propagation still lags, with too much early-commitment persistence.
- This run already beats the frozen prompt baseline, but still trails the
  stronger local NumPy multitask baseline.

## Stage 21 - HF Balanced vs Frozen Prompt Baseline

Status: completed

- Compared the new HF balanced full run against the frozen prompt
  `source_revision` baseline on the same Belief-R test split.
- Added:
  - `analysis/cipc_qwen05b_balanced_full_vs_prompt_test_report.md`
  - `paper_assets/cipc_qwen05b_balanced_full_vs_prompt_test_metrics.csv`

Readout:

- Overall test answer accuracy:
  - HF balanced full `CIPC`: `0.6462`
  - frozen prompt `source_revision`: `0.3615`
- Overturn answer accuracy:
  - HF balanced full `CIPC`: `0.5728`
  - frozen prompt `source_revision`: `0.2718`
- Overturn early-commitment persistence:
  - HF balanced full `CIPC`: `0.4272`
  - frozen prompt `source_revision`: `0.7282`

Interpretation:

- The local HF LoRA path is now clearly better than the frozen prompt baseline.
- The project's main remaining comparison is no longer against prompt tuning;
  it is against the stronger local NumPy training baseline and, later, against
  larger real-model training runs.
