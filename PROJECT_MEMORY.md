# Project Memory

Last updated: 2026-04-17 UTC

## Collaboration Rules

- This research project is persistent and continuous.
- Conversation context must be saved to the workspace so future sessions on a new server can resume without restarting from zero.
- After important discussions, decisions, plans, assumptions, and constraints, update this file.
- Treat this file as the durable project memory across servers and sessions.

## Current User Requirements

- The user wants all important conversation context preserved.
- The user does not want to repeat the project background when switching to a new server.

## Session Notes

### 2026-04-17

- Established the rule that project context should be persisted in this repository.
- Created `PROJECT_MEMORY.md` as the durable handoff document for future sessions.
- Checked overall project progress and GitHub sync state.
- Confirmed remote repository: `origin = https://github.com/laiyuchen164-creator/training.git`.
- Confirmed local branch status at check time: `main` was ahead of `origin/main` by 1 commit, plus untracked `PROJECT_MEMORY.md`.
- Confirmed latest pushed GitHub commit was `d762144` (`Add Runpod experiment reports and HF tradeoff diagnostics`).
- Confirmed latest local unpushed commit was `ec2cc3b` (`Add conditional consistency HF experiments`).

## Current Research Status

- Active main line remains Belief-R only, training-based CIPC only, HF/LoRA only, fixed split/seed.
- Current best aggressive Runpod frontier point remains `highrank_v1`.
- `answer_loss_weight` was the cleanest stable trade-off lever in the earlier Runpod phase, but did not beat `highrank_v1` overall.
- Simple checkpoint reselection did not explain the trade-off.
- Simple condition-sampling repair did not fix the trade-off.
- Minimal consistency-loss variants showed the trade-off is controllable in principle, but the formulation was unstable.

## Latest Local Experiment Status

- `conditional_consistency_v1` introduced a conditional propagation loss and produced a stronger aggressive point:
  - overall answer: `0.8923`
  - overturn answer: `0.9806`
  - no-overturn answer: `0.5556`
- Interpretation: the propagation objective is active, but `v1` was too replace-heavy and did not satisfy the main repair goal.
- `conditional_consistency_v2` reduced `lambda_prop` and the replace margin:
  - overall answer: `0.8154`
  - overturn answer: `0.8738`
  - no-overturn answer: `0.5926`
- Interpretation: `v2` moved in the expected direction relative to `v1`, but still failed to exceed `highrank_v1` on either overturn or no-overturn, so it is not a new frontier point.
- `split_conditional_consistency_v1` changed the propagation objective again by separating preserve-side and replace-side aggregation and weighting:
  - overall answer: `0.8462`
  - overturn answer: `0.9223`
  - no-overturn answer: `0.5556`
- Interpretation: separate aggregation recovered aggressive overturn behavior but did not protect no-overturn. It did not beat `highrank_v1` on the main repair criterion and is not the frontier repair.
- `preserve_margin_conditional_consistency_v1` kept split aggregation but replaced preserve-side CE with a preserve-side margin:
  - overall answer: `0.8077`
  - overturn answer: `0.8350`
  - no-overturn answer: `0.7037`
- Interpretation: preserve-side structural constraints matter. This was the first consistency-style run to improve no-overturn above `highrank_v1`, but it gave back too much overturn to become the new frontier point.
- `preserve_margin_conditional_consistency_v2` weakened only the preserve-side margin:
  - `preserve_margin_m: 0.5 -> 0.3`
  - overall answer: `0.8154`
  - overturn answer: `0.8544`
  - no-overturn answer: `0.6667`
- Interpretation: preserve-side margin is a real smooth control axis. Weakening the margin recovered some overturn and overall accuracy, but removed the no-overturn gain over `highrank_v1`.
- `preserve_hybrid_conditional_consistency_v1` changed preserve-side to `CE_to_early + small preserve margin` on top of the split design:
  - `beta_preserve_margin = 0.1`
  - `preserve_margin_m = 0.3`
  - overall answer: `0.8615`
  - overturn answer: `0.9223`
  - no-overturn answer: `0.6296`
- Interpretation: the hybrid preserve loss recovered aggressive performance and beat `highrank_v1` on both overall and overturn, but it did not preserve the maintain-side gain from pure preserve margin.
- `preserve_hybrid_conditional_consistency_v2` increased only the hybrid preserve margin weight:
  - `beta_preserve_margin: 0.1 -> 0.2`
  - overall answer: `0.8615`
  - overturn answer: `0.9515`
  - no-overturn answer: `0.5185`
- Interpretation: strengthening the margin term inside the current hybrid preserve form did not repair maintain cases; it pushed the system further toward aggressive overturn behavior.
- Added a frozen external-model baseline using `Qwen/Qwen2.5-0.5B-Instruct` as a zero-shot multiple-choice scorer on the same Belief-R commitment-control split.
- Frozen Qwen MC baseline results:
  - train overall answer: `0.7200`
  - dev overall answer: `0.6929`
  - test overall answer: `0.7308`
  - test overturn answer: `0.8835`
  - test no-overturn answer: `0.1481`
- Interpretation: the frozen Qwen baseline is much stronger than the old frozen prompt baseline on overturn, but it nearly collapses on no-overturn by over-predicting `replace` / answer `c`. It is an informative external reference point, but not competitive with the trained CIPC line on the actual trade-off objective.
- Added direct external API baselines that do not reuse the old `source_revision` prompt framework. These baselines directly answer each test example with final label `a/b/c` and are then mapped back into commitment metrics.
- Direct API baseline results on `belief_r_commitment_control_test.jsonl`:
  - OpenAI `gpt-5.4-mini`: overall `0.2077`, overturn `0.0000`, no-overturn `1.0000`
  - DeepSeek `deepseek-chat`: overall `0.2192`, overturn `0.0097`, no-overturn `1.0000`
- Interpretation: both direct API baselines collapse toward always preserving the early commitment and almost never predict the required `c` answer on overturn cases. This is the opposite extreme from the frozen local Qwen baseline and further supports that the key challenge is calibrated preserve-vs-revise control rather than generic language-model strength.

## Current Working Conclusion

- The core problem still looks like an objective/representation-geometry issue rather than checkpoint selection or sampler balance.
- The conditional propagation redesign is directionally meaningful but not yet sufficient in its current weight-only form.
- Separate preserve/replace aggregation alone is insufficient; preserve-side supervision likely needs a stronger structural constraint than plain CE to the early label.
- Preserve-side margin is a real control lever: it can recover no-overturn above `highrank_v1`, but the current setting over-corrects and suppresses overturn too much.
- Reducing preserve-side margin from `0.5` to `0.3` predictably shifts the frontier back toward overturn while giving up maintain-side gains. This confirms smooth controllability, but still does not produce a point that dominates `highrank_v1`.
- Adding CE back into preserve-side supervision creates a stronger aggressive point, but the current small hybrid margin is too weak to keep no-overturn above `highrank_v1`.
- In the current hybrid preserve formulation, increasing the preserve margin weight from `0.1` to `0.2` makes no-overturn much worse while further increasing overturn. This suggests the problem is now objective form, not just scalar strength.
- The next main-line step should continue to focus on reshaping `L_prop`, especially preserve-side influence and preserve/replace aggregation, rather than repeating simple weight sweeps.

## Next Update Template

When new context appears, append:

- date
- current objective
- important decisions
- assumptions
- open questions
- next steps
