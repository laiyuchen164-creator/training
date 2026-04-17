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

## Current Working Conclusion

- The core problem still looks like an objective/representation-geometry issue rather than checkpoint selection or sampler balance.
- The conditional propagation redesign is directionally meaningful but not yet sufficient in its current weight-only form.
- The next main-line step should continue to focus on reshaping `L_prop`, especially preserve-side influence and preserve/replace aggregation, rather than repeating simple weight sweeps.

## Next Update Template

When new context appears, append:

- date
- current objective
- important decisions
- assumptions
- open questions
- next steps
