# Qwen 0.5B Consistency Redesign Plan

## Why Redesign Is Needed

The first minimal consistency implementation established two things:

1. consistency-style supervision can move the maintain/overturn trade-off
2. the current formulation is unstable

Observed behavior:

- `consistency_v1` repaired no-overturn but gave up too much overturn
- `consistency_v2` swung back toward aggressive replacement and collapsed
  no-overturn

So the issue is no longer whether consistency matters.

The issue is that the current consistency loss is not the right shape.

## Current Formulation

The current minimal implementation tries to align:

- `P(control = preserve)`
- with the answer head's probability on the early commitment label

This is simple and cheap, but it has a structural weakness:

- it only supervises the preserve side directly
- it does not explicitly shape the replace side beyond what falls out
  indirectly
- it couples the two heads in a way that can flip sharply with small
  weight changes

In practice, that appears to make the optimization brittle.

## Redesign Goal

Keep the active setup fixed:

- same Belief-R split
- same seed
- same Qwen 0.5B HF/LoRA base
- same current trainer and metrics

Change only the consistency definition so it better matches the dataset
semantics.

## Dataset Semantics To Respect

Current Belief-R training records behave as follows:

- `preserve` means the final answer should equal the early commitment label
- `replace` means the final answer should differ from the early commitment
  label

In the current v1 Belief-R path:

- `weaken` is present in the label space
- but not active in the actual data

So the redesign should focus on the binary active semantics:

- preserve -> stay on early label
- replace -> move off early label

## Proposed Redesign

### Option A: Direct masked consistency loss

Preferred next design.

For each example:

- compute `p_early`, the answer-head probability on the early label
- compute `p_change = 1 - p_early`

Then use control labels directly:

- if gold control is `preserve`, penalize low `p_early`
- if gold control is `replace`, penalize low `p_change`

Equivalent implementation choices:

- binary cross-entropy on `p_change` with target
  `0` for preserve and `1` for replace
- or a two-sided log loss:
  - preserve: `-log(p_early)`
  - replace: `-log(1 - p_early)`

Why this is better:

- it supervises both sides directly
- it is anchored to the actual dataset semantics
- it avoids the extra instability of matching one probabilistic head to
  another probabilistic head

### Option B: Margin consistency loss

Backup if Option A still swings too sharply.

For preserve examples:

- require answer logit on the early label to exceed the best alternative by
  at least margin `m`

For replace examples:

- require the best non-early answer logit to exceed the early-label logit by
  at least margin `m`

Why it may help:

- it enforces directional preference rather than probability matching
- it can be less brittle if the answer distribution is overconfident

## Recommended Next Experiment

Implement Option A first.

Keep everything else fixed to Runpod `highrank_v1`:

- `answer_loss_weight = 1.0`
- same LoRA config
- same epochs
- same selection rule
- same sampler

Use a small consistency weight first, because the current runs already showed
that strong consistency can over-correct.

Suggested first config:

- `consistency_loss_weight = 0.05`

Why `0.05`:

- `0.2` was clearly too strong
- `0.1` was already unstable under the old formulation
- the redesigned loss should start conservatively

## Success Criterion

The redesigned consistency term is worth keeping only if it can beat the
current Runpod `highrank_v1` frontier in the intended sense:

- improve no-overturn relative to `0.6667`
- without giving back too much overturn relative to `0.9029`

Practical success threshold for the first redesigned run:

- no-overturn materially above `0.6667`
- overturn still clearly above the NumPy baseline `0.7961`

If it fails that bar, the redesign is not yet good enough.

## Implementation Notes

Keep the code change minimal:

- do not change dataset files
- do not change prediction format
- do not change evaluation code

Only change:

- how consistency loss is computed inside the current model forward pass

This keeps the attribution clean:

- if behavior changes, it came from the new consistency definition

