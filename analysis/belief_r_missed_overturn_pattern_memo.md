# Belief-R Missed-Overturn Pattern Memo

Source:

- `runs/belief_r_openai_gated_stable_v1_scale50/`
- filtered to `source_revision` failures on `incremental_overturn_reasoning`

## Summary

- Total missed overturns: `30`
- Relation buckets:
  - `confirm`: `15`
  - `replace`: `13`
  - `unrelated`: `2`
- Premise-role buckets:
  - `alternative_pathway`: `14`
  - `extra_requirement`: `11`
  - `unclear`: `3`
  - `contradiction`: `2`

## Dominant Semantic Patterns

### Pattern 1: Same scenario, tighter restatement, misread as independent route

Examples:

- makes a single enemy -> resolves every challenge
  vs
  deems Jessica's competitive spirit a hurdle -> resolves every challenge
- lobbies to become leader of a garden club
  vs
  is elected chief coordinator of a horticultural society

Observed failure:

- The model labels these `confirm` / `alternative_pathway`.

Why this matters:

- These are often closer to tighter versions or contextual restatements of the
  same route than to genuinely separate routes.

### Pattern 2: Outcome-supporting context that may change necessity

Examples:

- believes the pen is a rare collectible -> high price / examine the pen
- invitation to a red-attire event -> wears the red scarf
- crosses finish line before Jessica -> wins the trophy

Observed failure:

- The model often preserves the old commitment because the consequent still
  sounds plausible.

Why this matters:

- The task is not asking whether the consequent is plausible in general.
- It is asking whether the original commitment still necessarily follows.

### Pattern 3: Parsing / focus misses

Examples:

- some cases are labeled `unrelated` or `unclear` even though the update does
  bear on the commitment

Why this matters:

- The prompt should keep the model focused on whether the new premise changes
  support for the earlier commitment, not merely whether it mentions the same
  nouns.

## Practical Prompt Implication

- The next prompt should preserve the ability to call `replace` when a new
  premise functions as a tighter version of the same route or changes whether
  the old route still warrants certainty.
- This should be stated without pushing the model toward blanket `confirm`.
