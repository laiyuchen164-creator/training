# Belief-R False-Replace Pattern Memo

Source:

- `runs/belief_r_openai_gated_stable_v1_scale50/`
- filtered to `source_revision` failures on `incremental_no_overturn`

## Summary

- Total failures: `20`
- `relation_to_prior = replace`: `20 / 20`
- `premise_role = extra_requirement`: `20 / 20`
- Modus:
  - `tollens`: `18`
  - `ponens`: `2`

## Dominant Semantic Patterns

### Pattern 1: Separate-agent route to the same outcome

Examples:

- helping Jessica with chores -> Jessica allows item use
- Jessica informs maintenance -> leak gets fixed
- Jessica admits unauthorized access -> John seeks to recover funds

Observed failure:

- The model treats these as if they narrow the original route.
- In the error traces, they are labeled `extra_requirement` + `replace`.

Why this matters:

- These updates usually introduce another coexisting cause of the same
  consequent, not a change to the truth conditions of the original commitment.

### Pattern 2: External policy / environment / situational trigger

Examples:

- pen lies within Jessica's reach -> Jessica must collect the pen
- report lacks essential details -> Jessica provides additional data
- Jessica is responsible for handing out awards -> Jessica distributes the certificates

Observed failure:

- The model reads the new trigger as replacing the original route rather than as
  another route or background condition.

Why this matters:

- These updates are often independent context triggers, not direct revisions of
  the earlier commitment.

### Pattern 3: Richer paraphrase that still leaves the original route intact

Examples:

- pleasant warmth -> Jessica feels comfortable
- algebra tutoring sessions -> John understands algebra and returns home
- ceiling water stains + heavy rainfall -> Jessica agrees to fix the roof

Observed failure:

- The model over-interprets added detail as a change in necessity.

Why this matters:

- The current prompt is too eager to map added detail to `replace`.

## Practical Prompt Implication

- The next prompt should explicitly warn that a new premise can describe another
  route, trigger, or richer scenario without changing whether the original
  commitment is still warranted.
- This should be framed as a local correction to false `replace`, not as a
  global conservatism shift.
