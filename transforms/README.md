# Transforms

The first transform converts Belief-R into a unified incremental JSONL format.

The second transform builds an explicit synthetic revision dataset from
ATOMIC. It uses:

- an initial rule `p -> q`
- an observed antecedent `p`
- either an alternative rule that preserves `q`, or
- an explicit correction that revises the earlier rule into a stricter form
  requiring an extra condition.

Derived conditions:

- `full_info`: all premises shown at once.
- `incremental_no_overturn`: additional evidence arrives later but the correct
  answer stays the same.
- `incremental_overturn_reasoning`: additional evidence arrives later and the
  correct answer changes from the initial basic inference to `c`.

The first milestone uses the strong paired subset because it supports clean
pairing between `basic_time_t.csv` and `queries_time_t1.csv`.

The ATOMIC transform is intentionally easier and more explicit than Belief-R.
It is useful as a second-dataset sanity check because it tests revision when
the update cue is directly stated instead of implicitly inferred through
commonsense.
