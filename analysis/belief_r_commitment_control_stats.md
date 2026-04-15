# Belief-R Commitment-Control Dataset Stats

## Summary

- Total pairs: `1282`
- Total examples: `2564`
- Active control labels: `preserve, replace`
- Skipped pairs: `0`

## Overall Distribution

- Conditions: `{'full_info': 1282, 'incremental_overturn_reasoning': 1018, 'incremental_no_overturn': 264}`
- Control labels: `{'replace': 2036, 'preserve': 528}`
- Final answers: `{'c': 2036, 'b': 264, 'a': 264}`

## Split Breakdown

### train

- Pairs: `1025` (`0.7995` of total)
- Examples: `2050` (`0.7995` of total)
- Conditions: `{'full_info': 1025, 'incremental_overturn_reasoning': 814, 'incremental_no_overturn': 211}`
- Control labels: `{'replace': 1628, 'preserve': 422}`
- Final answers: `{'c': 1628, 'b': 210, 'a': 212}`

### dev

- Pairs: `127` (`0.0991` of total)
- Examples: `254` (`0.0991` of total)
- Conditions: `{'full_info': 127, 'incremental_overturn_reasoning': 101, 'incremental_no_overturn': 26}`
- Control labels: `{'replace': 202, 'preserve': 52}`
- Final answers: `{'c': 202, 'a': 28, 'b': 24}`

### test

- Pairs: `130` (`0.1014` of total)
- Examples: `260` (`0.1014` of total)
- Conditions: `{'full_info': 130, 'incremental_no_overturn': 27, 'incremental_overturn_reasoning': 103}`
- Control labels: `{'preserve': 54, 'replace': 206}`
- Final answers: `{'a': 24, 'c': 206, 'b': 30}`
