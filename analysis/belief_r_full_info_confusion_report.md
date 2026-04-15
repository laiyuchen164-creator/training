# Belief-R Full-Info Confusion Report

- Source run: `belief_r_api_pilot_openai_medium_ablation`
- Total full-info predictions: 50

## Confusion Counts

| gold | predicted | count |
|---|---|---:|
| b | b | 5 |
| c | a | 14 |
| c | b | 28 |
| c | c | 3 |

## Sample Mistakes

### gold `c` -> predicted `a`

- Example: `belief_r::99-strong::ponens::full_info`
  Raw output: `{"label":"a","rationale":"Premise 1 and 2 directly imply John enjoys signing autographs. Premise 3 is just another independent route to the same outcome, not a limiting condition."}`
- Example: `belief_r::402-strong::ponens::full_info`
  Raw output: `{"label":"a","rationale":"Premise 1 plus Premise 2 already guarantee that Jessica will receive the same salary. Premise 3 is just another sufficient route, not a limiting condition, so the conclusion is maintained."}`
- Example: `belief_r::942-strong::ponens::full_info`
  Raw output: `{"label":"a","rationale":"Premise 1 and 2 already imply Jessica raises their eyebrows. Premise 3 is just an additional independent trigger for the same outcome, not a requirement that weakens the inference."}`

### gold `c` -> predicted `b`

- Example: `belief_r::629-strong::tollens::full_info`
  Raw output: `{"label":"b","rationale":"Premise 1 says wise organization would imply chores completed. Premise 2 says chores are not completed, so that initial condition is blocked. Premise 3 describes an additional way chores could b`
- Example: `belief_r::450-strong::tollens::full_info`
  Raw output: `{"label":"b","rationale":"Premise 1 says organizing categories would imply Jessica appreciates him. Premise 2 says she does not appreciate him, so that initial inference is blocked and John cannot have organized the task`
- Example: `belief_r::442-strong::tollens::full_info`
  Raw output: `{"label":"b","rationale":"Premise 1 says learning how to maintain pressure would lead to avoiding a pipe burst; Premise 2 says they do not avoid a pipe burst, so John does not learn that. Premise 3 gives a different suff`

