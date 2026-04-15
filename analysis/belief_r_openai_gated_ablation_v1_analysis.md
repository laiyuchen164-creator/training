# Belief-R Run Analysis: `belief_r_openai_gated_ablation_v1`

## Aggregate Metrics

| system | condition | n | accuracy | avg_prompt_tokens | avg_api_calls | assistant_assumption_survival | correction_uptake |
|---|---|---:|---:|---:|---:|---:|---:|
| raw_history | full_info | 8 | 0.75 | 1028.0 | 1.0 | 0.0 | 0.0 |
| raw_history | incremental_no_overturn | 8 | 0.625 | 1377.5 | 2.0 | 0.0 | 0.0 |
| raw_history | incremental_overturn_reasoning | 8 | 0.25 | 1377.0 | 2.0 | 0.75 | 0.25 |
| running_summary | full_info | 8 | 0.75 | 1028.0 | 1.0 | 0.0 | 0.0 |
| running_summary | incremental_no_overturn | 8 | 0.5 | 1249.5 | 2.0 | 0.0 | 0.0 |
| running_summary | incremental_overturn_reasoning | 8 | 0.5 | 1244.375 | 2.0 | 0.5 | 0.5 |
| source_no_revision | full_info | 8 | 0.75 | 1028.0 | 1.0 | 0.0 | 0.0 |
| source_no_revision | incremental_no_overturn | 8 | 0.75 | 1228.625 | 2.0 | 0.0 | 0.0 |
| source_no_revision | incremental_overturn_reasoning | 8 | 0.25 | 1234.125 | 2.0 | 0.75 | 0.25 |
| source_revision | full_info | 8 | 0.75 | 1028.0 | 1.0 | 0.0 | 0.0 |
| source_revision | incremental_no_overturn | 8 | 0.5 | 1305.625 | 2.0 | 0.0 | 0.0 |
| source_revision | incremental_overturn_reasoning | 8 | 0.625 | 1311.125 | 2.0 | 0.375 | 0.625 |
| structured_no_source | full_info | 8 | 0.75 | 1028.0 | 1.0 | 0.0 | 0.0 |
| structured_no_source | incremental_no_overturn | 8 | 0.5 | 1192.625 | 2.0 | 0.0 | 0.0 |
| structured_no_source | incremental_overturn_reasoning | 8 | 0.125 | 1198.125 | 2.0 | 0.875 | 0.125 |

## Final-Label Confusions

### raw_history / full_info

| gold | predicted | count |
|---|---|---:|
| c | a | 2 |
| c | c | 6 |

### raw_history / incremental_no_overturn

| gold | predicted | count |
|---|---|---:|
| a | a | 4 |
| b | b | 1 |
| b | c | 3 |

### raw_history / incremental_overturn_reasoning

| gold | predicted | count |
|---|---|---:|
| c | a | 3 |
| c | b | 3 |
| c | c | 2 |

### running_summary / full_info

| gold | predicted | count |
|---|---|---:|
| c | a | 2 |
| c | c | 6 |

### running_summary / incremental_no_overturn

| gold | predicted | count |
|---|---|---:|
| a | a | 4 |
| b | c | 4 |

### running_summary / incremental_overturn_reasoning

| gold | predicted | count |
|---|---|---:|
| c | a | 3 |
| c | b | 1 |
| c | c | 4 |

### source_no_revision / full_info

| gold | predicted | count |
|---|---|---:|
| c | a | 2 |
| c | c | 6 |

### source_no_revision / incremental_no_overturn

| gold | predicted | count |
|---|---|---:|
| a | a | 4 |
| b | b | 2 |
| b | c | 2 |

### source_no_revision / incremental_overturn_reasoning

| gold | predicted | count |
|---|---|---:|
| c | a | 3 |
| c | b | 3 |
| c | c | 2 |

### source_revision / full_info

| gold | predicted | count |
|---|---|---:|
| c | a | 2 |
| c | c | 6 |

### source_revision / incremental_no_overturn

| gold | predicted | count |
|---|---|---:|
| a | a | 4 |
| b | c | 4 |

### source_revision / incremental_overturn_reasoning

| gold | predicted | count |
|---|---|---:|
| c | a | 3 |
| c | c | 5 |

### structured_no_source / full_info

| gold | predicted | count |
|---|---|---:|
| c | a | 2 |
| c | c | 6 |

### structured_no_source / incremental_no_overturn

| gold | predicted | count |
|---|---|---:|
| a | a | 4 |
| b | c | 4 |

### structured_no_source / incremental_overturn_reasoning

| gold | predicted | count |
|---|---|---:|
| c | a | 3 |
| c | b | 4 |
| c | c | 1 |

## Relation-To-Prior Distributions

### source_no_revision

| condition | relation_to_prior | count |
|---|---|---:|
| incremental_no_overturn | confirm | 5 |
| incremental_no_overturn | replace | 2 |
| incremental_no_overturn | unrelated | 1 |
| incremental_overturn_reasoning | confirm | 4 |
| incremental_overturn_reasoning | replace | 3 |
| incremental_overturn_reasoning | unrelated | 1 |

### source_revision

| condition | relation_to_prior | count |
|---|---|---:|
| incremental_no_overturn | confirm | 3 |
| incremental_no_overturn | replace | 5 |
| incremental_overturn_reasoning | confirm | 2 |
| incremental_overturn_reasoning | replace | 6 |

## Premise-Role Distributions

### source_no_revision

| condition | premise_role | count |
|---|---|---:|
| incremental_no_overturn | alternative_pathway | 7 |
| incremental_no_overturn | extra_requirement | 1 |
| incremental_overturn_reasoning | alternative_pathway | 6 |
| incremental_overturn_reasoning | contradiction | 1 |
| incremental_overturn_reasoning | extra_requirement | 1 |

### source_revision

| condition | premise_role | count |
|---|---|---:|
| incremental_no_overturn | alternative_pathway | 3 |
| incremental_no_overturn | extra_requirement | 5 |
| incremental_overturn_reasoning | alternative_pathway | 2 |
| incremental_overturn_reasoning | extra_requirement | 6 |

## Source-Revision No-Overturn Failures

- Total failures: 4
- wrong_relation_to_prior: 4

Representative examples:
- `belief_r::125-strong::tollens::incremental`
  gold `b -> b`, pred `b -> c`
  relation_to_prior `replace`, premise_role `extra_requirement`, model_prediction `c`
  raw `{"label":"c","premise_role":"extra_requirement","relation_to_prior":"replace","rationale":"The new rule adds an alternative way for Jessica to allow use of the item, so her not allowing it no longer forces that John did not give cash. The prior conclusion is w`
- `belief_r::654-strong::tollens::incremental`
  gold `b -> b`, pred `b -> c`
  relation_to_prior `replace`, premise_role `extra_requirement`, model_prediction `c`
  raw `{"label":"c","premise_role":"extra_requirement","relation_to_prior":"replace","rationale":"The new premise gives an alternative way for the leak to be fixed (Jessica informs maintenance), so John’s noticing/reporting is no longer necessary. The prior negative `
- `belief_r::308-strong::tollens::incremental`
  gold `b -> b`, pred `b -> c`
  relation_to_prior `replace`, premise_role `extra_requirement`, model_prediction `c`
  raw `{"label":"c","premise_role":"extra_requirement","relation_to_prior":"replace","rationale":"The new rule adds a further condition for Jessica having to collect the pen, so the earlier inference that John did not place the pen is no longer forced; the conclusion`
- `belief_r::432-strong::tollens::incremental`
  gold `b -> b`, pred `b -> c`
  relation_to_prior `replace`, premise_role `extra_requirement`, model_prediction `c`
  raw `{"label":"c","premise_role":"extra_requirement","relation_to_prior":"replace","rationale":"The new premise gives an alternative route to Jessica feeling comfortable, so the earlier inference that John must not keep 70°F is no longer forced. John’s temperature `

## Source-Revision Missed Overturns

- Total missed overturns: 3
- label_semantic_confusion: 1
- wrong_relation_to_prior: 2

Representative examples:
- `belief_r::779-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `a`
  raw `{"label":"a","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new premise gives another sufficient condition for the same conclusion, so it confirms the existing commitment that John resolves every challenge to savor their vi`
- `belief_r::16-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `a`
  raw `{"label":"a","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new premise gives another sufficient condition for the same conclusion, so it supports the existing commitment that Jessica will follow John's gardening plans."}`
- `belief_r::586-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `replace`, premise_role `extra_requirement`, model_prediction `a`
  raw `{"label":"a","premise_role":"extra_requirement","relation_to_prior":"replace","rationale":"The new rule adds an extra sufficient condition for setting a high price, but it does not negate the original John-asks rule or the examination fact. So the prior conclu`

## Source-No-Revision Missed Overturns

- Total missed overturns: 6
- label_semantic_confusion: 1
- wrong_relation_to_prior: 5

Representative examples:
- `belief_r::882-strong::tollens::incremental`
  gold `b -> c`, pred `b -> b`
  relation_to_prior `unrelated`, premise_role `alternative_pathway`, model_prediction `b`
  raw `{"label":"b","premise_role":"alternative_pathway","relation_to_prior":"unrelated","rationale":"The new rule gives a different sufficient condition for finishing faster, but it does not affect the existing inference from not finishing faster to not turning off `
- `belief_r::1-strong::tollens::incremental`
  gold `b -> c`, pred `b -> b`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `b`
  raw `{"label":"b","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new premise gives a second route to smugness, but it does not undermine the existing inference that if John bought the rod he would feel smug; combined with John d`
- `belief_r::779-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `a`
  raw `{"label":"a","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new premise gives another sufficient condition for the same conclusion, so it supports the existing commitment rather than changing it."}`
- `belief_r::16-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `a`
  raw `{"label":"a","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new premise gives another sufficient route to the same conclusion: if John is elected chief coordinator, Jessica will follow his gardening plans. It supports, rath`
- `belief_r::471-strong::tollens::incremental`
  gold `b -> c`, pred `b -> b`
  relation_to_prior `replace`, premise_role `contradiction`, model_prediction `b`
  raw `{"label":"b","premise_role":"contradiction","relation_to_prior":"replace","rationale":"The new premise gives a separate way for Jessica to remain with the club, which weakens the earlier inference from John attending to Jessica remaining. Since Jessica still w`
- `belief_r::586-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `a`
  raw `{"label":"a","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new premise gives another sufficient condition for the same conclusion, so it confirms the existing commitment rather than weakening or replacing it."}`

