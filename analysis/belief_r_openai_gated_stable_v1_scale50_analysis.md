# Belief-R Run Analysis: `belief_r_openai_gated_stable_v1_scale50`

## Aggregate Metrics

| system | condition | n | accuracy | avg_prompt_tokens | avg_api_calls | assistant_assumption_survival | correction_uptake |
|---|---|---:|---:|---:|---:|---:|---:|
| raw_history | full_info | 50 | 0.52 | 1041.34 | 1.0 | 0.0 | 0.0 |
| raw_history | incremental_no_overturn | 50 | 0.82 | 1338.6 | 2.0 | 0.0 | 0.0 |
| raw_history | incremental_overturn_reasoning | 50 | 0.22 | 1342.38 | 2.0 | 0.78 | 0.22 |
| running_summary | full_info | 50 | 0.5 | 1041.34 | 1.0 | 0.0 | 0.0 |
| running_summary | incremental_no_overturn | 50 | 0.6 | 1216.88 | 2.0 | 0.0 | 0.0 |
| running_summary | incremental_overturn_reasoning | 50 | 0.46 | 1219.66 | 2.0 | 0.54 | 0.46 |
| source_no_revision | full_info | 50 | 0.52 | 1041.34 | 1.0 | 0.0 | 0.0 |
| source_no_revision | incremental_no_overturn | 50 | 0.86 | 1203.96 | 2.0 | 0.0 | 0.0 |
| source_no_revision | incremental_overturn_reasoning | 50 | 0.18 | 1208.82 | 2.0 | 0.82 | 0.18 |
| source_revision | full_info | 50 | 0.5 | 1041.34 | 1.0 | 0.0 | 0.0 |
| source_revision | incremental_no_overturn | 50 | 0.6 | 1281.0 | 2.0 | 0.0 | 0.0 |
| source_revision | incremental_overturn_reasoning | 50 | 0.4 | 1285.78 | 2.0 | 0.6 | 0.4 |
| structured_no_source | full_info | 50 | 0.5 | 1041.34 | 1.0 | 0.0 | 0.0 |
| structured_no_source | incremental_no_overturn | 50 | 0.88 | 1167.96 | 2.0 | 0.0 | 0.0 |
| structured_no_source | incremental_overturn_reasoning | 50 | 0.12 | 1172.82 | 2.0 | 0.88 | 0.12 |

## Final-Label Confusions

### raw_history / full_info

| gold | predicted | count |
|---|---|---:|
| a | a | 3 |
| b | c | 5 |
| c | a | 19 |
| c | c | 23 |

### raw_history / incremental_no_overturn

| gold | predicted | count |
|---|---|---:|
| a | a | 26 |
| b | b | 15 |
| b | c | 9 |

### raw_history / incremental_overturn_reasoning

| gold | predicted | count |
|---|---|---:|
| c | a | 21 |
| c | b | 18 |
| c | c | 11 |

### running_summary / full_info

| gold | predicted | count |
|---|---|---:|
| a | a | 3 |
| b | c | 5 |
| c | a | 19 |
| c | b | 1 |
| c | c | 22 |

### running_summary / incremental_no_overturn

| gold | predicted | count |
|---|---|---:|
| a | a | 26 |
| b | b | 4 |
| b | c | 20 |

### running_summary / incremental_overturn_reasoning

| gold | predicted | count |
|---|---|---:|
| c | a | 22 |
| c | b | 5 |
| c | c | 23 |

### source_no_revision / full_info

| gold | predicted | count |
|---|---|---:|
| a | a | 3 |
| b | c | 5 |
| c | a | 19 |
| c | c | 23 |

### source_no_revision / incremental_no_overturn

| gold | predicted | count |
|---|---|---:|
| a | a | 26 |
| b | b | 17 |
| b | c | 7 |

### source_no_revision / incremental_overturn_reasoning

| gold | predicted | count |
|---|---|---:|
| c | a | 22 |
| c | b | 19 |
| c | c | 9 |

### source_revision / full_info

| gold | predicted | count |
|---|---|---:|
| a | a | 3 |
| b | c | 5 |
| c | a | 19 |
| c | b | 1 |
| c | c | 22 |

### source_revision / incremental_no_overturn

| gold | predicted | count |
|---|---|---:|
| a | a | 24 |
| a | c | 2 |
| b | b | 6 |
| b | c | 18 |

### source_revision / incremental_overturn_reasoning

| gold | predicted | count |
|---|---|---:|
| c | a | 19 |
| c | b | 11 |
| c | c | 20 |

### structured_no_source / full_info

| gold | predicted | count |
|---|---|---:|
| a | a | 3 |
| b | c | 5 |
| c | a | 19 |
| c | b | 1 |
| c | c | 22 |

### structured_no_source / incremental_no_overturn

| gold | predicted | count |
|---|---|---:|
| a | a | 26 |
| b | b | 18 |
| b | c | 6 |

### structured_no_source / incremental_overturn_reasoning

| gold | predicted | count |
|---|---|---:|
| c | a | 21 |
| c | b | 23 |
| c | c | 6 |

## Relation-To-Prior Distributions

### source_no_revision

| condition | relation_to_prior | count |
|---|---|---:|
| incremental_no_overturn | confirm | 31 |
| incremental_no_overturn | replace | 15 |
| incremental_no_overturn | unrelated | 4 |
| incremental_overturn_reasoning | confirm | 27 |
| incremental_overturn_reasoning | contradict | 2 |
| incremental_overturn_reasoning | replace | 18 |
| incremental_overturn_reasoning | unrelated | 3 |

### source_revision

| condition | relation_to_prior | count |
|---|---|---:|
| incremental_no_overturn | confirm | 20 |
| incremental_no_overturn | replace | 30 |
| incremental_overturn_reasoning | confirm | 15 |
| incremental_overturn_reasoning | replace | 33 |
| incremental_overturn_reasoning | unrelated | 2 |

## Premise-Role Distributions

### source_no_revision

| condition | premise_role | count |
|---|---|---:|
| incremental_no_overturn | alternative_pathway | 35 |
| incremental_no_overturn | contradiction | 9 |
| incremental_no_overturn | extra_requirement | 5 |
| incremental_no_overturn | unclear | 1 |
| incremental_overturn_reasoning | alternative_pathway | 36 |
| incremental_overturn_reasoning | contradiction | 6 |
| incremental_overturn_reasoning | extra_requirement | 8 |

### source_revision

| condition | premise_role | count |
|---|---|---:|
| incremental_no_overturn | alternative_pathway | 20 |
| incremental_no_overturn | contradiction | 4 |
| incremental_no_overturn | extra_requirement | 26 |
| incremental_overturn_reasoning | alternative_pathway | 14 |
| incremental_overturn_reasoning | contradiction | 2 |
| incremental_overturn_reasoning | extra_requirement | 31 |
| incremental_overturn_reasoning | unclear | 3 |

## Source-Revision No-Overturn Failures

- Total failures: 20
- wrong_relation_to_prior: 20

Representative examples:
- `belief_r::125-strong::tollens::incremental`
  gold `b -> b`, pred `b -> c`
  relation_to_prior `replace`, premise_role `extra_requirement`, model_prediction `c`
  raw `{"label":"c","premise_role":"extra_requirement","relation_to_prior":"replace","rationale":"The new rule gives Jessica allowing use of the item via helping with chores, so her not allowing use once no longer forces the cash condition. The prior conclusion that `
- `belief_r::201-strong::ponens::incremental`
  gold `a -> a`, pred `a -> c`
  relation_to_prior `replace`, premise_role `extra_requirement`, model_prediction `c`
  raw `{"label":"c","premise_role":"extra_requirement","relation_to_prior":"replace","rationale":"The new premise adds an additional condition (ignoring mentorship sessions) for missing out on guidance, so the earlier unconditional conclusion is weakened; without tha`
- `belief_r::654-strong::tollens::incremental`
  gold `b -> b`, pred `b -> c`
  relation_to_prior `replace`, premise_role `extra_requirement`, model_prediction `c`
  raw `{"label":"c","premise_role":"extra_requirement","relation_to_prior":"replace","rationale":"The new evidence adds an alternative way for the leak to be fixed via Jessica, so John’s noticing/reporting is no longer necessary. The prior negative conclusion is weak`
- `belief_r::308-strong::tollens::incremental`
  gold `b -> b`, pred `b -> c`
  relation_to_prior `replace`, premise_role `extra_requirement`, model_prediction `c`
  raw `{"label":"c","premise_role":"extra_requirement","relation_to_prior":"replace","rationale":"The new premise adds a further condition for Jessica having to collect the pen, so the earlier inference from 'Jessica does not have to collect the pen' to John's not pl`
- `belief_r::432-strong::tollens::incremental`
  gold `b -> b`, pred `b -> c`
  relation_to_prior `replace`, premise_role `extra_requirement`, model_prediction `c`
  raw `{"label":"c","premise_role":"extra_requirement","relation_to_prior":"replace","rationale":"The new premise gives an alternative sufficient condition for Jessica feeling comfortable, so the earlier inference that John must not keep the room at 70°F is no longer`
- `belief_r::213-strong::tollens::incremental`
  gold `b -> b`, pred `b -> c`
  relation_to_prior `replace`, premise_role `extra_requirement`, model_prediction `c`
  raw `{"label":"c","premise_role":"extra_requirement","relation_to_prior":"replace","rationale":"The new rule adds an additional condition for John seeking to recover funds, so it does not force the prior negation or its opposite. The earlier commitment is weakened `
- `belief_r::380-strong::tollens::incremental`
  gold `b -> b`, pred `b -> c`
  relation_to_prior `replace`, premise_role `extra_requirement`, model_prediction `c`
  raw `{"label":"c","premise_role":"extra_requirement","relation_to_prior":"replace","rationale":"The new tutoring rule adds an alternative sufficient condition for understanding algebra and returning home, so the earlier inference that John does not practice math is`
- `belief_r::97-strong::ponens::incremental`
  gold `a -> a`, pred `a -> c`
  relation_to_prior `replace`, premise_role `extra_requirement`, model_prediction `c`
  raw `{"label":"c","premise_role":"extra_requirement","relation_to_prior":"replace","rationale":"The new premise gives a different sufficient condition for Jessica acknowledging it, adding extra requirements (water stains and heavy rainfall) rather than directly sup`

## Source-Revision Missed Overturns

- Total missed overturns: 30
- label_semantic_confusion: 13
- wrong_relation_to_prior: 17

Representative examples:
- `belief_r::1-strong::tollens::incremental`
  gold `b -> c`, pred `b -> b`
  relation_to_prior `unrelated`, premise_role `unclear`, model_prediction `c`
  raw `{"label":"c","premise_role":"unclear","relation_to_prior":"unrelated","rationale":"The new premise about consistently outperforming Jessica implies smugness, but it does not bear on whether John buys a better fishing rod. The prior inference that he does not b`
- `belief_r::779-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `a`
  raw `{"label":"a","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new rule gives another sufficient condition for the same conclusion, so it supports the existing commitment rather than changing it."}`
- `belief_r::16-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `a`
  raw `{"label":"a","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new premise gives another sufficient condition for the same conclusion, so it supports rather than weakens the existing commitment."}`
- `belief_r::586-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `confirm`, premise_role `unclear`, model_prediction `a`
  raw `{"label":"a","premise_role":"confirm","relation_to_prior":"confirm","rationale":"The new evidence supports the same conclusion already implied by the accepted rule and John asking about the price, so the commitment remains that Jessica will set a high price an`
- `belief_r::948-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `a`
  raw `{"label":"a","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new rule gives a second sufficient condition for the same conclusion, so it supports the existing commitment that John wears the red scarf to a small party."}`
- `belief_r::768-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `a`
  raw `{"label":"a","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new rule gives another sufficient condition for John winning the trophy, so it supports the existing conclusion without changing it."}`
- `belief_r::170-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `a`
  raw `{"label":"a","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new premise gives another sufficient route to the same conclusion, so it supports rather than weakens the existing commitment that Jessica will mimic John's questi`
- `belief_r::160-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `a`
  raw `{"label":"a","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new premise gives another sufficient condition for the same outcome, so it supports rather than weakens the existing conclusion that he can proceed to examine pati`

## Source-No-Revision Missed Overturns

- Total missed overturns: 41
- label_semantic_confusion: 12
- wrong_premise_role: 1
- wrong_relation_to_prior: 28

Representative examples:
- `belief_r::882-strong::tollens::incremental`
  gold `b -> c`, pred `b -> b`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `b`
  raw `{"label":"b","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new rule gives another sufficient condition for finishing faster, so it is an alternative pathway to the same outcome. Since John does not finish faster, the origi`
- `belief_r::1-strong::tollens::incremental`
  gold `b -> c`, pred `b -> b`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `b`
  raw `{"label":"b","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new premise gives another route to smugness, but it does not support buying the rod. Since John does not feel smug and buying the rod would imply smugness, the rod`
- `belief_r::779-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `a`
  raw `{"label":"a","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new premise gives another sufficient condition for the same conclusion, so it supports rather than weakens the existing commitment that John resolves every challen`
- `belief_r::16-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `a`
  raw `{"label":"a","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new premise gives another sufficient condition for the same conclusion, so it supports rather than weakens the existing commitment."}`
- `belief_r::471-strong::tollens::incremental`
  gold `b -> c`, pred `b -> b`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `b`
  raw `{"label":"b","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new premise gives another way for Jessica to remain with the book club members, but it does not affect the existing inference that John does not attend. That concl`
- `belief_r::586-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `a`
  raw `{"label":"a","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new rule gives another sufficient condition for the same conclusion, so it supports rather than weakens the existing commitment."}`
- `belief_r::637-strong::ponens::incremental`
  gold `a -> c`, pred `a -> a`
  relation_to_prior `confirm`, premise_role `alternative_pathway`, model_prediction `a`
  raw `{"label":"a","premise_role":"alternative_pathway","relation_to_prior":"confirm","rationale":"The new evidence gives another sufficient condition for the same conclusion, so it supports rather than weakens the existing commitment."}`
- `belief_r::879-strong::tollens::incremental`
  gold `b -> c`, pred `b -> b`
  relation_to_prior `replace`, premise_role `contradiction`, model_prediction `b`
  raw `{"label":"b","premise_role":"contradiction","relation_to_prior":"replace","rationale":"The new premise gives a separate way to finish the report, so the earlier inference that John does not turn off the television is no longer forced. Since John may finish the`

