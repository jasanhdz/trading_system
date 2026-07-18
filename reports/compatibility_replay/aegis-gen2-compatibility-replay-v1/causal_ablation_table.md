# Gen2 causal ablations

| Stage | Changed axis | Trades | PF | Net expectancy | Win rate | Max DD | Overlap Stage 0 | Overlap E2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| STAGE_0 | NONE | 688 | 1.74082396916 | 0.14012459096 | 0.579941860465 | 10.6985161553 | 1.0 | None |
| STAGE_1 | sampling | 2404 | 0.646052003292 | -0.122124842952 | 0.405574043261 | 297.302595665 | 0.08575581395348837 | 0.02192638997650744 |
| STAGE_2 | features | 688 | 1.84574384501 | 0.150601044368 | 0.585755813953 | 8.34446981696 | 0.47238372093023256 | 0.0 |
| STAGE_3 | model_capacity | 688 | 1.04803156062 | 0.0119658712406 | 0.469476744186 | 35.1969394082 | 0.313953488372093 | 0.0 |
| STAGE_4 | runtime_selection | 715 | 0.949592278003 | -0.0210625043233 | 0.524475524476 | 55.4626423578 | 0.3168604651162791 | 0.0 |
| STAGE_5 | eqm_population | 688 | 1.74082396916 | 0.14012459096 | 0.579941860465 | 10.6985161553 | 1.0 | 0.0 |

Deltas are isolated diagnostics and are not additive; interactions between axes remain unidentified.
