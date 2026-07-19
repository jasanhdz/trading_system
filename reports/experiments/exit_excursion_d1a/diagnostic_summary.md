# EXIT_EXCURSION_D1A

Exploratory descriptive analysis of the 1,292 frozen E3 SHORT entries. No stop, target, trailing, callback or risk unit is evaluated.

## Data quality

- P0 maximum absolute reproduction error: `0.0` (tolerance `1e-12`).
- Complete 5m trajectories: `1292`; gaps: `0`; same-bar extrema order ambiguities: `4`.

## Required questions

1. Typical MFE: median 42.864505 bps; mean 52.828393 bps.
2. Typical MAE: median 38.036384 bps; mean 52.413548 bps.
3. Ever favorable: 97.910217%. Reached 10 bps: 85.913313%; 25 bps: 68.962848%; 50 bps: 42.337461%.
4. Finished negative after being favorable: 45.743034%.
5. Median H12 capture ratio among nonzero-MFE trades: 0.102499.
6. Giveback of at least 50%: 66.795666%; at least 100%: 46.052632%.
7. Median time to MFE: 25.0 minutes within H12.
8. MFE preceded MAE in 52.554180%; MAE preceded MFE in 47.136223%; ties are reported separately.
9. Favorable movement before adverse movement is reported for every frozen bps pair in `excursion_summary.json`; same-bar events are not ordered.
10. Shorter deterministic exits are listed without selection in `temporal_exit_results.json`; deltas versus T60 are {"T10":{"folds_improved":3,"pooled_delta_vs_t60":0.0002786880652563247},"T15":{"folds_improved":3,"pooled_delta_vs_t60":0.0003148293454632101},"T30":{"folds_improved":1,"pooled_delta_vs_t60":0.00010635620570056351},"T45":{"folds_improved":2,"pooled_delta_vs_t60":1.3166472482480226e-05},"T5":{"folds_improved":3,"pooled_delta_vs_t60":0.0001842484442520437}}.
11. Horizons improving versus T60 in at least 3/4 folds: ['T5', 'T10', 'T15'].
12. T60 B_BASE net expectancy is -0.001534179038; all horizon/scenario net results are {"T10":{"A_OPTIMISTIC":-0.0008471576390931475,"B_BASE":-0.0012554909724264809,"C_PESSIMISTIC":-0.001872157639093147},"T15":{"A_OPTIMISTIC":-0.0008068496922195953,"B_BASE":-0.0012193496922195954,"C_PESSIMISTIC":-0.0018443496922195953},"T30":{"A_OPTIMISTIC":-0.0010028228319822422,"B_BASE":-0.001427822831982242,"C_PESSIMISTIC":-0.0020778228319822418},"T45":{"A_OPTIMISTIC":-0.0010835125652003256,"B_BASE":-0.0015210125652003253,"C_PESSIMISTIC":-0.0021960125652003256},"T5":{"A_OPTIMISTIC":-0.0009457639267640951,"B_BASE":-0.0013499305934307618,"C_PESSIMISTIC":-0.001958263926764095},"T60":{"A_OPTIMISTIC":-0.0010841790376828054,"B_BASE":-0.0015341790376828055,"C_PESSIMISTIC":-0.002234179037682806}}.
13. Mean MFE all trades 52.828393 bps; without best 1% 50.814448 bps; winsorized 52.517947 bps.
14. Descriptive decision: `ADAPTIVE_EXIT_RESEARCH_JUSTIFIED`. This does not approve an exit policy.
15. Yes. A new unambiguous risk unit would be required before any R-based trailing or callback experiment.

## Classification

`MATERIAL_EXCURSION_GIVEN_BACK`

## Next decision

`ADAPTIVE_EXIT_RESEARCH_JUSTIFIED`
