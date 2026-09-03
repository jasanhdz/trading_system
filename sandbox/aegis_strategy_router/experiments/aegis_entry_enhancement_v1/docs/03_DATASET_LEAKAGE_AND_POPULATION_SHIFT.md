# Dataset, Leakage And Population Shift

## Causal reconstruction

Checksum-verified Binance public monthly archives from January-April 2026 were
used only to supply Phase 1 warmup. They were joined to the existing neutral
May-August candle dataset. All eleven symbol sources are continuous at one
minute with no conflicting duplicates or filled gaps.

For every eligible Aegis timestamp the Phase 1 builder reconstructed all eleven
symbols, both sides, 1m/5m/15m/1h/4h/1d context, structural levels, BTC/ETH and
breadth. The Aegis side was joined only after neutral market features and both
counterfactual directional scores existed.

- `max_feature_available_at > signal_timestamp`: 0 rows.
- Opportunity retrained: no.
- Directional Alpha retrained: no.
- Aegis side changed: no.
- FINAL_HOLDOUT target columns: none.

`LEAKAGE_CHECK_PASSED = TRUE`

## Population shift

July Aegis signals had median Opportunity `0.9691`, compared with `0.9797` in
the Entry Quality V1 reference population. The frozen Opportunity threshold is
`0.9999067`, while the maximum July Aegis score was `0.9995714`. Consequently
the Opportunity and combined gates accepted zero signals.

Only 0.93% of July rows exceeded the frozen p01/p99 feature-support fraction
gate, so broad feature OOD was not the principal failure. The practical failure
was score/threshold transfer. Recalibrating the threshold on these outcomes is
prohibited and would constitute a separate V2 hypothesis.

July VALIDATION contains 108 SHORT and zero LONG signals. It has adequate raw
count for diagnostics but no independent evidentiary status and no LONG
transfer evidence.

`VALIDATION_SUPPORT_SUFFICIENT = FALSE`
