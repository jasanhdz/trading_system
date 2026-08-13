# Aegis Opportunity Atlas B1 - Result

## Verdict

`B1_PATH_RISK_LEARNABLE_DIRECTION_AND_RANKING_NOT_DEMONSTRATED`

B1 separated the trading problem into opportunity, direction, symbol ranking
and path risk. It evaluated 5,607 independent four-hour market events at 60
minutes and 5,606 at 240 minutes, with 123,354 and 123,332 symbol-side rows.
Neither combined policy passed.

## Component Results

### Opportunity

At 60 minutes, ROC AUC was 0.683 in validation and 0.628 in pseudo-forward.
This initially looks useful, but the target base rate was 92.0% and 81.0%.
At 240 minutes it was 99.4% and 98.3%, and pseudo-forward AUC fell to 0.478.

The label asked whether the best ex-post result among 22 symbol-side choices
exceeded 42 bps. With that many choices, almost every timestamp has a winner.
This is a multiple-comparison target, not a useful ex-ante opportunity label.
Calibration probabilities consequently saturated near one and the frozen q90
threshold approached 1.0.

### Direction

Balanced accuracy ranged from 0.488 to 0.527. This is chance-level behavior.
The current causal event features did not reliably predict whether the best
future side would be LONG or SHORT.

### Symbol Ranking

Rank Spearman ranged from -0.026 to +0.032. LONG top-ranked selections did not
beat deterministic random selection. SHORT sometimes beat random pointwise,
but rank correlation remained near zero and the result did not transfer as a
stable component.

The system therefore still cannot answer which symbol offers the best future
return, even after identifying an active market timestamp.

### Path Risk

This was the useful finding. MAE rank correlations ranged from 0.306 to 0.389
and MFE rank correlations from 0.307 to 0.390 across validation,
pseudo-forward, LONG, SHORT and both horizons.

The available features can estimate relative path adversity and favorable
excursion materially better than chance. This supports a future risk-quality
filter, but not an autonomous directional strategy.

## Combined Policy

The frozen combination selected zero events in most partitions and one losing
event in one partition. This was correct fail-closed behavior: direction and
ranking failed, predicted gross return rarely cleared 42 bps, and the
opportunity probability threshold was saturated.

B1 does not export a model and does not authorize forward collection, Shadow
or Live.

## What Should Change Next

Opportunity Atlas B2 should preserve the successful path-risk decomposition
and replace the failed targets:

1. Define market opportunity using a preregistered basket or cross-sectional
   quantile, not the ex-post maximum of 22 choices.
2. Define symbol alpha as beta-neutral residual return against BTC and the
   market basket, not raw return.
3. Train timestamp-grouped pairwise or listwise ranking on residual utility.
4. Estimate direction only inside separately demonstrated economic regimes;
   do not force a universal LONG/SHORT classifier.
5. Use the MAE/MFE model as a quality and abstention component, never as proof
   of direction.
6. Require a transparent directional baseline to show positive gross edge
   before adding ML.
7. Reserve genuinely new forward evidence for any promotion claim.

This is a narrower and more defensible successor than adding a larger
committee to the same targets.

## Reproducibility And Safety

- Result SHA-256: `b76d4c27f27ee18d625e1577b27979ff513a332727b654303143d881c6f11631`
- 60m event dataset: `8ba9d92c3bf6b5f5bb2497fe123a23c757d86da96e4277a6e3112664774a63ef`
- 240m event dataset: `961f0217398aee865935099b2e447ed1a97c52b024176ce45ae92abd5ad55608`
- 60m symbol-side dataset: `60c8341cac908ffc8882c150617bbfe91fe936c6a9352325c3068893b65508f5`
- 240m symbol-side dataset: `3e5b7fcfa4847c2933f101010f51759a9d66e4b9c93e9ebce8fa4574552f6a70`
- Repeated evaluation reproduced hashes and metrics.
- Focused A1/A2/B1 tests: `14 passed`.
- Full unit suite: `778 passed, 5 failed`. The five failures are unchanged
  historical branch-authority checks requiring
  `feature/aegis-ts-clean-rebuild`; this work remains isolated on
  `work/entry-quality-evidence-20260726`.
- Exchange calls and mutations: `0`.
- Runtime, PM2, Live, Shadow and TypeScript changes: `NONE`.
