# Aegis Regime-Aware Directional v6 Research Plan

## Status

`PREREGISTERED_RESEARCH_ONLY`

This protocol studies an asymmetric directional system. It does not authorize
Live selection, exchange mutations, automatic promotion, or changes to the
currently running TypeScript execution and position-management semantics.

## Economic Objective

A successful entry is not required to reach the full take profit. It must have
a statistically supported probability of reaching a protectable positive
advantage before material adverse excursion, then preserve positive net value
through the existing price protections or a causal deterioration exit.

Primary evidence is protected net return after costs. Supporting evidence is:

- MAE and MFE;
- time to 5%, 10%, 20%, and 50% ROE at the frozen research leverage;
- time underwater;
- early reversal rate;
- break-even and trailing activation;
- ExitEye close behavior using subsequent real committee observations;
- trade frequency and no-trade gap distribution.

Low MAE alone is not a success label. Full take-profit attainment alone is not
a success requirement.

## Regime Contract

Regime is factorized into direction, volatility, structure, and phase. Global
context is derived from BTC plus the eleven-symbol cross-section. Local context
is derived from each symbol's causal 5m, 15m, and 1h history.

The deterministic axes are descriptive inputs only. A separate probabilistic
router receives 16 causal BTC and cross-sectional features and predicts the
realized global direction over the frozen 24-bar horizon. Its future return
and breadth label is never an input. Inside every walk-forward fold the router
is fitted on training data, calibrated on calibration data, and compared with
the training-prior and majority baselines on independent test timestamps.
Promotion requires skill in at least three of four folds.

In a bearish regime, SHORT is the primary trend role and LONG is a tactical
countertrend role. In a bullish regime the roles reverse. Neutral regimes are
selective for both sides. These roles are model inputs and audit groups, not
permission fabricated from test-period outcomes.

## Dataset Contract

The canonical dataset contains both LONG and SHORT paths from the same market
snapshots. Features end at the signal close. Entry is the next bar open. Future
path data is used only for labels. Training includes winners, losers, ambiguous
paths, slow recoveries, and censored horizons.

Overlapping rows may be used for training. Scoring rows must be non-overlapping.
Temporal folds use purge and embargo. Test folds cannot determine thresholds,
feature definitions, labels, model families, or acceptance gates.

## Model Contract

LONG and SHORT use separate specialist fits under one shared schema. Each side
estimates protectable advantage, directional success, early reversal, protected
net return, q90 MAE, and time to advantage. Probabilities are calibrated only
on calibration data and audited by side, regime direction, and directional
role.

Unknown, sparse, non-finite, or unsupported states abstain. No trade quota may
force selection.

## Lifecycle Replay

Price protection replays both admissible OHLC paths and takes the pessimistic
result. The current Python brain is replayed causally on subsequent candles to
provide action and vote observations for ExitEye. Missing committee evidence
does not become a neutral vote; the affected replay is marked incomplete and
falls back only to the separately reported price-protection result.

The replay does not alter take profit, stop loss, break-even, trailing, callback,
fees, leverage, or current committee outputs.

## Validation And Promotion

Historical evaluation uses four purged expanding walk-forward folds, bootstrap
uncertainty, leave-one-symbol-out checks, regime attribution, and comparison
with the unfiltered and existing v4/v5 controls. Promotion requires the frozen
gates in the YAML contract. Any failure leaves v6 in research only.

Historical success permits only a separate prospective Shadow deployment.
Shadow must produce matured, independent outcomes before an owner review. No
historical or Shadow result automatically authorizes Live.

## Planned Tools

1. Canonical directional dataset builder.
2. Path-dependent MAE/MFE/time labeler.
3. Global/local regime auditor.
4. LONG and SHORT specialist trainer.
5. Full price-protection and ExitEye replay.
6. Purged walk-forward and calibration evaluator.
7. Specialist and combination ablation evaluator.
8. Current-versus-v6 Shadow eligibility gate.
9. Promotion, rollback, and evidence report generator.

## Prohibitions

- No training only on winners.
- No future data in features.
- No threshold tuning on test folds.
- No fabricated committee votes.
- No automatic Live promotion.
- No exchange mutation authority.
- No USD 100 activation.
