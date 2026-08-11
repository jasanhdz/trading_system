# Aegis Calibrated Horizon V11 Research Plan

## Question

V10 proved that direction and barrier outcomes contain out-of-sample signal,
but that signal did not produce positive conservative entry utility. V11 asks
whether stricter calibration, horizon sharing, causal regime context, and an
explicit clean-entry target can convert that component skill into economically
useful selection without relaxing V10's costs or promotion gates.

## Unchanged authority

V11 reuses V10's nine future-OHLC barrier outcomes byte-for-byte. It does not
relabel V10 results, reduce costs, shorten the episode spacing, or tune a
threshold using test outcomes. The 48,191 side-neutral episodes remain the
independent evidence units.

## Clean-entry target

The primary 10%-ROE/60-minute contract is clean only when the favorable barrier
is reached first, before bar seven, and pre-event MAE does not exceed half of
the adverse barrier. This target uses future OHLC only. Regime, indicators,
volume, and model decisions cannot define success.

## Horizon specialists

Three specialists represent 30, 60, and 120 minutes. Within one horizon the
5%, 10%, and 20% barriers are stacked and barrier magnitude is an input. This
shares evidence across related targets while allowing different temporal
behavior. The original per-contract V10 structure remains a test-only
component control and has no selection authority.

## Calibration

Each chronological fold is divided into training, probability calibration,
policy selection, and untouched test periods with embargoes. Global Platt
calibration is mandatory. Symbol and causal-regime calibration is used only
when a group has at least 200 rows and 20 examples of the class; group estimates
are shrunk toward the global estimate. Test data never calibrates probabilities
or thresholds.

## Causal regime

Regime is computed from closed historical 4h/12h return, volatility, volume,
and range-location context already present in V10. It is an explanatory and
calibration grouping variable, never an outcome label or an unconditional
block.

## Utility and attribution

Predicted utility separately records favorable value, adverse value, severe
cost, ambiguous penalty, unresolved penalty, and a small clean-entry bonus.
The bonus cannot make a negative base utility positive by itself: final utility
must still pass the frozen positive threshold and all economic test gates.

## Validation

Promotion requires independently skilled direction, at least two of three
horizon specialists, a skilled clean-entry model, positive incremental utility
in three of four folds, non-negative worst-fold utility, improved CVaR, payoff
ratio at least one, acceptable opportunity frequency, and eight passing
leave-one-symbol-out checks. Horizon specialists must not regress against the
V10 per-contract component control.

Current TypeScript protection, MAE, and time-to-positive outcomes are reported
after entry selection only. They cannot tune the selector or substitute for a
failed entry gate.

## Safety

- Runtime effect: `NONE`.
- Shadow activation: prohibited without separate authorization.
- Live activation: prohibited.
- Model export: prohibited until every gate passes.
- Exchange calls: zero.
- Exchange mutations: zero.
