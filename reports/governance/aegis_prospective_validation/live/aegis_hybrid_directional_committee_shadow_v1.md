# Aegis Hybrid Directional Committee V1

## Purpose

This preregistered research candidate tests whether specialized LONG and SHORT
opportunity estimators can be combined with shared path-risk estimators without
changing the current trading system. It is an offline and Shadow experiment.
It has no exchange authority and cannot create, cancel, modify, or close an
order.

## Architecture

The committee contains independent LONG and SHORT opportunity specialists.
They are independent because bullish and bearish opportunities are not assumed
to be mirror images. A shared path model receives the proposed direction as an
explicit input and estimates directional MAE, MFE, terminal net return, and
adverse-path danger. Shared heads may learn common volatility and market
structure while preserving directional interactions.

The shared representation contains the original 83 normalized features, a
directionalized copy (`feature * LONG_OR_SHORT_SIGN`), and the direction sign.
The explicit interactions prevent tree learners from treating a bullish move
as equivalent evidence for LONG and SHORT merely because a single side flag has
no marginal split value.

The committee does not use majority voting. Every output retains a measurable
meaning:

- probability of a positive directional terminal return after frozen costs;
- probability of a direction-specific bad path;
- median and upper-quantile directional MAE;
- median directional MFE;
- expected terminal net return.

The combined score is an observational ranking only. It ranks every hypothesis
from opportunity, inverse danger, predicted path efficiency, and a bounded net
return factor. It deliberately does not suppress observations when a regressor
is conservative. It cannot authorize an entry and is not consumed by
TypeScript execution.

## Causal Labels

Features are computed at the finalized signal close. Labels use only the next
12 finalized five-minute candles and assume entry at the next candle open.
Manual selection of visually clean historical rises is prohibited. Positive,
negative, failed-breakout, false-reversal, and high-MAE paths remain in the
population.

ADA is a mandatory diagnostic slice, not the sole training source. The primary
population contains all 11 symbols. A leave-ADA-out test measures whether the
candidate generalizes to ADA instead of memorizing it.

## Evidence Standard

Training, calibration, and scoring blocks are chronologically disjoint with a
120-minute embargo. Model selection or threshold adjustment from a scoring
block is prohibited. The final lockbox beginning on 2026-04-27 remains
unavailable to this development experiment.

No candidate is promoted merely because it fits historical clean rises. It
must improve average precision over prevalence, preserve calibration, estimate
MAE quantiles with valid coverage, and show positive net expectancy across
every scoring fold without excessive symbol concentration. A passing offline
result still requires prospective Shadow evidence and separate owner
authorization.

## Runtime Boundary

- mode: `SHADOW_ONLY`
- automatic training: disabled
- automatic promotion: disabled
- TypeScript selection effect: none
- exchange authority: false
- exchange mutations: zero
- Live or PM2 restart required by this change: no

## Initial Offline Result

The initial run evaluated 172,480 causal rows per direction, or 344,960
directional hypotheses. Neither direction passed the complete promotion gate in
any of four scoring folds.

The opportunity specialists showed modest discrimination above prevalence and
kept ECE below 0.08. The shared MAE q90 head achieved valid coverage and beat
the constant quantile baseline in every fold. These component results justify
continued observation, but they do not establish an economically useful entry
policy.

Selecting the highest observational score per direction and hour remained
negative after the frozen 0.10% round-trip cost in every fold. LONG mean
expectancy ranged from approximately -0.077% to -0.152%; SHORT ranged from
approximately -0.076% to -0.120%. Symbol concentration stayed below 15%.

The leave-ADA-out diagnostic also remained negative on ADA: approximately
-0.133% for LONG and -0.067% for SHORT. This indicates limited cross-symbol
signal but no demonstrated tradable edge. The artifact is therefore marked
`OFFLINE_VALIDATION_FAILED_OBSERVATION_ONLY` and must remain Shadow-only.
