# Retrospective Rules-Only Falsification Protocol

Freeze date: `2026-08-18` UTC

## Classification

- `STUDY = RETROSPECTIVE_DISCOVERY_ONLY`
- `FINAL_VALIDATION = FALSE`
- `RULES_CHANGED_DURING_BACKTEST = FALSE`
- `MODEL_TRAINING_AUTHORIZED = FALSE`
- `READY_FOR_SHADOW = FALSE`
- `READY_FOR_LIVE = FALSE`

This protocol measures the five generators exactly as frozen in
`15_PHASE2_RULE_FREEZE.md`. It cannot promote a strategy. Any hypothesis
created after results are visible requires a new version and may not treat this
same historical window as clean confirmation.

## Historical window and exclusions

The local complete 1m source begins on 2026-05-01, inside sealed periods from
prior W1-W14 studies. It is prohibited for this backtest even though the raw
files are locally readable.

The retrospective source is frozen to public Binance USD-M 1m candles:

- warmup starts: `2023-09-01T00:00:00Z`;
- candidate anchors: `2024-01-01T00:00:00Z` through
  `2024-09-30T23:00:00Z` inclusive;
- source ends: `2024-10-01T00:00:00Z` exclusive so the final allowed horizon
  can close without crossing the exclusion boundary;
- temporal blocks: the nine calendar months January through September 2024.

Explicitly excluded reservations include:

| Prior study | Sealed period relevant to audit | Treatment |
|---|---|---|
| W3 | 2024-10-01 through 2026-07-31 | Entirely excluded |
| W2 | 2025-04-01 through 2026-07-31 | Entirely excluded |
| W1 | 2026-05-01 through 2026-07-31 | Entirely excluded |
| W7 | 2026-05-01 through 2026-07-17 | Entirely excluded |
| W9.1 | June 2026, not downloaded for its holdout | Not accessed |
| W12 | 2026-07-27 through 2026-07-31 | Entirely excluded |
| W11/W14 | 2026-08-01 onward | Entirely excluded |
| W4/W8/W9/W10/W13 | future/unpopulated sealed holdouts | Not accessed |

The selected candidate window ends before the earliest dated exclusion above.
W1-W14 artifacts are read only to identify exclusions; no holdout market rows,
episode IDs, labels, or outcomes are loaded.

## Frozen population

The unchanged general-market pipeline runs every 15 minutes for all 11 symbols
and evaluates LONG and SHORT independently. Candidate population events retain
the existing definition: `ELIGIBLE` with disposition `CANDIDATE` or
`ENTERABLE`. Terminal waits, invalidations, `UNKNOWN`, and `INELIGIBLE` remain
diagnostic populations and are not silently converted into entries.

Independent episodes use the existing setup identity and 60-minute overlap
suppression per strategy, symbol, and side. The report also groups same-time
cross-symbol episodes into hourly temporal clusters as a conservative effective
sample-size diagnostic.

## Frozen outcomes

Reference volatility is ATR14 from the last fully closed 15m bar at the actual
candidate/confirmation timestamp. It remains fixed for the episode.

- favorable barrier: `+0.50 ATR` in proposed direction;
- adverse barrier: `-0.50 ATR`;
- horizon: 60 minutes;
- same 1m bar touching both barriers: `ADVERSE_FIRST`;
- realized common payoff: favorable barrier bps, negative adverse barrier bps,
  or terminal directional return for `NEITHER`.

Secondary diagnostics are fixed-horizon directional return, MFE, MAE,
MFE/MAE, event ordering/timing, path efficiency, structural invalidation and,
where setup timestamps differ from candidate timestamps, consumed versus
remaining movement.

## Frozen economics

- gross: realized common payoff without cost;
- conservative net: gross minus 20 bps round trip;
- latency diagnostic: directional implementation shortfall from candidate
  reference price to the next 1m open, reported separately and also deducted
  from latency-stressed net;
- leverage: 1x only;
- economic hurdle: 20 bps.

Two distinct statements are reported: whether gross plausibly pays the frozen
20 bps cost, and whether net expectancy itself exceeds 20 bps. The latter
requires gross expectancy above 40 bps and is intentionally stricter.

## Baselines

Every candidate is compared without fitting against:

1. the empirical unconditional LONG/SHORT population at all valid anchors;
2. the empirical same-side population at all valid anchors;
3. random eligible selection from the comparable symbol/side/month population;
4. simple 15m directional persistence, defined before outcomes as proposed
   side matching the sign of the frozen 15m `return_3_bps` feature.

Real empirical prevalences replace any theoretical 33.3% assumption.

## Statistics and support

- 10,000 episode bootstrap samples;
- 10,000 calendar-month block bootstrap samples;
- deterministic seed `20260818`;
- Benjamini-Hochberg FDR across five strategy-family payoff comparisons;
- minimum support: 500 independent episodes, six symbols, four monthly blocks;
- side claims require 150 independent episodes for that side.

No strategy is called promising solely because its mean is positive. Temporal,
symbol, side, tail-risk and concentration diagnostics are mandatory.

`PROMISING_RETROSPECTIVELY` requires the support gate, lower 95% episode-CI
above zero for both matched-baseline payoff improvement and net expectancy,
positive net expectancy in at least two thirds of represented monthly blocks,
at least six positive symbols, no best symbol above 35% of positive payoff,
and FDR survival. These criteria classify discovery only.

Catalog diversity is sufficient only if at least two strategies meet support,
the largest strategy contributes no more than 80% of independent episodes, and
at least 1% of candidate-bearing snapshots contain multiple strategies. This
criterion does not build or optimize a router.
