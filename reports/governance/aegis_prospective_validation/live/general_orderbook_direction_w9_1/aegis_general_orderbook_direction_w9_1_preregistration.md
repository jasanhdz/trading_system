# Aegis W9.1 General Order-Book Direction - Preregistration

## Question

Does causal L2 evolution and aggressive trade flow predict which economically
meaningful short-horizon barrier is reached first, without conditioning
training on W7 Opportunity?

## Authority

- Mode: `OFFLINE_HISTORICAL_RESEARCH_ONLY`.
- W1-W9 holdouts remain sealed.
- W7 Opportunity is not loaded, fitted or used for W9.1 training.
- W9.2 is prohibited unless every W9.1 gate passes.

## Data

- Provider: free first-day-of-month Tardis samples for Binance USD-M Futures.
- Types: sequential `incremental_book_L2`, `quotes`, and `trades`.
- Symbols fixed before outcomes: `ADAUSDT`, `BNBUSDT`, `BTCUSDT`, `ETHUSDT`,
  `SOLUSDT`, `XRPUSDT`.
- TRAIN months: September 2024, March 2025, September 2025.
- VALIDATION months: December 2025, March 2026.
- FINAL_HOLDOUT: June 2026, `SEALED_NOT_OPENED` and not downloaded.

## Episodes And Targets

- Anchors: fixed UTC grid every 120 seconds.
- Maximum label horizon: 60 seconds, so target windows do not overlap.
- Feature lookbacks: 100/250/500 ms and 1/2/5 seconds.
- Primary target: first `+25 bps` or `-25 bps` barrier within 60 seconds.
- Diagnostic targets: 10 bps/30 seconds and 15 bps/60 seconds.
- Classes: `UP_FIRST`, `DOWN_FIRST`, `NEITHER`.
- `SKIP` is mandatory when directional probability is ambiguous.

The target family was frozen after measuring only spread and unconditional
barrier prevalence in the existing ADA September 2025 pilot. No feature/outcome
relationship or VALIDATION result was inspected.

## Models And Gates

- Models: multinomial logistic, dual binary logistic, directional ridge,
  depth-3 tree and constrained histogram gradient boosting.
- Ablations: static book, dynamics, trade flow, pressure/response, absorption,
  and full.
- Selection uses TRAIN only; confidence thresholds are limited to
  0.55/0.60/0.65/0.70.
- Primary economics: 14 bps round trip; stress: 20 bps.
- Required validation edge: at least +3 bps/taken episode, positive bootstrap
  lower bound, profit factor above one, four positive symbols, both validation
  months positive, positive at 20 bps cost and at 250 ms latency.
- Bootstrap: 10,000 at symbol-day block level. FDR: Benjamini-Hochberg.
- Leave-one-symbol-out transfer is mandatory.

The machine-readable authority is
`config/experiments/aegis_general_orderbook_direction_w9_1.yaml`.

Production, TypeScript, Aegis Brain, guards, leverage, PM2, Shadow, Live,
authenticated requests and exchange mutations remain prohibited.
