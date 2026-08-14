# Aegis Volume Wave W1 - Preregistration

## Purpose

W1 tests, and attempts to refute, the hypothesis that some closed 5-minute
volume impulses begin a short-lived directional wave. It does not assume that
the next candle has the same color and it does not treat high volume as
monotonically favorable. Continuation is defined by favorable-before-adverse
path, MFE/MAE, directional persistence, path efficiency and net utility.

The three owner-supplied XRP screenshots are intuition examples only. One shows
clean bearish follow-through; another shows high-volume selling near a local
low followed by reversal. Their coexistence motivates explicit exhaustion and
remaining-space variables, but the images are not statistical evidence.

## Existing Components To Reuse

### Research

- `binance_public_archive.py`: exact Binance public archive identities,
  checksums and atomic downloads.
- `market_event_fast_track_m1a.py`: strict 1-minute kline parsing, taker-buy
  flow, complete closed-bar resampling and causal regime primitives.
- `event_path_quality_c2a.py`: chunked aggregate-trade import, next-open path
  labels, adverse-first triple barriers, MFE/MAE and clustered bootstrap.
- `market_event_fast_track_evaluation.py`: economic metrics, drawdown, CVaR,
  symbol concentration and fail-closed evidence gates.
- `hybrid_ts_protection_replay.py`: offline replay of existing break-even and
  trailing behavior if a W1 entry family first demonstrates standalone edge.

### Future Runtime, Not Used By W1

- TypeScript `WebSocketManager`: closed candle, aggregate-trade and depth
  subscriptions with resubscription.
- `TradingService`: signal consumption, sizing, bracket validation,
  reconciliation, break-even, trailing and recovery.
- `BinanceAdapter`: exchange filters, leverage/margin operations, entries and
  stop/take-profit placement.
- `FsStateStore`: durable local operational state.

W1 will not import or modify these operational components. If evidence later
supports a real-time detector, it should emit a decision contract into the
existing pipeline rather than create a second execution engine.

## Frozen Scientific Boundary

- Eleven Binance USD-M symbols, LONG and SHORT separately.
- Closed 5-minute decision candles and completed 15-minute context only.
- BTC context aligned causally to each decision timestamp.
- Twelve checksum-verified months, August 2025 through July 2026.
- TRAIN ends February 2026; VALIDATION ends May 2026; FINAL HOLDOUT remains
  sealed from May through July 2026.
- Four causal entry variants: immediate, one-bar confirmation, bounded
  pullback and extreme-break confirmation.
- Five horizons from one to six 5-minute bars and six ATR barrier pairs.
- Base round-trip cost 14 bps; stress costs 20 and 30 bps.
- Day-cluster bootstrap, Beta(1,1) posterior and FDR correction.

The registered additive ladder tests where information is added: volume,
candle quality, taker alignment, 5-minute trend, completed 15-minute trend, BTC
non-opposition and remaining space. No validation threshold search is allowed.
Models are prohibited unless an interpretable registered family first shows
standalone net edge.

## Gates

W1 cannot proceed to modeling, Shadow or Live merely because a bin or one
symbol looks favorable. Validation requires positive net expectancy with a
positive lower confidence bound, profit-factor confidence above one, survival
at 20 bps, at least seven positive symbols, temporal transfer and superiority
to all registered controls. The holdout may be opened once only after the
validation gate passes.

Current states:

- `W1_DATASET_READY=false`
- `W1_RULE_EDGE_FOUND=false`
- `W1_MODELING_JUSTIFIED=false`
- `W1_READY_FOR_SHADOW=false`
- `W1_READY_FOR_LIVE=false`

No credentials, authenticated exchange requests, exchange mutations, PM2
changes or runtime changes are authorized by this experiment.
