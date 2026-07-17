# Aegis Clean Rebuild Phase 2 Final Report

## Result

**PIPELINE CORRECT, BUNDLE REJECTED.**

The scientific pipeline and Python/TypeScript contracts are materially better validated offline. The real local-data experiment did not meet pre-registered breadth and fold-stability criteria, so no artifact was approved for shadow and no runtime shadow collection was activated.

## Validation delivered

- Python suite expanded from 14 to 43 behavioral tests.
- Deterministic LONG, SHORT and NO_TRADE paths.
- Ranking, tie breaking, blocked candidates, cooldowns and no-slot behavior.
- Feature parity, canonical order, causality and label-isolation tests.
- Temporal split/embargo/train-only normalization invariants.
- Direct numerical and monotonic layer tests.
- Corrupt bundle/schema/universe/timeframe/normalizer fail-closed tests.
- Deterministic H12 shadow outcome and replay evidence.
- TypeScript manifest/contract/gate/shadow suite expanded to 28 focused tests.
- Explicit non-executing shadow mode with coordinated-cycle validation.

No coverage package was available and none was installed. Direct behavior coverage is mapped in `AEGIS_SCIENTIFIC_VALIDATION.md`.

## Candidate experiment

The read-only local experiment used 574,200 source rows and 4,231 coordinated cycles over January-June 2026. The full-layer final test had positive expectancy (0.003421) and profit factor (2.043) on 34 signals, while simple directional baselines were negative. This was not sufficient: only one of four folds was positive, two had zero signals, and SUIUSDT represented 61.8% of signals.

The experimental artifact remains `REJECTED_EXPERIMENT`, is not in the approved registry, and did not replace `aegis-offline-reference-v1`.

## Shadow integration

Shadow contracts, replay and evidence are implemented and tested but not attached to a running candle service. `brain.mode` is explicit, while `execution.enabledByConfig` remains false. The strict gate denies every shadow response with `SHADOW_MODE_NON_EXECUTING`. There is no `createOrder`, close, sizing or Binance dependency in the new brain integration.

The absence of a single existing coordinated eleven-symbol close event was handled conservatively: no changes were made to `TradingService`. An approved bundle and a later operator-controlled integration phase are required before a read-only process is started.

## Performance

The eleven-symbol fixture completed at p50 119.77 ms, p95 129.97 ms and p99 139.93 ms across 200 unique evaluations. This fits comfortably inside a 5m research cycle in the measured environment, but no production SLA is asserted.

## Operational safety

No Binance endpoint or credential was used. No order/position/kill-switch/authorization/PM2 path was touched. No deployment or process restart occurred. Execution remains disabled and no push was made.
