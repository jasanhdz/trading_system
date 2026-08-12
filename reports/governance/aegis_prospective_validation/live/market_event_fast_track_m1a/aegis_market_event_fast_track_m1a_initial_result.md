# Aegis Market Event Fast Track M1A Initial Result

## Status

`M1A_HISTORICAL_DATA_READY_PILOT_INTEGRATION_PASS`

`M1A_READY_FOR_FORWARD_SHADOW = FALSE`

`M1A_READY_FOR_LIVE = FALSE`

## Historical data

The official Binance archive fast track is operational. It downloaded and
verified 682 of 682 expected monthly one-minute kline archives for Spot and
USD-M Futures across all 11 symbols from January 2024 through July 2026.

The archives total approximately 1.2 GB. Every accepted file is bound to the
official `.CHECKSUM` value and the local SHA-256 in the private manifest.
Coverage audit result: `PASS`, fraction `1.0`.

For the one-minute first pass, aggressive buy/sell flow is reconstructed from
the physical Binance kline fields `taker buy quote volume` and total quote
volume. This avoids downloading hundreds of GB of redundant tick-level
aggTrades. Tick archives remain optional for a separately justified subminute
experiment.

## ADA July integration pilot

The pilot used one symbol and one month only. It is an integration diagnostic,
not promotion evidence.

- common complete minutes: 44,640;
- TRAIN minutes: 26,784;
- validation minutes: 17,856;
- raw candidates: 1,031;
- independent candidates after one-hour spacing: 380;
- pattern/side groups: 13;
- horizon: 60 minutes;
- entry: next one-minute open;
- base fees, slippage and funding: included.

Twelve of thirteen pattern/side groups had negative mean net expectancy. The
only favorable group was `TREND_PULLBACK_CONTINUATION:LONG`:

- events: 26;
- mean net expectancy: `+0.2415%` per event;
- profit factor: `2.856`;
- mean MAE: `0.387%`;
- bootstrap 95% expectancy lower bound: `+0.0557%`.

This result is insufficient because it contains only 26 events, one symbol and
one month. The frozen gate requires at least 100 independent validation events
per pattern/direction, controlled symbol concentration, multiple temporal
thirds and matched-random outperformance. No threshold may be changed to
amplify this pilot result.

## What is now possible

- causal 1m, 5m, 15m, 1h, 4h and 1d reconstruction;
- factorized direction, volatility and liquidity regime;
- eight preregistered micro-pattern families;
- thresholds fitted only on TRAIN;
- next-minute economic replay with costs, MAE and MFE;
- daily-block bootstrap and fail-closed gates;
- offline checksum and coverage audit.

News and whale attribution remain outside M1A because their historical event
timestamps and revisions are not yet governed. They are candidates for M1B,
not inputs to retrofit into this result.

## Integrity

- implementation commit: `08c1e39c85d1cec4508b0305e9e357e5d4c4640e`;
- M1A configuration SHA-256:
  `d948dffe06bf4af7abc2a99855810b7e21728b2c00814ba38dbaaec00ed211bf`;
- archive coverage report SHA-256:
  `848089d2bde7cb2fc977953a8e79f233825d868f5c445f960937970f3e4f3eca`;
- pilot report SHA-256:
  `4fa0fc4352e92630707c8ee46429145dbff0de53e6e954e2fc0ba6dd59f2679f`;
- authenticated requests: `0`;
- exchange mutations: `0`;
- Live/Shadow/PM2 changes: `NONE`.

## Next evaluation

Run the frozen engine across all 11 symbols and all retrospective partitions,
then compare each family/direction against time- and regime-matched random
controls. Only families passing without threshold changes may enter fresh
forward confirmation.
