# E4 Source And Coverage Audit

## Core panel

The core source is the checksum-manifested Binance USD-M public 1-minute candle
panel under `data/independent_entry_quality_discovery_v1/candles_1m`.

- 11 symbols.
- 3,876,960 source minute rows in total; 352,800 per symbol except SUIUSDT
  (348,960 after its later listing start).
- No duplicate minutes and no gaps within each declared symbol interval.
- Fields include OHLCV, trade count, taker-buy base volume and taker-buy quote
  volume.
- The E4 development window is September 1 through December 6, 2023.

This supports causal 5m/15m/1h/4h candles and candle-aggregated taker flow. It
does not support tick-by-tick sequencing inside a minute.

## Optional sources

- L2: validated bundles exist only on isolated first days in 2024-2026. They do
  not form a matched continuous panel for E4.
- Open Interest: no broad authentic causal history overlaps the core panel.
- Liquidations: no broad clean causal stream overlaps the core panel.
- Funding: local public archives begin in 2024 for the audited source and do not
  match the 2023 core panel.

No proxy is presented as true L2, OI or liquidation information. Their
subexperiments are `INSUFFICIENT_CAUSAL_COVERAGE`.
