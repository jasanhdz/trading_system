# Directional Alpha V1 Source Audit

## Main panel

The existing Entry Quality V1 source covers only 2023-05 onward and its
development period cannot provide clean confirmation for this follow-up.
Directional Alpha V1 therefore uses checksum-verified Binance USD-M public
monthly 1-minute kline archives from 2022-01 through 2023-03.

These archives contain OHLCV, trade count, taker-buy base volume and taker-buy
quote volume. They support causal price, flow-effectiveness and cross-market
features, but not true tick-level flow or order-book reconstruction.

LTCUSDT and XRPUSDT each contain two missing intervals before TRAIN. Their
experimental sources are causally cropped after the last gap; they become
eligible only after rebuilding the complete 99-bar daily warmup. No missing
minute is repaired or interpolated.

## L2 audit

Locally validated Tardis L2/quote/trade bundles exist for six symbols on five
isolated days: 2024-09-01, 2025-03-01, 2025-09-01, 2025-12-01 and 2026-03-01.
All were already evaluated in W9.1. The 2024 day is inside the explicitly
contaminated Router V1 period; later days overlap sealed W2/W3 holdout windows.

They may remain engineering fixtures but are not eligible evidence here.

`L2_SUBEXPERIMENT_STATUS = NOT_RUN_NO_CLEAN_ELIGIBLE_L2_PERIOD`

## Positioning audit

- Historical funding exists in local public archives but is not required by
  the primary directional experiment.
- The general candle database has an empty market-data table for open interest.
- No broad, clean, causal liquidation stream aligns with the main panel.

No OI or liquidation proxy is invented.

`POSITIONING_DIRECTIONAL_ALPHA = NOT_RUN_INSUFFICIENT_AUTHENTIC_COVERAGE`

## Evidence status

This remains retrospective temporal-OOS discovery. It is independent of the
Entry Quality V1 train/calibration/validation dates, but it is not prospective
evidence and does not authorize opening any sealed holdout.
