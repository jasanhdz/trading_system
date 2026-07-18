# Aegis Features V2 and SHORT Labels

## Contract

`aegis-features-v2` contains the 39 v1 columns followed by 44 historically validated
SHORT/risk columns. Its ordered-name hash is
`2dc278b4353585fe22503233187e12832cabfd67e2a2e58f4cd683ee6f3b9454`.
V1 remains loadable only with its own names and hash; no v1-to-v2 conversion exists.

The 83-column result exceeds the plan's orientative 60-80 target by three because every
family explicitly marked `PORTAR` is retained. Absolute EMAs, redundant returns, and
red/green flags remain rejected.

## Ported feature matrix

All windows end at the coordinated closed candle `t`. Prior-high/low windows are shifted
and therefore end at `t-1`. Golden values for every row below are frozen in
`tests/fixtures/aegis_features_v2_golden.json`.

| Family | Features | Formula / window | Scientific use |
|---|---|---|---|
| SHORT breakdown | `breakdown_proxy_12/24`, `close_below_rolling_low_12/24` | positive distance below prior low; close comparison, prior 12/24 | setup |
| SHORT room | `distance_to_rolling_low_12/24`, `distance_to_rolling_high_12/24`, `room_to_fall_proxy_24` | close distance to prior extrema, divided by close | setup and room |
| Reclaim | `failed_breakdown_proxy`, `fake_breakdown_risk_proxy` | low sweeps prior low and close reclaims; lower wick > .35 for fake risk | setup/tail |
| Extension | `extension_down_proxy`, `exhaustion_down_proxy` | negative close-vs-EMA24; negative ret12 scaled by lower wick | setup/tail |
| Reversal/squeeze | `rebound_risk_proxy`, `squeeze_risk_proxy_causal` | lower wick + positive ret3; ret12 < -2 ATR12 + lower wick | setup/tail |
| Risk state | `immediate_reversal_risk_proxy`, `overextended_down_risk_proxy`, `low_room_to_fall_risk_proxy`, `high_wick_reclaim_risk_proxy`, `squeeze_plus_reclaim_risk_proxy` | deterministic combinations of wick, close location, ATR, EMA24, room and reclaim | TRRM |
| EMA distance | `close_vs_ema_6/12/24/48` | close / causal EMA - 1 | trend |
| EMA evolution | `ema_slope_6/24` | EMA(t) / EMA(t-3) - 1 | trend |
| Trend stack | `trend_stack_short/long`, `trend_compression` | EMA6 < or > EMA12 < or > EMA24; abs(EMA6-EMA24)/close | context |
| Asymmetric momentum | `downside_momentum_6`, `upside_momentum_6` | min/max(ret6, 0) | SHORT asymmetry |
| Range dispersion | `rolling_range_std_12/24`, `volatility_compression_12_24` | population std of range/close; mean12/mean24 | RV2 |
| Volume event | `volume_spike_12`, `volume_trend_12` | volume > 1.8 mean12; volume/volume(t-12)-1 | quality |
| Relative volatility | `high_vol_regime_proxy`, `low_vol_regime_proxy` | range mean24 vs q75/q25 of causal trailing 96 observations | regime context |
| Leader context | `btc/eth_volatility_12`, `btc/eth_trend_proxy` | leader range mean12 and SHORT EMA stack at the same timestamp | cross-sectional context |
| Pressure sequence | `consecutive_red_count`, `consecutive_green_count` | trailing run length from closed candles | sequence |

`momentum_acceleration_3_12` now means `ret_3 - ret_12`, matching the historical
definition. The previous `ret_3 - ret_12/4` had no documented unit or validated reason.

## Labels

`aegis-labels-short-v4` uses the signal candle close as the label entry and final candles
`[t+1, t+12]`. ECON deliberately uses next-bar-open and is a separate offline program.

- MFE: `max(0, (entry - min(low_future))/entry)`.
- MAE: `max(0, (max(high_future) - entry)/entry)`.
- Net quality: `MFE - MAE - round_trip_cost_fraction`.
- Default cost: two sides of 4 bps fee + 1 bp slippage = 0.001 price fraction.
- Historical ROE thresholds are divided by the documented reference leverage 20; labels
  and model outputs remain price fractions.
- Equal-time MFE/MAE is not favorable-first. Same-bar target and stop is an ambiguous stop.
- Partial, non-final, or gapped future windows are quarantined and never interpolated.
- There is no premium-symbol allowlist.

Training and inference call the same `DeterministicFeaturePipeline`. Labels are available
only in `aegis.training` and never enter runtime feature construction.
