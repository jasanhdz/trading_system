# Aegis Live Entry Multi-Timeframe Audit

## Verdict

- Episodes: 718; causal features: 104.
- Context: `1m`, `5m`, `15m`, `1h`, closed bars only.
- Validation allowed/total: 178/213.
- Validation bad-rate reduction: 6.2%.
- Clean-good retention: 82.2%.
- Bootstrap 95% CI: [-7.6%, 21.2%].
- `MULTITIMEFRAME_BAD_ENTRY_FILTER_FOUND = FALSE`.
- `PRODUCTION_CHANGE_JUSTIFIED = FALSE`.

## Discovery differences

| Feature | Good median | Bad median | Standardized difference |
|---|---:|---:|---:|
| `tf15m__atr_pct_bps` | 46.3626 | 55.7276 | +0.456 |
| `tf1m__atr_pct_bps` | 10.2657 | 13.1824 | +0.424 |
| `tf5m__atr_pct_bps` | 25.3518 | 30.6100 | +0.388 |
| `tf5m__distance_recent_low_atr` | 3.4140 | 4.2868 | +0.311 |
| `tf15m__distance_recent_high_atr` | 3.5280 | 2.6908 | -0.307 |
| `tf15m__atr_percentile_96` | 0.5208 | 0.6146 | +0.284 |
| `tf60m__atr_pct_bps` | 95.0992 | 104.6203 | +0.257 |
| `tf15m__ema7_slope_atr` | -0.0826 | 0.0930 | +0.256 |
| `tf5m__prior_move_6_atr` | 0.2804 | 0.6732 | +0.239 |
| `tf60m__atr_percentile_96` | 0.5000 | 0.5729 | +0.233 |
| `tf15m__ema25_slope_atr` | -0.0456 | 0.0338 | +0.229 |
| `tf1m__volume_z50` | -0.0816 | 0.1627 | +0.226 |
| `tf5m__distance_recent_high_atr` | 3.6120 | 3.0806 | -0.221 |
| `tf15m__volume_z50` | 0.0615 | 0.3461 | +0.214 |
| `tf15m__distance_recent_low_atr` | 3.4491 | 3.9266 | +0.207 |

These are descriptive TRAIN differences. Only the frozen validation result determines whether they can support a filter.

## Direction-aware descriptive stability

| Feature | Discovery effect | Validation effect |
|---|---:|---:|
| `dir1m__return_1_bps` | -0.296 | -0.209 |
| `dir1m__taker_imbalance` | -0.192 | -0.817 |
| `dir5m__prior_move_6_atr` | -0.179 | -0.175 |
| `dir15m__prior_move_6_atr` | -0.239 | -0.172 |
| `dir1m__return_6_bps` | -0.192 | -0.171 |
| `dir1m__adverse_space_atr` | -0.151 | -0.405 |
| `dir15m__return_6_bps` | -0.149 | -0.201 |
| `dir60m__adverse_space_atr` | +0.141 | +0.496 |
| `dir5m__return_6_bps` | -0.173 | -0.133 |
| `dir15m__taker_imbalance` | -0.124 | -0.208 |
| `dir1m__ema99_extension_atr` | -0.335 | -0.118 |
| `dir15m__ema7_slope_atr` | -0.327 | -0.117 |
| `dir1m__prior_move_6_atr` | -0.134 | -0.093 |
| `dir60m__favorable_space_atr` | -0.086 | -0.386 |
| `dir15m__return_3_bps` | -0.074 | -0.107 |

Positive values mean BAD entries had a larger value than clean GOOD entries after orienting the feature to the trade side. This table is descriptive and was produced after the primary model failed; it cannot promote a guard.
