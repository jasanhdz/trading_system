from aegis.research.recent_strategy_atlas_w16 import variants


def test_variant_family_is_small_and_predeclared():
    config = {"families": {
        "mean_reversion": {"extension_atr": [1], "rsi_extremes": [[30, 70]], "maximum_abs_ema25_slope_atr": [.2], "maximum_hold_bars": [6]},
        "trend_pullback": {"minimum_ema25_slope_atr": [.1], "maximum_hold_bars": [6]},
        "breakout": {"lookback_bars": [12], "minimum_volume_ratio": [1.5], "maximum_hold_bars": [6]},
    }}
    result = variants(config)
    assert [item.family for item in result] == ["mean_reversion", "trend_pullback", "breakout"]
