from aegis_alpha.event_risk.event_risk_detector import evaluate_event_risk_auto


def ctx(symbol: str, action: str = "LONG", score: float = 0.72, **kwargs):
    return {
        "symbol": symbol,
        "turbo_action": action,
        "turbo_score": score,
        "freshness": {"is_fresh": True, "feature_age_seconds": 10, "max_feature_age_seconds": 300},
        **kwargs,
    }


def test_normal_context_suggests_normal():
    result = evaluate_event_risk_auto({
        "symbol": "SOLUSDT",
        "btc": ctx("BTCUSDT"),
        "eth": ctx("ETHUSDT"),
        "market": {"alt_signal_count": 4, "alt_hold_count": 0, "alt_block_shadow_count": 0},
    })

    assert result["suggested_mode"] == "NORMAL"
    assert result["execute"] is False
    assert result["production_allowed"] is False


def test_btc_eth_weak_suggests_risk_off():
    result = evaluate_event_risk_auto({
        "symbol": "SOLUSDT",
        "btc": ctx("BTCUSDT", "HOLD", 0.20, recent_return_15m=-0.03),
        "eth": ctx("ETHUSDT", "HOLD", 0.25, recent_return_15m=-0.02, atr_percentile=0.90),
        "market": {"alt_signal_count": 6, "alt_hold_count": 3, "alt_block_shadow_count": 4},
    })

    assert result["suggested_mode"] == "RISK_OFF"
    assert "btc_weak_or_hold" in result["reasons"]


def test_single_weak_major_suggests_caution():
    result = evaluate_event_risk_auto({
        "symbol": "SOLUSDT",
        "btc": ctx("BTCUSDT", "HOLD", 0.30),
        "eth": ctx("ETHUSDT", "LONG", 0.70),
        "market": {"alt_signal_count": 4, "alt_hold_count": 1, "alt_block_shadow_count": 0},
    })

    assert result["suggested_mode"] == "CAUTION"


def test_missing_data_suggests_caution_or_manual_only_and_never_executes():
    result = evaluate_event_risk_auto({
        "symbol": "SOLUSDT",
        "btc": None,
        "eth": ctx("ETHUSDT", "LONG", 0.70),
    })

    assert result["suggested_mode"] in {"CAUTION", "MANUAL_ONLY"}
    assert result["execute"] is False
    assert result["does_not_change_event_risk_mode"] is True

