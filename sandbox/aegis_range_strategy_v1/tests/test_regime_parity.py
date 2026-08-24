from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from aegis_range_v1.atr import RangeAtr14V1
from aegis_range_v1.models import Candle5m
from aegis_range_v1.regime import RangeRegimeAdapter
from aegis_range_v1.regime_bridge import TypeScriptRegimeEvaluator


def _fixture_candles(spec):
    origin = datetime.fromisoformat(spec["origin"].replace("Z", "+00:00"))
    candles = []
    for index in range(160):
        if spec["shape"] == "oscillating":
            close = 100.0 + ((index % 12) - 6) * 0.1
        elif spec["shape"] == "uptrend":
            close = 100.0 + index * 0.2
        else:
            close = 100.0 + ((index % 6) - 3) * 0.02 if index < 150 else 100.0 + (index - 149) * 0.8
        open_time = origin + timedelta(minutes=5 * index)
        candles.append(Candle5m("BTCUSDT", open_time, open_time + timedelta(minutes=5), close - 0.05, close + 0.4, close - 0.4, close, 100.0 + index, 0))
    return candles


def test_typescript_golden_and_python_adapter_exact_parity():
    root = Path(__file__).resolve().parents[3]
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/regime_parity_golden_v1.json").read_text(encoding="utf-8"))
    evaluator = TypeScriptRegimeEvaluator(root)
    for case in fixture["cases"]:
        candles = _fixture_candles(case)
        decision = evaluator.evaluate(symbol="BTCUSDT", candles=tuple(candles), timeframe="5m")
        assert decision == case["expected_decision"]
        published = [decision["confidence"], *decision["scores"].values()]
        published.extend(value for value in decision["indicators"].values() if isinstance(value, float))
        assert all(value == round(value, 6) for value in published)
        snapshot = RangeRegimeAdapter(evaluator).snapshot("BTCUSDT", candles)
        expected = case["expected_snapshot"]
        assert snapshot.technical_regime == expected["technical_regime"]
        assert snapshot.transition_risk == expected["transition_risk"]
        assert snapshot.failed_breakout_count == expected["failed_breakout_count"]
        for field in ("adx", "atr_percentile", "bollinger_width_percentile", "volume_ratio", "chop_risk"):
            assert getattr(snapshot, field) == expected[field]
        assert snapshot.atr14_raw.hex() == case["atr14_raw_hex"]
        assert json.loads(json.dumps(snapshot.atr14_raw)) == RangeAtr14V1.calculate(candles)
