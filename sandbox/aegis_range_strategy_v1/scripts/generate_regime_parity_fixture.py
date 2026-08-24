#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from aegis_range_v1.atr import RangeAtr14V1
from aegis_range_v1.models import Candle5m
from aegis_range_v1.regime import RangeRegimeAdapter
from aegis_range_v1.regime_bridge import TypeScriptRegimeEvaluator


def candles(shape: str) -> list[Candle5m]:
    origin = datetime(2030, 1, 1, tzinfo=timezone.utc)
    result = []
    for index in range(160):
        if shape == "oscillating":
            close = 100.0 + ((index % 12) - 6) * 0.1
        elif shape == "uptrend":
            close = 100.0 + index * 0.2
        else:
            close = 100.0 + ((index % 6) - 3) * 0.02 if index < 150 else 100.0 + (index - 149) * 0.8
        open_time = origin + timedelta(minutes=5 * index)
        result.append(Candle5m("BTCUSDT", open_time, open_time + timedelta(minutes=5), close - 0.05, close + 0.4, close - 0.4, close, 100.0 + index, 0))
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    evaluator = TypeScriptRegimeEvaluator(root)
    cases = []
    for shape in ("oscillating", "uptrend", "compression_breakout"):
        history = candles(shape)
        decision = evaluator.evaluate(symbol="BTCUSDT", candles=tuple(history), timeframe="5m")
        snapshot = RangeRegimeAdapter(evaluator).snapshot("BTCUSDT", history)
        cases.append(
            {
                "name": shape,
                "shape": shape,
                "origin": "2030-01-01T00:00:00Z",
                "expected_decision": decision,
                "expected_snapshot": {
                    "technical_regime": snapshot.technical_regime,
                    "transition_risk": snapshot.transition_risk,
                    "adx": snapshot.adx,
                    "atr_percentile": snapshot.atr_percentile,
                    "bollinger_width_percentile": snapshot.bollinger_width_percentile,
                    "volume_ratio": snapshot.volume_ratio,
                    "failed_breakout_count": snapshot.failed_breakout_count,
                    "chop_risk": snapshot.chop_risk,
                },
                "atr14_raw_hex": RangeAtr14V1.calculate(history).hex(),
            }
        )
    fixture = {
        "schema_version": "aegis-range-regime-v2-parity-golden-v1",
        "synthetic_only": True,
        "cases": cases,
    }
    output = root / "sandbox/aegis_range_strategy_v1/fixtures/regime_parity_golden_v1.json"
    output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
