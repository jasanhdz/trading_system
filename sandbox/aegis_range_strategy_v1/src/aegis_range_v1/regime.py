from __future__ import annotations

from typing import Any, Protocol

from .atr import RangeAtr14V1
from .data_adapter import RangeDataAdapter
from .models import Candle5m, RegimeSnapshot


class RegimeEvaluator(Protocol):
    def evaluate(self, *, symbol: str, candles: tuple[Candle5m, ...], timeframe: str) -> dict[str, Any]: ...


class RangeRegimeAdapter:
    """Read-only scientific boundary around an injected RegimeEngineV2 equivalent."""

    def __init__(self, evaluator: RegimeEvaluator):
        self._evaluator = evaluator

    def snapshot(self, symbol: str, history: list[Candle5m]) -> RegimeSnapshot:
        window = RangeDataAdapter.contiguous_window(history, 160)
        # No market keyword is sent: this is TypeScript market=undefined semantics.
        decision = self._evaluator.evaluate(symbol=symbol, candles=window, timeframe="5m")
        indicators = decision["indicators"]
        scores = decision["scores"]
        transition = decision["transition"]
        return RegimeSnapshot(
            technical_regime=decision["technicalRegime"],
            transition_risk=transition["risk"],
            adx=float(indicators["adx"]),
            atr_percentile=float(indicators["atrPercentile"]),
            bollinger_width_percentile=float(indicators["bollingerWidthPercentile"]),
            volume_ratio=float(indicators["volumeRatio"]),
            range_breakout=indicators["rangeBreakout"],
            failed_breakout_count=int(indicators["failedBreakoutCount"]),
            structure=indicators["structure"],
            chop_risk=float(scores["chopRisk"]),
            atr14_raw=RangeAtr14V1.calculate(window),
        )
