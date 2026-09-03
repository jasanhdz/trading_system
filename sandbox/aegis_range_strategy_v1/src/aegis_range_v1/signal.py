from __future__ import annotations

from .candidates import RangeCandidate
from .models import Candle5m, RangePair, Signal


class RangeSignalV1:
    @staticmethod
    def evaluate(
        candle: Candle5m,
        pair: RangePair,
        atr14_raw: float,
        candidate: RangeCandidate,
        counted_cluster_ids: frozenset[str],
    ) -> Signal:
        body_floor = max(abs(candle.close - candle.open), 0.01 * atr14_raw)
        long_valid = (
            pair.support_cluster_id in counted_cluster_ids
            and candle.close > candle.open
            and pair.support < candle.close <= pair.midpoint
            and (min(candle.open, candle.close) - candle.low) / body_floor >= candidate.rejection_min_wick_body_ratio
        )
        short_valid = (
            pair.resistance_cluster_id in counted_cluster_ids
            and candle.close < candle.open
            and pair.midpoint <= candle.close < pair.resistance
            and (candle.high - max(candle.open, candle.close)) / body_floor >= candidate.rejection_min_wick_body_ratio
        )
        if long_valid == short_valid:
            return Signal("NONE", candle.available_at, "NO_UNAMBIGUOUS_COUNTED_TOUCH_REJECTION")
        return Signal("LONG" if long_valid else "SHORT", candle.available_at, "COUNTED_TOUCH_REJECTION")
