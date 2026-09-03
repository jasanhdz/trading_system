from __future__ import annotations

from datetime import datetime, timedelta

from .candidates import RangeCandidate
from .models import Candle5m, LevelCluster, Pivot, RangePair, Touch
from .numeric import cluster_id, iso_utc_millis

LOOKBACK = timedelta(days=7)


class RangeLevelsV1:
    def __init__(self, symbol: str, candidate: RangeCandidate):
        self.symbol = symbol
        self.candidate = candidate
        self.clusters: dict[str, LevelCluster] = {}
        self._pair_first_eligible: dict[tuple[str, str], datetime] = {}
        self._bar_index = -1

    def reset(self) -> None:
        self.clusters.clear()
        self._pair_first_eligible.clear()
        self._bar_index = -1

    @staticmethod
    def detect_available_pivots(history: list[Candle5m]) -> tuple[Pivot, ...]:
        if len(history) < 5:
            return ()
        index = len(history) - 3
        current = history[index]
        neighbors = (history[index - 2], history[index - 1], history[index + 1], history[index + 2])
        pivots: list[Pivot] = []
        if all(current.high > candle.high for candle in neighbors):
            pivots.append(
                Pivot(
                    symbol=current.symbol,
                    side="HIGH",
                    price=current.high,
                    source_price=current.high_source or repr(current.high),
                    pivot_at=current.available_at,
                    available_at=history[index + 2].available_at,
                )
            )
        if all(current.low < candle.low for candle in neighbors):
            pivots.append(
                Pivot(
                    symbol=current.symbol,
                    side="LOW",
                    price=current.low,
                    source_price=current.low_source or repr(current.low),
                    pivot_at=current.available_at,
                    available_at=history[index + 2].available_at,
                )
            )
        return tuple(pivots)

    def expire(self, decision_at: datetime) -> None:
        cutoff = decision_at - LOOKBACK
        remove: list[str] = []
        for identifier, cluster in self.clusters.items():
            cluster.pivots = [pivot for pivot in cluster.pivots if pivot.pivot_at >= cutoff]
            cluster.touches = [touch for touch in cluster.touches if touch.touch_at >= cutoff]
            if not cluster.pivots:
                remove.append(identifier)
        for identifier in remove:
            del self.clusters[identifier]
        active_ids = set(self.clusters)
        self._pair_first_eligible = {
            key: value
            for key, value in self._pair_first_eligible.items()
            if key[0] in active_ids and key[1] in active_ids
        }

    def insert_pivot(self, pivot: Pivot, atr14_raw: float) -> LevelCluster:
        tolerance = self.candidate.cluster_tolerance_atr * atr14_raw
        compatible: list[tuple[float, datetime, str, LevelCluster]] = []
        for cluster in self.clusters.values():
            if cluster.side != pivot.side:
                continue
            distance = abs(pivot.price - cluster.center)
            if distance <= tolerance:
                normalized = distance / tolerance if tolerance > 0 else (0.0 if distance == 0 else float("inf"))
                compatible.append((normalized, cluster.first_pivot_at, cluster.cluster_id, cluster))
        if compatible:
            selected = min(compatible, key=lambda item: (item[0], item[1], item[2]))[3]
            selected.pivots.append(pivot)
            return selected
        identifier = cluster_id(pivot.symbol, pivot.side, pivot.pivot_at, pivot.source_price)
        cluster = LevelCluster(identifier, pivot.symbol, pivot.side, pivot.pivot_at, [pivot])
        self.clusters[identifier] = cluster
        return cluster

    def update_touches(self, candle: Candle5m, atr14_raw: float) -> frozenset[str]:
        self._bar_index += 1
        counted: set[str] = set()
        tau = self.candidate.cluster_tolerance_atr * atr14_raw
        for identifier in sorted(self.clusters):
            cluster = self.clusters[identifier]
            if len(cluster.pivots) < 2:
                continue
            level = cluster.center
            if cluster.side == "LOW":
                if candle.low > level + 2.0 * tau:
                    cluster.armed = True
                geometry = candle.low <= level + tau and candle.low >= level - 0.35 * atr14_raw and candle.close > level
            else:
                if candle.high < level - 2.0 * tau:
                    cluster.armed = True
                geometry = candle.high >= level - tau and candle.high <= level + 0.35 * atr14_raw and candle.close < level
            last_index = cluster.touches[-1].bar_index if cluster.touches else None
            separated = last_index is None or self._bar_index - last_index >= 6
            if geometry and cluster.armed and separated:
                cluster.touches.append(Touch(candle.available_at, self._bar_index, level))
                cluster.armed = False
                counted.add(identifier)
        return frozenset(counted)

    def build_pairs(self, close: float, decision_at: datetime) -> tuple[RangePair, ...]:
        supports = [cluster for cluster in self.clusters.values() if cluster.side == "LOW" and len(cluster.pivots) >= 2 and len(cluster.touches) >= 2]
        resistances = [cluster for cluster in self.clusters.values() if cluster.side == "HIGH" and len(cluster.pivots) >= 2 and len(cluster.touches) >= 2]
        pairs: list[RangePair] = []
        currently_eligible: set[tuple[str, str]] = set()
        for support in supports:
            for resistance in resistances:
                support_value = support.center
                resistance_value = resistance.center
                if not support_value < close < resistance_value:
                    continue
                midpoint = (support_value + resistance_value) / 2.0
                amplitude = (resistance_value - support_value) / midpoint
                if not self.candidate.min_range_amplitude_pct <= amplitude <= 0.08:
                    continue
                key = (support.cluster_id, resistance.cluster_id)
                currently_eligible.add(key)
                first_eligible = self._pair_first_eligible.setdefault(key, decision_at)
                support_touch = support.last_touch_at
                resistance_touch = resistance.last_touch_at
                if support_touch is None or resistance_touch is None:
                    continue
                pairs.append(
                    RangePair(
                        support.cluster_id,
                        resistance.cluster_id,
                        support_value,
                        resistance_value,
                        midpoint,
                        amplitude,
                        len(support.touches),
                        len(resistance.touches),
                        min(support_touch, resistance_touch),
                        first_eligible,
                    )
                )
        for key in set(self._pair_first_eligible) - currently_eligible:
            del self._pair_first_eligible[key]
        pairs.sort(
            key=lambda pair: (
                -min(pair.support_touches, pair.resistance_touches),
                -(pair.support_touches + pair.resistance_touches),
                -pair.pair_recency_at.timestamp(),
                pair.pair_first_eligible_at,
                pair.support_cluster_id,
                pair.resistance_cluster_id,
            )
        )
        return tuple(pairs)

    def advance(self, history: list[Candle5m], atr14_raw: float) -> tuple[tuple[RangePair, ...], frozenset[str]]:
        candle = history[-1]
        self.expire(candle.available_at)
        for pivot in sorted(self.detect_available_pivots(history), key=lambda item: (item.pivot_at, item.side, item.price)):
            self.insert_pivot(pivot, atr14_raw)
        counted = self.update_touches(candle, atr14_raw)
        return self.build_pairs(candle.close, candle.available_at), counted

    def structural_snapshot(self) -> tuple:
        return tuple(
            (
                identifier,
                cluster.side,
                tuple((iso_utc_millis(p.pivot_at), p.price) for p in cluster.pivots),
                tuple((iso_utc_millis(t.touch_at), t.level_at_touch) for t in cluster.touches),
                cluster.armed,
            )
            for identifier, cluster in sorted(self.clusters.items())
        )
