from __future__ import annotations

from datetime import datetime

from .candidates import RangeCandidate
from .models import Episode, LevelCluster, LevelSnapshot, RangePair
from .numeric import range_episode_id, range_id


class RangeDetectorV1:
    """Owns operable-range episode identity independently from trade lifecycle."""

    def __init__(self, symbol: str, candidate: RangeCandidate):
        self.symbol = symbol
        self.candidate = candidate
        self.episode: Episode | None = None

    def end(self, decision_at: datetime | None, reason: str) -> str | None:
        if self.episode is None:
            return None
        identifier = self.episode.range_episode_id
        self.episode.ended_at = decision_at
        self.episode.end_reason = reason
        self.episode = None
        return identifier

    def confirm(self, pair: RangePair, decision_at: datetime, atr14_raw: float) -> Episode:
        identifier = range_episode_id(
            self.symbol,
            decision_at,
            pair.support_cluster_id,
            pair.resistance_cluster_id,
        )
        snapshot = LevelSnapshot(
            decision_at,
            pair,
            atr14_raw,
            identifier,
            range_id(identifier, decision_at, pair.support, pair.resistance, pair.midpoint),
        )
        self.episode = Episode(
            self.symbol,
            identifier,
            decision_at,
            pair.support_cluster_id,
            pair.resistance_cluster_id,
            snapshot,
        )
        return self.episode

    def active_pair(self, clusters: dict[str, LevelCluster]) -> RangePair | None:
        episode = self.episode
        if episode is None:
            return None
        support = clusters.get(episode.support_cluster_id)
        resistance = clusters.get(episode.resistance_cluster_id)
        if support is None or resistance is None:
            return None
        if len(support.pivots) < 2 or len(resistance.pivots) < 2 or len(support.touches) < 2 or len(resistance.touches) < 2:
            return None
        midpoint = (support.center + resistance.center) / 2.0
        amplitude = (resistance.center - support.center) / midpoint
        if not self.candidate.min_range_amplitude_pct <= amplitude <= 0.08:
            return None
        if support.last_touch_at is None or resistance.last_touch_at is None:
            return None
        old = episode.previous_snapshot.pair
        return RangePair(
            support.cluster_id,
            resistance.cluster_id,
            support.center,
            resistance.center,
            midpoint,
            amplitude,
            len(support.touches),
            len(resistance.touches),
            min(support.last_touch_at, resistance.last_touch_at),
            old.pair_first_eligible_at,
        )

    def winner_replaces_active(self, winner: RangePair | None) -> bool:
        return self.episode is not None and winner is not None and (
            winner.support_cluster_id != self.episode.support_cluster_id
            or winner.resistance_cluster_id != self.episode.resistance_cluster_id
        )
