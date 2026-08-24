from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from typing import Any

from .breakout import RangeBreakoutV1
from .candidates import RangeCandidate
from .levels import RangeLevelsV1
from .data_adapter import RangeDataAdapter
from .detector import RangeDetectorV1
from .lifecycle import RangeLifecycleV1
from .models import Candle5m, Episode, LevelSnapshot, PendingEntry, RangePair
from .numeric import iso_utc_millis, range_id
from .regime import RangeRegimeAdapter
from .safety import RangeSafetyV1
from .signal import RangeSignalV1


class RangeEngineV1:
    def __init__(self, symbol: str, candidate: RangeCandidate, regime_adapter: RangeRegimeAdapter):
        self.symbol = symbol
        self.candidate = candidate
        self.regime_adapter = regime_adapter
        self.levels = RangeLevelsV1(symbol, candidate)
        self.detector = RangeDetectorV1(symbol, candidate)
        self.lifecycle = RangeLifecycleV1(candidate)
        self.history: list[Candle5m] = []
        self.range_history: list[Candle5m] = []
        self.outputs: list[dict[str, Any]] = []

    @property
    def episode(self) -> Episode | None:
        return self.detector.episode

    @episode.setter
    def episode(self, value: Episode | None) -> None:
        self.detector.episode = value

    def _end_episode(self, decision_at, reason: str) -> None:
        identifier = self.detector.end(decision_at, reason)
        self.lifecycle.reset_episode(identifier)
        self.levels.reset()
        self.range_history.clear()

    def on_data_integrity(self) -> None:
        self._end_episode(self.history[-1].available_at if self.history else None, "DATA_INTEGRITY")
        self.lifecycle = RangeLifecycleV1(self.candidate)
        self.history.clear()
        self.range_history.clear()

    def on_split_boundary(self) -> None:
        self._end_episode(self.history[-1].available_at if self.history else None, "SPLIT_BOUNDARY")
        self.lifecycle = RangeLifecycleV1(self.candidate)
        self.history = self.history[-160:]
        self.range_history.clear()

    def _record(self, output: dict[str, Any]) -> dict[str, Any]:
        output["structure"] = self.levels.structural_snapshot()
        output["episode"] = None if self.episode is None else (
            self.episode.range_episode_id,
            iso_utc_millis(self.episode.range_confirmed_at),
            self.episode.support_cluster_id,
            self.episode.resistance_cluster_id,
            self.episode.outside_direction,
            self.episode.outside_count,
        )
        output["pending_entry"] = None if self.lifecycle.pending_entry is None else asdict(self.lifecycle.pending_entry)
        output["position_thesis"] = None if self.lifecycle.position is None else (
            self.lifecycle.position.thesis_serialized,
            self.lifecycle.position.thesis_feature_hash,
        )
        self.outputs.append(output)
        return output

    def _active_pair(self) -> RangePair | None:
        return self.detector.active_pair(self.levels.clusters)

    def process(self, candle: Candle5m, *, same_split: bool = True, embargo: bool = False) -> dict[str, Any]:
        if candle.symbol != self.symbol:
            raise ValueError("symbol mismatch")
        try:
            RangeDataAdapter.validate_5m(candle)
        except ValueError:
            self.on_data_integrity()
            raise
        integrity_reset = bool(self.history) and (
            candle.segment_id != self.history[-1].segment_id
            or candle.open_time != self.history[-1].open_time + timedelta(minutes=5)
        )
        if integrity_reset:
            self.on_data_integrity()

        self.lifecycle.assert_open_invariants()
        exit_event = self.lifecycle.process_position_open_and_intrabar(candle)
        active = self.episode is not None and self.episode.ended_at is None
        entry = self.lifecycle.consume_pending_entry(
            open_at=candle.open_time,
            raw_open=candle.open,
            same_split=same_split,
            episode_active=active,
        )
        if entry is not None:
            entry_bar_exit = self.lifecycle.process_position_open_and_intrabar(candle, include_open_gaps=False)
            if entry_bar_exit is not None:
                exit_event = entry_bar_exit
        self.history.append(candle)
        self.range_history.append(candle)
        self.lifecycle.process_close(candle.close)
        output: dict[str, Any] = {
            "decision_at": iso_utc_millis(candle.available_at),
            "signal": "NONE",
            "episode_event": None,
            "entry_hash": entry.thesis_feature_hash if entry else None,
            "entry_cancel_reason": self.lifecycle.last_entry_cancel_reason,
            "exit_reason": exit_event.reason if exit_event else None,
            "data_integrity_reset": integrity_reset,
        }
        try:
            regime = self.regime_adapter.snapshot(self.symbol, self.history)
        except ValueError as error:
            if str(error) != "INSUFFICIENT_HISTORY":
                raise
            output["status"] = "INSUFFICIENT_HISTORY"
            return self._record(output)

        if self.episode is not None and RangeBreakoutV1.update_episode(self.episode, candle.close):
            output["episode_event"] = "CONFIRMED_BREAKOUT"
            self._end_episode(candle.available_at, "CONFIRMED_BREAKOUT")
            output["status"] = "NOT_OPERABLE"
            return self._record(output)
        if self.episode is not None and candle.available_at - self.episode.range_confirmed_at >= timedelta(hours=48):
            output["episode_event"] = "EXPIRED_48H"
            self._end_episode(candle.available_at, "EXPIRED_48H")
            output["status"] = "NOT_OPERABLE"
            return self._record(output)

        self.levels.expire(candle.available_at)
        if self.episode is not None and self._active_pair() is None:
            output["episode_event"] = "STRUCTURE_LOST"
            self._end_episode(candle.available_at, "STRUCTURE_LOST")
            output["status"] = "NOT_OPERABLE"
            return self._record(output)

        for pivot in sorted(self.levels.detect_available_pivots(self.range_history), key=lambda item: (item.pivot_at, item.side, item.price)):
            self.levels.insert_pivot(pivot, regime.atr14_raw)
        counted = self.levels.update_touches(candle, regime.atr14_raw)
        pairs = self.levels.build_pairs(candle.close, candle.available_at)
        winner = pairs[0] if pairs else None

        if self.detector.winner_replaces_active(winner):
            output["episode_event"] = "PAIR_REPLACED"
            self._end_episode(candle.available_at, "PAIR_REPLACED")
            output["status"] = "NOT_OPERABLE"
            return self._record(output)

        if self.episode is None and winner is not None and not embargo:
            self.detector.confirm(winner, candle.available_at, regime.atr14_raw)
            output["episode_event"] = "CONFIRMED"

        active_pair = self._active_pair()
        if self.episode is None or active_pair is None:
            output["status"] = "RANGE_CANDIDATE" if winner else "NOT_OPERABLE"
            return self._record(output)
        snapshot = LevelSnapshot(
            candle.available_at,
            active_pair,
            regime.atr14_raw,
            self.episode.range_episode_id,
            range_id(self.episode.range_episode_id, candle.available_at, active_pair.support, active_pair.resistance, active_pair.midpoint),
        )
        RangeBreakoutV1.publish_snapshot(self.episode, snapshot)
        episode_age = (candle.available_at - self.episode.range_confirmed_at).total_seconds() / 3600.0
        safety = RangeSafetyV1.evaluate(
            active_pair,
            regime,
            self.candidate,
            episode_age,
            episode_operable=True,
            flat=self.lifecycle.position is None and self.lifecycle.pending_entry is None,
            no_pending_exit=self.lifecycle.position is None or self.lifecycle.position.pending_exit_reason is None,
            cooldown_ready=self.lifecycle.cooldown_ready(),
            quota_ready=(
                self.lifecycle.quota_ready(self.episode.range_episode_id, "LONG")
                or self.lifecycle.quota_ready(self.episode.range_episode_id, "SHORT")
            ),
        )
        output["status"] = "OPERABLE_RANGE" if safety.allowed else "RANGE_CANDIDATE"
        output["score"] = safety.descriptive_score
        output["blocker_reason"] = None if safety.allowed else safety.reason
        if safety.allowed:
            signal = RangeSignalV1.evaluate(candle, active_pair, regime.atr14_raw, self.candidate, counted)
            side_quota_ready = signal.side == "NONE" or self.lifecycle.quota_ready(self.episode.range_episode_id, signal.side)
            output["signal"] = signal.side if side_quota_ready else "NONE"
            if not side_quota_ready:
                output["status"] = "RANGE_CANDIDATE"
                output["blocker_reason"] = "QUOTA"
            if signal.side != "NONE" and side_quota_ready:
                self.lifecycle.schedule_entry(
                    PendingEntry(
                        self.symbol,
                        signal.side,
                        candle.available_at,
                        candle.available_at,
                        self.episode.range_episode_id,
                        snapshot.range_id,
                        self.episode.range_confirmed_at,
                        active_pair.support,
                        active_pair.resistance,
                        active_pair.midpoint,
                        regime.atr14_raw,
                        regime.technical_regime,
                        safety.descriptive_score,
                        None,
                    )
                )
        output["range_id"] = snapshot.range_id
        output["range_episode_id"] = snapshot.range_episode_id
        return self._record(output)

    def deterministic_outputs(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.outputs)
