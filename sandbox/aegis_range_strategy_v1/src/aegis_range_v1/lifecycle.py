from __future__ import annotations

import math
from datetime import datetime, timedelta

from .breakout import RangeBreakoutV1
from .candidates import RangeCandidate
from .costs import BASELINE, adverse_fill
from .models import Candle5m, FillEvent, PendingEntry, Position, Side
from .thesis import build_thesis


class StateInvariantViolation(RuntimeError):
    pass


class RangeLifecycleV1:
    def __init__(self, candidate: RangeCandidate):
        self.candidate = candidate
        self.pending_entry: PendingEntry | None = None
        self.position: Position | None = None
        self.traded_sides: dict[str, set[Side]] = {}
        self.trade_counts: dict[str, int] = {}
        self.closed_bars_since_exit: int | None = None
        self.last_entry_cancel_reason: str | None = None

    def reset_episode(self, episode_id: str | None = None) -> None:
        if self.pending_entry is not None and (episode_id is None or self.pending_entry.range_episode_id == episode_id):
            self.pending_entry = None

    def quota_ready(self, episode_id: str, side: Side) -> bool:
        return self.trade_counts.get(episode_id, 0) < 2 and side not in self.traded_sides.get(episode_id, set())

    def cooldown_ready(self) -> bool:
        return self.closed_bars_since_exit is None or self.closed_bars_since_exit >= 12

    def schedule_entry(self, pending: PendingEntry) -> None:
        if self.position is not None or self.pending_entry is not None:
            raise StateInvariantViolation("STATE_INVARIANT_VIOLATION")
        if not self.quota_ready(pending.range_episode_id, pending.side):
            raise StateInvariantViolation("STATE_INVARIANT_VIOLATION")
        self.pending_entry = pending

    def assert_open_invariants(self) -> None:
        if self.pending_entry is not None and self.position is not None:
            raise StateInvariantViolation("STATE_INVARIANT_VIOLATION")

    def _entry_levels(self, pending: PendingEntry) -> tuple[float, float]:
        if pending.side == "LONG":
            target = pending.midpoint - self.candidate.target_buffer_atr * pending.atr_entry
            stop = pending.support - self.candidate.stop_buffer_atr * pending.atr_entry
        else:
            target = pending.midpoint + self.candidate.target_buffer_atr * pending.atr_entry
            stop = pending.resistance + self.candidate.stop_buffer_atr * pending.atr_entry
        return stop, target

    def consume_pending_entry(
        self,
        *,
        open_at: datetime,
        raw_open: float | None,
        same_split: bool,
        episode_active: bool,
        has_pending_exit: bool = False,
    ) -> Position | None:
        pending = self.pending_entry
        self.last_entry_cancel_reason = None
        if pending is None:
            return None
        if self.position is not None or has_pending_exit:
            raise StateInvariantViolation("STATE_INVARIANT_VIOLATION")
        invalid_reason = None
        if raw_open is None:
            invalid_reason = "MISSING_NEXT_BAR"
        elif not math.isfinite(raw_open) or raw_open <= 0:
            invalid_reason = "INVALID_OPEN"
        elif not same_split:
            invalid_reason = "OUTSIDE_SPLIT"
        elif open_at != pending.entry_available_at:
            invalid_reason = "NOT_NEXT_BAR_OPEN"
        elif open_at - pending.range_confirmed_at >= timedelta(hours=48):
            invalid_reason = "EPISODE_EXPIRED"
        elif not episode_active:
            invalid_reason = "EPISODE_ENDED"
        elif not pending.support < raw_open < pending.resistance:
            invalid_reason = "OPEN_OUTSIDE_RANGE"
        if invalid_reason is not None:
            self.last_entry_cancel_reason = invalid_reason
            self.pending_entry = None
            return None
        fill = adverse_fill(raw_open, pending.side, BASELINE.slippage_bps_per_side)
        stop, target = self._entry_levels(pending)
        favorable = target > fill if pending.side == "LONG" else target < fill
        distance = abs(target - fill) / fill
        risk = fill - stop if pending.side == "LONG" else stop - fill
        reward = target - fill if pending.side == "LONG" else fill - target
        if not favorable:
            self.last_entry_cancel_reason = "TARGET_NOT_FAVORABLE"
        elif distance < 0.0042:
            self.last_entry_cancel_reason = "TARGET_DISTANCE_LT_42_BPS"
        elif risk <= 0 or reward / risk < 1.0:
            self.last_entry_cancel_reason = "REWARD_RISK_LT_1"
        if self.last_entry_cancel_reason is not None:
            self.pending_entry = None
            return None
        thesis = build_thesis(pending, self.candidate, fill, stop, target)
        position = Position(
            symbol=pending.symbol,
            side=pending.side,
            entry_at=open_at,
            entry_fill=fill,
            range_episode_id=pending.range_episode_id,
            range_id=pending.range_id,
            range_confirmed_at=pending.range_confirmed_at,
            support_at_entry=pending.support,
            resistance_at_entry=pending.resistance,
            midpoint_at_entry=pending.midpoint,
            atr_entry=pending.atr_entry,
            stop_at_entry=stop,
            target_at_entry=target,
            thesis_serialized=thesis.serialized,
            thesis_feature_hash=thesis.sha256,
        )
        self.position = position
        self.pending_entry = None
        self.trade_counts[pending.range_episode_id] = self.trade_counts.get(pending.range_episode_id, 0) + 1
        self.traded_sides.setdefault(pending.range_episode_id, set()).add(pending.side)
        return position

    def _exit(self, candle: Candle5m, base_price: float, reason: str) -> FillEvent:
        position = self.position
        if position is None:
            raise StateInvariantViolation("STATE_INVARIANT_VIOLATION")
        transaction_side: Side = "SHORT" if position.side == "LONG" else "LONG"
        fill = adverse_fill(base_price, transaction_side, BASELINE.slippage_bps_per_side)
        event = FillEvent(position.symbol, position.side, candle.open_time, fill, reason)
        self.position = None
        self.closed_bars_since_exit = 0
        return event

    def process_position_open_and_intrabar(self, candle: Candle5m, *, include_open_gaps: bool = True) -> FillEvent | None:
        position = self.position
        if position is None:
            return None
        if position.pending_exit_reason == "TRADE_BREAKOUT":
            return self._exit(candle, candle.open, "TRADE_BREAKOUT")
        if position.pending_exit_reason == "MAX_HOLD":
            return self._exit(candle, candle.open, "MAX_HOLD")
        if position.side == "LONG":
            if include_open_gaps and candle.open <= position.stop_at_entry:
                return self._exit(candle, candle.open, "STOP_GAP")
            if include_open_gaps and candle.open >= position.target_at_entry:
                return self._exit(candle, position.target_at_entry, "TARGET_GAP")
            stop_touched = candle.low <= position.stop_at_entry
            target_touched = candle.high >= position.target_at_entry
        else:
            if include_open_gaps and candle.open >= position.stop_at_entry:
                return self._exit(candle, candle.open, "STOP_GAP")
            if include_open_gaps and candle.open <= position.target_at_entry:
                return self._exit(candle, position.target_at_entry, "TARGET_GAP")
            stop_touched = candle.high >= position.stop_at_entry
            target_touched = candle.low <= position.target_at_entry
        if stop_touched:
            return self._exit(candle, position.stop_at_entry, "STOP")
        if target_touched:
            return self._exit(candle, position.target_at_entry, "TARGET")
        return None

    def process_close(self, close: float) -> None:
        if self.closed_bars_since_exit is not None:
            self.closed_bars_since_exit += 1
        position = self.position
        if position is None:
            return
        position.closed_bars += 1
        breakout = RangeBreakoutV1.update_trade(position, close)
        if breakout:
            position.pending_exit_reason = "TRADE_BREAKOUT"
        elif position.closed_bars == 144:
            position.pending_exit_reason = "MAX_HOLD"
