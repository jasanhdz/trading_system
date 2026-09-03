from __future__ import annotations

from .models import Episode, LevelSnapshot, Position


class RangeBreakoutV1:
    @staticmethod
    def update_episode(episode: Episode, close: float) -> bool:
        previous = episode.previous_snapshot
        direction = None
        if close > previous.pair.resistance + 0.10 * previous.atr14_raw:
            direction = "UP"
        elif close < previous.pair.support - 0.10 * previous.atr14_raw:
            direction = "DOWN"
        if direction is None:
            episode.outside_direction = None
            episode.outside_count = 0
            return False
        if direction == episode.outside_direction:
            episode.outside_count += 1
        else:
            episode.outside_direction = direction
            episode.outside_count = 1
        return episode.outside_count == 2

    @staticmethod
    def update_trade(position: Position, close: float) -> bool:
        adverse = (
            close < position.support_at_entry - 0.10 * position.atr_entry
            if position.side == "LONG"
            else close > position.resistance_at_entry + 0.10 * position.atr_entry
        )
        position.adverse_breakout_count = position.adverse_breakout_count + 1 if adverse else 0
        return position.adverse_breakout_count == 2

    @staticmethod
    def publish_snapshot(episode: Episode, snapshot: LevelSnapshot) -> None:
        episode.previous_snapshot = snapshot
