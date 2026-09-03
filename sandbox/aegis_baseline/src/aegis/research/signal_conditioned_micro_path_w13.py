"""Fail-closed W13 sample audit and passive micro-path capture primitives."""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class W13SampleRequirements:
    minimum_train_episodes: int = 1_000
    minimum_validation_episodes: int = 500
    minimum_symbols_per_partition: int = 4
    minimum_temporal_days_per_partition: int = 3
    minimum_directions_per_partition: int = 1


def stable_signal_episode_id(symbol: str, side: str, timestamp_us: int) -> str:
    normalized_symbol = symbol.upper()
    normalized_side = side.upper()
    if not normalized_symbol or normalized_side not in {"LONG", "SHORT"} or timestamp_us <= 0:
        raise ValueError("AEGIS_W13_SIGNAL_ID_INVALID")
    material = f"W13|{normalized_symbol}|{normalized_side}|{timestamp_us}".encode()
    return "W13-" + hashlib.sha256(material).hexdigest()


def assess_sample_gate(
    partition_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    requirements: W13SampleRequirements = W13SampleRequirements(),
) -> dict[str, Any]:
    observed: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    minima = {
        "W13_TRAIN": requirements.minimum_train_episodes,
        "W13_VALIDATION": requirements.minimum_validation_episodes,
    }
    for partition in ("W13_TRAIN", "W13_VALIDATION"):
        rows = list(partition_rows.get(partition, ()))
        symbols = {str(row["symbol"]).upper() for row in rows}
        days = {str(row["date"]) for row in rows}
        directions = {str(row["side"]).upper() for row in rows}
        observed[partition] = {
            "episodes": len(rows),
            "symbols": len(symbols),
            "temporal_days": len(days),
            "directions": len(directions),
            "symbol_values": sorted(symbols),
            "day_values": sorted(days),
            "direction_values": sorted(directions),
        }
        checks = {
            "EPISODES": (len(rows), minima[partition]),
            "SYMBOLS": (len(symbols), requirements.minimum_symbols_per_partition),
            "TEMPORAL_DAYS": (len(days), requirements.minimum_temporal_days_per_partition),
            "DIRECTIONS": (len(directions), requirements.minimum_directions_per_partition),
        }
        blockers.extend(
            f"INSUFFICIENT_{partition}_{name}:{actual}<{required}"
            for name, (actual, required) in checks.items()
            if actual < required
        )
    return {"passes": not blockers, "observed": observed, "blockers": blockers}


@dataclass(frozen=True)
class PassiveSignal:
    episode_id: str
    symbol: str
    side: str
    signal_exchange_timestamp_us: int
    signal_local_timestamp_us: int


class PassiveMicroPathCollector:
    """Inert event sink: no sockets, credentials, decisions, or execution hooks."""

    def __init__(self, *, pre_signal_seconds: int = 30, post_signal_seconds: int = 180) -> None:
        if pre_signal_seconds < 0 or post_signal_seconds <= 0:
            raise ValueError("AEGIS_W13_COLLECTOR_WINDOW_INVALID")
        self.pre_us = pre_signal_seconds * 1_000_000
        self.post_us = post_signal_seconds * 1_000_000
        self._buffers: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self._active: dict[str, dict[str, Any]] = {}
        self._seen_events: set[str] = set()

    @staticmethod
    def _event(event: Mapping[str, Any]) -> dict[str, Any]:
        required = {"event_id", "event_type", "symbol", "exchange_timestamp_us", "local_receive_timestamp_us"}
        if not required.issubset(event):
            raise ValueError("AEGIS_W13_EVENT_SCHEMA_INVALID")
        normalized = dict(event)
        normalized["symbol"] = str(normalized["symbol"]).upper()
        normalized["event_type"] = str(normalized["event_type"]).upper()
        if normalized["event_type"] not in {"BOOK", "QUOTE", "TRADE"}:
            raise ValueError("AEGIS_W13_EVENT_TYPE_INVALID")
        if int(normalized["exchange_timestamp_us"]) <= 0 or int(normalized["local_receive_timestamp_us"]) <= 0:
            raise ValueError("AEGIS_W13_EVENT_TIMESTAMP_INVALID")
        return normalized

    def observe_market_event(self, event: Mapping[str, Any]) -> None:
        normalized = self._event(event)
        event_id = str(normalized["event_id"])
        if event_id in self._seen_events:
            return
        self._seen_events.add(event_id)
        symbol = str(normalized["symbol"])
        now = int(normalized["exchange_timestamp_us"])
        buffer = self._buffers[symbol]
        buffer.append(normalized)
        while buffer and int(buffer[0]["exchange_timestamp_us"]) < now - self.pre_us:
            buffer.popleft()
        for episode in self._active.values():
            signal = episode["signal"]
            if signal.symbol == symbol and now <= signal.signal_exchange_timestamp_us + self.post_us:
                episode["events"].append(normalized)

    def observe_signal(
        self,
        *,
        symbol: str,
        side: str,
        signal_exchange_timestamp_us: int,
        signal_local_timestamp_us: int,
    ) -> str:
        episode_id = stable_signal_episode_id(symbol, side, signal_exchange_timestamp_us)
        if episode_id in self._active:
            return episode_id
        signal = PassiveSignal(
            episode_id=episode_id,
            symbol=symbol.upper(),
            side=side.upper(),
            signal_exchange_timestamp_us=signal_exchange_timestamp_us,
            signal_local_timestamp_us=signal_local_timestamp_us,
        )
        pre_events = [
            event for event in self._buffers[signal.symbol]
            if signal_exchange_timestamp_us - self.pre_us
            <= int(event["exchange_timestamp_us"])
            <= signal_exchange_timestamp_us
        ]
        self._active[episode_id] = {"signal": signal, "events": pre_events}
        return episode_id

    def finalize(self, *, through_exchange_timestamp_us: int) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        for episode_id, episode in list(self._active.items()):
            signal: PassiveSignal = episode["signal"]
            if through_exchange_timestamp_us < signal.signal_exchange_timestamp_us + self.post_us:
                continue
            events = tuple(episode["events"])
            completed.append({
                "schema_id": "aegis-w13-passive-micro-path-episode-v1",
                "episode_id": episode_id,
                "symbol": signal.symbol,
                "side": signal.side,
                "signal_exchange_timestamp_us": signal.signal_exchange_timestamp_us,
                "signal_local_timestamp_us": signal.signal_local_timestamp_us,
                "events": events,
                "event_count": len(events),
                "capture_only": True,
                "execution_authority": False,
            })
            del self._active[episode_id]
        return completed
