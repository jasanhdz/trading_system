from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SHADOW_MODE = "SHADOW_ONLY"


@dataclass(frozen=True)
class ShadowSignal:
    candidate: str
    candidate_status: str
    live_enabled: bool
    enabled: bool
    strategy_name: str
    strategy_version: str
    mode: str
    symbol: str
    timestamp: str
    action: str
    would_execute: bool
    execute: bool
    reason: str
    size_mode: str
    position_fraction: float
    edge_score_h12: float
    tail_risk_score: float
    regime: str
    risk_tier: str
    model_status: str
    not_live_reason: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = SHADOW_MODE
        payload["execute"] = False
        payload["live_enabled"] = False
        return payload


def build_shadow_signal(
    *,
    candidate: str = "",
    candidate_status: str = "UNKNOWN",
    live_enabled: bool = False,
    enabled: bool = True,
    strategy_name: str = "",
    strategy_version: str = "",
    symbol: str = "",
    timestamp: str = "",
    action: str = "HOLD",
    would_execute: bool | None = None,
    reason: str = "",
    size_mode: str = "blocked",
    position_fraction: float = 0.0,
    edge_score_h12: float = 0.0,
    tail_risk_score: float = 0.0,
    regime: str = "unknown",
    risk_tier: str = "blocked",
    model_status: str = "UNKNOWN",
    not_live_reason: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    shadow_would_execute = bool(would_execute) if would_execute is not None else str(action).upper() not in {"HOLD", "IDLE"}
    signal = ShadowSignal(
        candidate=str(candidate or strategy_name),
        candidate_status=str(candidate_status or model_status),
        live_enabled=False,
        enabled=bool(enabled),
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        mode=SHADOW_MODE,
        symbol=symbol,
        timestamp=str(timestamp),
        action=str(action),
        would_execute=shadow_would_execute,
        execute=False,
        reason=str(reason),
        size_mode=str(size_mode),
        position_fraction=float(position_fraction),
        edge_score_h12=float(edge_score_h12),
        tail_risk_score=float(tail_risk_score),
        regime=str(regime),
        risk_tier=str(risk_tier),
        model_status=str(model_status),
        not_live_reason=[str(item) for item in not_live_reason],
    )
    return signal.to_dict()
