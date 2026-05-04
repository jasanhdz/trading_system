from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SHADOW_MODE = "SHADOW_ONLY"


@dataclass(frozen=True)
class ShadowSignal:
    strategy_name: str
    strategy_version: str
    mode: str
    symbol: str
    timestamp: str
    action: str
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
        return payload


def build_shadow_signal(
    *,
    strategy_name: str,
    strategy_version: str,
    symbol: str,
    timestamp: str,
    action: str,
    reason: str,
    size_mode: str,
    position_fraction: float,
    edge_score_h12: float,
    tail_risk_score: float,
    regime: str,
    risk_tier: str,
    model_status: str,
    not_live_reason: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    signal = ShadowSignal(
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        mode=SHADOW_MODE,
        symbol=symbol,
        timestamp=str(timestamp),
        action=str(action),
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
