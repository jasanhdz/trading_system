from __future__ import annotations

from aegis_alpha.config import RiskConfig


def step_reward(prev_equity: float, new_equity: float, invalid_action: bool, opened: bool, cfg: RiskConfig) -> float:
    safe_prev = max(prev_equity, 1e-10)
    delta = (new_equity - prev_equity) / safe_prev
    reward = delta * (15.0 if delta > 0 else 12.0)
    if opened:
        reward -= 0.04
    if invalid_action:
        reward -= 0.02
    return max(-20.0, min(20.0, reward))
