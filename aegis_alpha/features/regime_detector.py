from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Regime:
    type: str
    confidence: float


def detect_regime(latest_features: np.ndarray) -> Regime:
    row = latest_features[-1] if latest_features.ndim == 2 else latest_features
    ema_1h = float(row[11])
    ema_4h = float(row[12])
    adx_norm = float(row[18])
    trend_eff = float(row[19])
    vol_regime = float(row[20])

    if vol_regime > 0.45:
        return Regime("high_vol", min(0.95, 0.55 + vol_regime * 0.35))
    if adx_norm < -0.65 and trend_eff < 0.45:
        return Regime("chop", 0.70)
    if ema_1h > 0.25 and ema_4h > -0.05 and adx_norm > -0.55:
        return Regime("trend_up", min(0.90, 0.55 + abs(ema_1h) * 0.08))
    if ema_1h < -0.25 and ema_4h < 0.05 and adx_norm > -0.55:
        return Regime("trend_down", min(0.90, 0.55 + abs(ema_1h) * 0.08))
    if vol_regime < -0.40 and adx_norm < -0.45:
        return Regime("compression", 0.62)
    return Regime("mixed", 0.50)
