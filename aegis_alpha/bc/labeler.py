from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aegis_alpha.env.action_mask import CLOSE, IDLE, LONG, SHORT


@dataclass(frozen=True)
class LabelerConfig:
    min_hold_steps: int = 6
    min_flat_steps: int = 12
    max_hold_steps: int = 12
    stop_loss_roe: float = -0.035
    take_profit_roe: float = 0.060
    long_ema_1h_min: float = 0.02
    short_ema_1h_max: float = -0.02
    adx_min: float = -0.85
    trend_eff_min: float = 0.20
    vol_regime_max: float = 0.90
    cvd_abs_max: float = 2.20


def label_prudent_action(
    row: np.ndarray,
    position_side: int,
    hold_steps: int,
    flat_steps: int,
    roe: float = 0.0,
    cfg: LabelerConfig | None = None,
) -> int:
    cfg = cfg or LabelerConfig()
    ema_1h, ema_4h = float(row[11]), float(row[12])
    rsi = float(row[4])
    cvd_z, cvd_roc = float(row[8]), float(row[9])
    cvd_div = float(row[14])
    adx_norm = float(row[18])
    trend_eff = float(row[19])
    vol_regime = float(row[20])

    if position_side == 0:
        if flat_steps < cfg.min_flat_steps:
            return IDLE
        regime_ok = adx_norm > cfg.adx_min and trend_eff > cfg.trend_eff_min and vol_regime < cfg.vol_regime_max
        long_ok = ema_1h > cfg.long_ema_1h_min and ema_4h > -0.38 and -0.68 <= rsi <= 0.72
        long_ok = long_ok and regime_ok and cvd_z > -cfg.cvd_abs_max and cvd_roc > -2.0 and cvd_div >= -0.5
        short_ok = ema_1h < cfg.short_ema_1h_max and ema_4h < 0.38 and -0.72 <= rsi <= 0.68
        short_ok = short_ok and regime_ok and cvd_z < cfg.cvd_abs_max and cvd_roc < 2.0 and cvd_div <= 0.5
        if long_ok:
            return LONG
        if short_ok:
            return SHORT
        return IDLE

    if hold_steps < cfg.min_hold_steps:
        return IDLE

    if roe <= cfg.stop_loss_roe:
        return CLOSE

    long_invalid = ema_1h < 0.04 or ema_4h < -0.22 or rsi > 0.68 or cvd_z < -0.90 or cvd_div < 0
    short_invalid = ema_1h > -0.04 or ema_4h > 0.22 or rsi < -0.68 or cvd_z > 0.90 or cvd_div > 0
    trend_faded = adx_norm < -0.70 or trend_eff < 0.30

    if position_side > 0 and long_invalid and (roe < -0.005 or roe >= cfg.take_profit_roe):
        return CLOSE
    if position_side < 0 and short_invalid and (roe < -0.005 or roe >= cfg.take_profit_roe):
        return CLOSE
    if roe > 0.0 and hold_steps >= 48 and trend_faded:
        return CLOSE
    if hold_steps >= cfg.max_hold_steps:
        return CLOSE
    return IDLE
