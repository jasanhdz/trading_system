from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from aegis_alpha.env.action_mask import CLOSE, IDLE, LONG, SHORT

LabelerVariant = Literal["conservative", "edge", "ultra"]


@dataclass(frozen=True)
class LabelerConfig:
    variant: LabelerVariant = "conservative"
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


VARIANT_CONFIGS: dict[LabelerVariant, LabelerConfig] = {
    "conservative": LabelerConfig(
        variant="conservative",
        min_hold_steps=8,
        min_flat_steps=18,
        max_hold_steps=18,
        stop_loss_roe=-0.030,
        take_profit_roe=0.055,
        long_ema_1h_min=0.035,
        short_ema_1h_max=-0.035,
        adx_min=-0.65,
        trend_eff_min=0.28,
        vol_regime_max=0.78,
        cvd_abs_max=1.80,
    ),
    "edge": LabelerConfig(
        variant="edge",
        min_hold_steps=6,
        min_flat_steps=12,
        max_hold_steps=16,
        stop_loss_roe=-0.035,
        take_profit_roe=0.060,
        long_ema_1h_min=0.020,
        short_ema_1h_max=-0.020,
        adx_min=-0.85,
        trend_eff_min=0.20,
        vol_regime_max=0.90,
        cvd_abs_max=2.20,
    ),
    "ultra": LabelerConfig(
        variant="ultra",
        min_hold_steps=10,
        min_flat_steps=24,
        max_hold_steps=24,
        stop_loss_roe=-0.025,
        take_profit_roe=0.070,
        long_ema_1h_min=0.055,
        short_ema_1h_max=-0.055,
        adx_min=-0.45,
        trend_eff_min=0.36,
        vol_regime_max=0.68,
        cvd_abs_max=1.45,
    ),
}


def get_labeler_config(variant: LabelerVariant = "conservative") -> LabelerConfig:
    if variant not in VARIANT_CONFIGS:
        allowed = ", ".join(sorted(VARIANT_CONFIGS))
        raise ValueError(f"Unknown BC labeler variant {variant!r}; expected one of: {allowed}")
    return VARIANT_CONFIGS[variant]


def label_bc_action(
    row: np.ndarray,
    position_side: int,
    hold_steps: int,
    flat_steps: int,
    roe: float = 0.0,
    cfg: LabelerConfig | None = None,
    variant: LabelerVariant = "conservative",
) -> int:
    cfg = cfg or get_labeler_config(variant)
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


def label_prudent_action(
    row: np.ndarray,
    position_side: int,
    hold_steps: int,
    flat_steps: int,
    roe: float = 0.0,
    cfg: LabelerConfig | None = None,
) -> int:
    return label_bc_action(row, position_side, hold_steps, flat_steps, roe=roe, cfg=cfg, variant="edge")
