from __future__ import annotations

import numpy as np

from aegis_alpha.env.action_mask import CLOSE, IDLE, LONG, SHORT


def label_prudent_action(row: np.ndarray, position_side: int, hold_steps: int, flat_steps: int) -> int:
    ema_1h, ema_4h = float(row[11]), float(row[12])
    rsi = float(row[4])
    cvd_z, cvd_roc = float(row[8]), float(row[9])
    cvd_div = float(row[14])
    adx_norm = float(row[18])
    trend_eff = float(row[19])
    vol_regime = float(row[20])

    if position_side == 0:
        if flat_steps < 12:
            return IDLE
        long_ok = ema_1h > 0.12 and ema_4h > -0.25 and -0.55 <= rsi <= 0.65 and adx_norm > -0.70
        long_ok = long_ok and trend_eff > 0.40 and vol_regime < 0.55 and cvd_z > -1.0 and cvd_roc > -2.0 and cvd_div >= 0
        short_ok = ema_1h < -0.12 and ema_4h < 0.25 and -0.65 <= rsi <= 0.55 and adx_norm > -0.70
        short_ok = short_ok and trend_eff > 0.40 and vol_regime < 0.55 and cvd_z < 1.0 and cvd_roc < 2.0 and cvd_div <= 0
        if long_ok:
            return LONG
        if short_ok:
            return SHORT
        return IDLE

    if hold_steps < 6:
        return IDLE
    if position_side > 0 and (ema_1h < 0.05 or ema_4h < -0.20 or rsi > 0.65 or cvd_z < -0.80 or cvd_div < 0):
        return CLOSE
    if position_side < 0 and (ema_1h > -0.05 or ema_4h > 0.20 or rsi < -0.65 or cvd_z > 0.80 or cvd_div > 0):
        return CLOSE
    return IDLE
