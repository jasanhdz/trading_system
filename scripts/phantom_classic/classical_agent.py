#!/usr/bin/env python3
"""
ClassicalCVD_MTF_Agent — Versión Corregida y Paritaria con Phantom
Usa exactamente las mismas 18 features que el RL.
"""

import numpy as np


class ClassicalAgent:
    """Agente clásico rule-based usando exactamente las mismas features que Phantom."""

    def __init__(self):
        self.name = "ClassicCVD_MTF_v2"

    def predict(self, obs, deterministic: bool = True):
        market = obs['market']      # (num_envs, window, 18)
        account = obs['account']    # (num_envs, 4)

        current = market[:, -1, :]  # última vela
        num_envs = current.shape[0]

        # Desempaquetar features correctamente según tu definición real
        log_ret         = current[:, 0]
        rsi_norm        = current[:, 4]
        ema_9_norm      = current[:, 5]
        ema_21_norm     = current[:, 6]
        ema_200_norm    = current[:, 7]
        cvd_z           = current[:, 8]
        cvd_roc         = current[:, 9]
        ema_1h_slope    = current[:, 11]
        ema_4h_slope    = current[:, 12]
        vol_z_arr       = current[:, 13]
        cvd_div         = current[:, 14]
        ema_1h_accel    = current[:, 15]
        ema_4h_accel    = current[:, 16]
        cvd_accel       = current[:, 17]

        in_trade = account[:, 3] > 0.5
        pnl_pct  = account[:, 2]

        actions = np.zeros(num_envs, dtype=np.int64)

        # ── Long Conditions ──
        mtf_bull = (ema_4h_slope > 0.15) & (ema_1h_slope > 0.20)
        accel_bull = (ema_1h_accel > 0.5) | (ema_4h_accel > 0.3)
        cvd_confirm = cvd_div > 0.10
        vol_confirm = vol_z_arr > 0.3
        not_exhausted = rsi_norm < 0.6

        go_long = (~in_trade) & mtf_bull & accel_bull & cvd_confirm & vol_confirm & not_exhausted

        # ── Short Conditions ──
        mtf_bear = (ema_4h_slope < -0.15) & (ema_1h_slope < -0.20)
        accel_bear = (ema_1h_accel < -0.5) | (ema_4h_accel < -0.3)
        cvd_confirm_short = cvd_div < -0.10
        not_exhausted_short = rsi_norm > -0.6

        go_short = (~in_trade) & mtf_bear & accel_bear & cvd_confirm_short & not_exhausted_short

        # ── Exit Conditions ──
        exit_long = in_trade & (pnl_pct > 0) & ((rsi_norm > 0.55) | (ema_1h_slope < 0))
        exit_short = in_trade & (pnl_pct < 0) & ((rsi_norm < -0.55) | (ema_1h_slope > 0))

        actions[go_long] = 1
        actions[go_short] = 2
        actions[exit_long | exit_short] = 3

        return actions, None

    @staticmethod
    def load(*args, **kwargs):
        return ClassicalAgent()
