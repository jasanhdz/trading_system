from __future__ import annotations

import numpy as np

from aegis_alpha.config import RiskConfig
from aegis_alpha.env.action_mask import CLOSE, IDLE, LONG, SHORT, ActionMaskConfig, coerce_action
from aegis_alpha.env.risk_engine import Position, close_position, current_roe, open_position
from aegis_alpha.env.reward import step_reward


class AegisEnv:
    """Small deterministic env for smoke tests and BC/Coliseum foundations."""

    def __init__(self, features: np.ndarray, close_prices: np.ndarray, risk: RiskConfig | None = None, window_size: int = 64):
        self.features = features.astype(np.float32)
        self.close_prices = close_prices.astype(np.float32)
        self.risk = risk or RiskConfig()
        self.mask_cfg = ActionMaskConfig(self.risk.min_hold_steps, self.risk.min_flat_steps)
        self.window_size = window_size
        self.reset()

    def reset(self, start_step: int | None = None) -> dict[str, np.ndarray]:
        self.step_idx = max(self.window_size, start_step or self.window_size)
        self.balance = self.risk.initial_balance
        self.position = Position()
        self.hold_steps = 0
        self.flat_steps = self.risk.min_flat_steps
        self.total_fees = 0.0
        self.opens = 0
        self.closes = 0
        self.invalid_actions = 0
        self.peak_equity = self.risk.initial_balance
        self.max_dd = 0.0
        return self._obs()

    def equity(self, price: float | None = None) -> float:
        price = float(self.close_prices[self.step_idx]) if price is None else price
        if self.position.side == 0:
            return self.balance
        pnl = abs(self.position.size) * (
            price - self.position.entry_price if self.position.side > 0 else self.position.entry_price - price
        )
        return self.balance + pnl

    def _obs(self) -> dict[str, np.ndarray]:
        start = self.step_idx - self.window_size
        market = self.features[start:self.step_idx]
        roe = current_roe(self.position, float(self.close_prices[self.step_idx]), self.risk)
        account = np.array(
            [
                self.equity() / self.risk.initial_balance,
                abs(self.position.size) * self.close_prices[self.step_idx] / max(self.equity(), 1e-10),
                roe,
                1.0 if self.position.side != 0 else 0.0,
                self.hold_steps / 288.0,
                roe,
            ],
            dtype=np.float32,
        )
        return {"market": market, "account": account}

    def step(self, raw_action: int):
        price = float(self.close_prices[self.step_idx])
        prev_equity = self.equity(price)
        action, invalid = coerce_action(raw_action, self.position.side, self.hold_steps, self.flat_steps, self.mask_cfg)
        opened = False

        if action in (LONG, SHORT) and self.position.side == 0:
            side = 1 if action == LONG else -1
            self.balance, self.position, fee = open_position(self.balance, side, price, self.step_idx, self.risk)
            opened = self.position.side != 0
            self.opens += int(opened)
            self.total_fees += fee
            self.hold_steps = 0
            self.flat_steps = 0
        elif action == CLOSE and self.position.side != 0:
            self.balance, _, fee = close_position(self.balance, self.position, price, self.risk)
            self.position = Position()
            self.closes += 1
            self.total_fees += fee
            self.hold_steps = 0
            self.flat_steps = 0
        elif action == IDLE:
            pass

        if invalid:
            self.invalid_actions += 1

        self.step_idx += 1
        if self.position.side == 0:
            self.flat_steps += 1
            self.hold_steps = 0
        else:
            self.hold_steps += 1
            self.flat_steps = 0

        new_equity = self.equity()
        self.peak_equity = max(self.peak_equity, new_equity)
        self.max_dd = max(self.max_dd, (self.peak_equity - new_equity) / max(self.peak_equity, 1e-10))
        done = self.step_idx >= len(self.close_prices) - 1 or new_equity <= self.risk.initial_balance * 0.1
        reward = step_reward(prev_equity, new_equity, invalid, opened, self.risk)
        info = {
            "equity": new_equity,
            "balance": self.balance,
            "fees": self.total_fees,
            "opens": self.opens,
            "closes": self.closes,
            "invalid_actions": self.invalid_actions,
            "max_dd": self.max_dd,
        }
        return self._obs(), reward, done, info
