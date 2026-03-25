#!/usr/bin/env python3
"""
Phantom Exit Agent V2 — "Ride the Wave" Environment
====================================================
SYNCHRONIZED with TradingService.ts Phase 1 parameters.

The agent spawns inside an ALREADY OPEN trade (Long or Short).
Its job: detect momentum deceleration and decide when to CLOSE.

Live Bot Parity (Phase 1):
  - 20x Leverage
  - -50% ROE Hard Stop (-2.5% price)  
  - +150% ROE Take Profit (+7.5% price)
  - 8 hour max trade duration (96 x 5min candles)
  - Trailing: activates at 30% ROE, 15% callback
  - 0.04% Binance Futures taker fee

Observation Space (12 features):
  [0]  current_pnl         - Current leveraged PnL %
  [1]  mfe                 - Maximum Favorable Excursion (peak ROE)
  [2]  mae                 - Maximum Adverse Excursion (worst ROE)  
  [3]  time_decay          - Normalized time in trade (0→1)
  [4]  atr_norm            - ATR normalized by entry price * leverage
  [5]  drawdown_from_peak  - How much given back from MFE (0→1)
  [6]  roe_velocity        - Speed of ROE change (1st derivative)
  [7]  roe_acceleration    - Acceleration of ROE (2nd derivative)
  [8]  cvd_z               - CVD Z-score (microstructure flow)
  [9]  cvd_roc             - CVD Rate of Change
  [10] distance_to_tp      - Distance to TP as fraction (0→1)
  [11] volume_ratio        - Current volume vs rolling avg
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from collections import deque


# Phase 1 Live Parameters
LEVERAGE = 20
HARD_STOP_ROE = -0.50       # -50% ROE = -2.5% price @ 20x
TAKE_PROFIT_ROE = 1.50      # +150% ROE = +7.5% price @ 20x
TRAILING_ACTIVATION = 0.30  # Activate trailing at 30% ROE
TRAILING_CALLBACK = 0.15    # 15% callback from peak ROE
MAX_STEPS = 96              # 96 x 5min = 8 hours
FEE = 0.0004                # 0.04% Binance Futures taker fee

N_OBS_FEATURES = 12


class MatrixExitEnvV2(gym.Env):
    """
    Phantom Exit Agent V2 — Momentum-Aware Exit Environment.
    
    The agent spawns inside an already-open trade and must decide
    HOLD (0) or CLOSE (1) at each 5-minute candle.
    
    Key innovation: ROE velocity and acceleration features allow
    the agent to detect momentum dying before the price reverses.
    """
    
    def __init__(self, tensors, leverage=LEVERAGE, max_steps=MAX_STEPS, fee=FEE):
        super().__init__()
        
        # Load tensors
        self.close_prices = tensors['close'].numpy()
        self.high = tensors['high'].numpy()
        self.low = tensors['low'].numpy()
        self.atr = tensors['atr'].numpy()
        self.cvd_z_arr = tensors['cvd_z'].numpy()
        self.cvd_roc_arr = tensors['cvd_roc'].numpy()
        self.volume = tensors['volume'].numpy()
        self.volume_ma = tensors['volume_ma'].numpy()
        self.n_candles = tensors['n_candles']
        
        self.leverage = leverage
        self.max_steps = max_steps
        self.fee = fee
        
        # Action: 0 = HOLD, 1 = CLOSE
        self.action_space = spaces.Discrete(2)
        
        # 12D observation
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, 
            shape=(N_OBS_FEATURES,), 
            dtype=np.float32
        )
        
        # State
        self.current_step = 0
        self.entry_step = 0
        self.entry_price = 0.0
        self.side = 1.0  # 1 for Long, -1 for Short
        self.steps_taken = 0
        self.mfe = 0.0
        self.mae = 0.0
        
        # Momentum tracking (history of ROE for velocity/acceleration)
        self.roe_history = deque(maxlen=6)  # Last 6 candles = 30 min
        self.peak_roe = 0.0
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        
        # Spawn randomly, leaving room for max_steps
        self.entry_step = np.random.randint(0, self.n_candles - self.max_steps - 1)
        self.current_step = self.entry_step
        self.entry_price = self.close_prices[self.entry_step]
        
        # 50% Long / 50% Short
        self.side = 1.0 if np.random.rand() > 0.5 else -1.0
        
        self.steps_taken = 0
        self.mfe = 0.0
        self.mae = 0.0
        self.peak_roe = 0.0
        self.roe_history.clear()
        self.roe_history.append(0.0)  # Initial ROE = 0
        
        return self._get_obs(), {}
    
    def _calc_roe(self, price):
        """Calculate leveraged ROE from entry price."""
        raw_pct = ((price - self.entry_price) / self.entry_price) * self.side
        return raw_pct * self.leverage
    
    def _get_obs(self):
        idx = min(self.current_step, self.n_candles - 1)
        current_price = self.close_prices[idx]
        
        # [0] Current PnL (gross, no fees)
        current_pnl = self._calc_roe(current_price)
        
        # Update MFE / MAE
        if current_pnl > self.mfe:
            self.mfe = current_pnl
        if current_pnl < self.mae:
            self.mae = current_pnl
        
        # Update peak ROE
        if current_pnl > self.peak_roe:
            self.peak_roe = current_pnl
        
        # Track ROE history for velocity/acceleration
        self.roe_history.append(current_pnl)
        
        # [1] MFE
        mfe = self.mfe
        
        # [2] MAE
        mae = self.mae
        
        # [3] Time decay (0 → 1)
        time_decay = self.steps_taken / self.max_steps
        
        # [4] ATR normalized
        current_atr = self.atr[idx]
        atr_norm = (current_atr / max(self.entry_price, 1e-10)) * self.leverage
        
        # [5] Drawdown from peak (0 = at peak, 1 = gave back everything)
        if self.peak_roe > 0:
            drawdown_from_peak = (self.peak_roe - current_pnl) / max(self.peak_roe, 1e-10)
        else:
            drawdown_from_peak = 0.0
        drawdown_from_peak = np.clip(drawdown_from_peak, 0.0, 5.0)
        
        # [6] ROE velocity (1st derivative: change per candle)
        if len(self.roe_history) >= 2:
            roe_velocity = self.roe_history[-1] - self.roe_history[-2]
        else:
            roe_velocity = 0.0
        roe_velocity = np.clip(roe_velocity, -2.0, 2.0)
        
        # [7] ROE acceleration (2nd derivative: velocity change)
        if len(self.roe_history) >= 3:
            vel_now = self.roe_history[-1] - self.roe_history[-2]
            vel_prev = self.roe_history[-2] - self.roe_history[-3]
            roe_acceleration = vel_now - vel_prev
        else:
            roe_acceleration = 0.0
        roe_acceleration = np.clip(roe_acceleration, -2.0, 2.0)
        
        # [8] CVD Z-score
        cvd_z_val = float(self.cvd_z_arr[idx])
        
        # [9] CVD ROC
        cvd_roc_val = float(self.cvd_roc_arr[idx])
        
        # [10] Distance to TP (1.0 = at entry, 0.0 = at TP)
        if TAKE_PROFIT_ROE > 0:
            distance_to_tp = max(0.0, (TAKE_PROFIT_ROE - current_pnl) / TAKE_PROFIT_ROE)
        else:
            distance_to_tp = 0.0
        distance_to_tp = np.clip(distance_to_tp, 0.0, 5.0)
        
        # [11] Volume ratio (current vs MA)
        vol = self.volume[idx]
        vol_ma = self.volume_ma[idx]
        volume_ratio = vol / max(vol_ma, 1e-10)
        volume_ratio = np.clip(volume_ratio, 0.0, 10.0)
        
        obs = np.array([
            current_pnl,          # [0]
            mfe,                  # [1]
            mae,                  # [2]
            time_decay,           # [3]
            atr_norm,             # [4]
            drawdown_from_peak,   # [5]
            roe_velocity,         # [6]
            roe_acceleration,     # [7]  
            cvd_z_val,            # [8]
            cvd_roc_val,          # [9]
            distance_to_tp,       # [10]
            volume_ratio,         # [11]
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        self.current_step += 1
        self.steps_taken += 1
        
        idx = min(self.current_step, self.n_candles - 1)
        current_price = self.close_prices[idx]
        high_price = self.high[idx]
        low_price = self.low[idx]
        
        # Calculate ROE using the worst intra-candle price for SL/TP checks
        current_roe = self._calc_roe(current_price)
        
        # For SL check, use the worst intra-candle price
        if self.side > 0:  # LONG
            worst_roe = self._calc_roe(low_price)
            best_roe = self._calc_roe(high_price)
        else:  # SHORT
            worst_roe = self._calc_roe(high_price)
            best_roe = self._calc_roe(low_price)
        
        # Net PnL with fees (entry + exit)
        net_pnl = current_roe - (self.fee * 2 * self.leverage)
        
        done = False
        reward = 0.0
        
        # === ACTION 1: CLOSE TRADE (agent's decision) ===
        if action == 1:
            done = True
            reward = self._compute_close_reward(net_pnl, current_roe)
            return self._get_obs(), reward, done, False, {'reason': 'AI_CLOSE'}
        
        # === MECHANICAL BRACKETS (Phase 1 Parity) ===
        
        # Hard Stop: -50% ROE
        if worst_roe <= HARD_STOP_ROE:
            done = True
            reward = HARD_STOP_ROE - (self.fee * 2 * self.leverage)  # Full SL pain
            return self._get_obs(), reward, done, False, {'reason': 'SL_HIT'}
        
        # Take Profit: +150% ROE
        if best_roe >= TAKE_PROFIT_ROE:
            done = True
            reward = TAKE_PROFIT_ROE - (self.fee * 2 * self.leverage)
            return self._get_obs(), reward, done, False, {'reason': 'TP_HIT'}
        
        # Trailing Stop (30% activation, 15% callback)
        if self.peak_roe >= TRAILING_ACTIVATION:
            trail_trigger = self.peak_roe * (1 - TRAILING_CALLBACK)
            if current_roe <= trail_trigger:
                done = True
                reward = net_pnl * 0.9  # Slightly penalize being trailed out
                return self._get_obs(), reward, done, False, {'reason': 'TRAILING_HIT'}
        
        # Time Limit: 8h (only if in profit, like live bot)
        if self.steps_taken >= self.max_steps:
            done = True
            if net_pnl > 0:
                reward = net_pnl
            else:
                reward = net_pnl * 0.5  # Less pain for time-limit losses
            return self._get_obs(), reward, done, False, {'reason': 'TIME_LIMIT'}
        
        # End of data
        if self.current_step >= self.n_candles - 1:
            done = True
            reward = net_pnl
            return self._get_obs(), reward, done, False, {'reason': 'DATA_END'}
        
        # === HOLD REWARD (Action 0) ===
        reward = self._compute_hold_reward(current_roe, net_pnl)
        
        return self._get_obs(), reward, done, False, {'reason': 'HOLD'}
    
    def _compute_close_reward(self, net_pnl, gross_roe):
        """
        Reward for AI choosing to close.
        Key: reward based on ACTUAL captured PnL, but with bonuses
        for good timing (closing before decline, closing at peaks).
        """
        reward = net_pnl  # Base: actual PnL captured
        
        # BONUS: Close during deceleration WITH profit
        if net_pnl > 0 and len(self.roe_history) >= 2:
            velocity = self.roe_history[-1] - self.roe_history[-2]
            if velocity < 0:
                # Closing during pullback while still in profit = smart
                reward += abs(velocity) * 5.0
        
        # BONUS: Saved from future loss (if near peak and decelerating)
        if self.peak_roe > 0.1:  # More than 10% ROE peak
            drawdown_pct = (self.peak_roe - gross_roe) / self.peak_roe
            if drawdown_pct > 0.05:  # Already gave back 5%+
                saved_ratio = net_pnl / max(self.peak_roe, 1e-10)
                reward += saved_ratio * 2.0  # Reward proportional to what we saved
        
        # PENALTY: Closing too early (below 10% ROE when peak was high)
        if net_pnl > 0 and net_pnl < 0.10 and self.steps_taken < 5:
            reward -= 0.1  # Small penalty for premature exits
        
        # PENALTY: Closing in loss when we could still recover
        if net_pnl < 0 and self.steps_taken < self.max_steps * 0.3:
            reward += net_pnl * 0.5  # Extra pain for panic selling early
        
        return reward
    
    def _compute_hold_reward(self, current_roe, net_pnl):
        """
        Reward for holding. Encourages holding during momentum,
        penalizes holding during decline.
        """
        reward = 0.0
        
        # Micro PnL tracking (incentive to be in the right direction)
        if len(self.roe_history) >= 2:
            velocity = self.roe_history[-1] - self.roe_history[-2]
            
            if velocity > 0:
                # Momentum positive: small reward for patience
                reward += 0.005 * (1 + current_roe)  # More reward at higher ROE
            elif velocity < 0 and current_roe > 0.05:
                # Decelerating WHILE in profit: gentle pressure to close
                reward -= 0.003 * abs(velocity) * 10
        
        # Drawdown pressure: if we gave back a lot from peak, punish holding
        if self.peak_roe > 0.1:
            dd = (self.peak_roe - current_roe) / max(self.peak_roe, 1e-10)
            if dd > 0.3:  # Gave back 30%+ of peak gains
                reward -= 0.01 * dd
        
        return reward


# === Sanity Test ===
if __name__ == "__main__":
    import sys, os
    sys.path.append(str(os.path.join(os.path.dirname(__file__), '..', '..')))
    from scripts.phantom_v30.exit_tensor_loader_v2 import load_exit_tensors_v2
    
    print("🧪 Exit Env V2 Sanity Test")
    tensors = load_exit_tensors_v2(split="train")
    
    env = MatrixExitEnvV2(tensors)
    obs, _ = env.reset()
    print(f"✅ Obs shape: {obs.shape}")  # (12,)
    print(f"   Features: {obs}")
    
    total_reward = 0
    for step in range(96):
        action = 0 if step < 90 else 1  # Hold then close
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        if done:
            print(f"   Done at step {step}: reward={total_reward:.4f}, reason={info['reason']}")
            break
    
    print("🎯 Sanity Test Passed!")
