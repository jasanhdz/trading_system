import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict

# Constants
COMMISSION_RATE = 0.0002 # 0.02% (Sniper Mode: Quality > Quantity)
SLIPPAGE = 0.0001 # 0.01% (Realistic friction)
INITIAL_BALANCE = 1000.0
LEVERAGE = 1.0 # Base leverage, can be scaled by action if needed, but keeping simple

class PhantomEnv(gym.Env):
    """
    Phantom V30 Trading Environment
    Action Space: Discrete(4)
        0: IDLE/HOLD (Do nothing)
        1: LONG (Open Long or Flip to Long)
        2: SHORT (Open Short or Flip to Short)
        3: CLOSE (Close any position)
    
    Observation Space:
        Dict:
            'market': Box(window_size, n_features) -> Price, Vol, CVD, Funding
            'account': Box(4,) -> [Balance, Position_Size, Avg_Entry, Unrealized_PnL]
    """
    
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, window_size: int = 64, render_mode: str = None):
        super(PhantomEnv, self).__init__()
        
        # Data Copy to avoid SettingWithCopy warnings
        self.df = df.copy()
        self.window_size = window_size
        self.render_mode = render_mode
        
        # --- Feature Engineering (On-The-Fly Normalization) ---
        # 1. Log Returns (Close-to-Close)
        self.df['log_ret'] = np.log(self.df['close'] / self.df['close'].shift(1)).fillna(0)
        
        # 2. High/Low relative to Close (Shadows)
        # Log(High / Close) -> Positive value indicating upside volatility
        self.df['high_norm'] = np.log(self.df['high'] / self.df['close']).fillna(0)
        # Log(Low / Close) -> Negative value indicating downside volatility
        self.df['low_norm'] = np.log(self.df['low'] / self.df['close']).fillna(0)
        
        # 3. Volume Normalization (Relative to 24-period MA)
        # Add 1e-8 to avoid div by zero
        vol_ma = self.df['volume'].rolling(window=24).mean()
        self.df['vol_norm'] = (self.df['volume'] / (vol_ma + 1e-8)).fillna(0) 
        # Clip huge volume spikes to 10x mean to prevent outliers
        self.df['vol_norm'] = self.df['vol_norm'].clip(0, 10)

        # Select ONLY Normalized Features
        self.market_features = ['log_ret', 'high_norm', 'low_norm', 'vol_norm']
        self.n_features = len(self.market_features)
        
        # --- Spaces ---
        self.action_space = spaces.Discrete(4) 
        
        self.observation_space = spaces.Dict({
            'market': spaces.Box(low=-10, high=10, shape=(window_size, self.n_features), dtype=np.float32),
            # Account: [Balance%, Leverage, PnL%, 0]
            'account': spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        })
        
        # --- State ---
        self.current_step = 0
        self.balance = INITIAL_BALANCE
        self.position = 0.0 # Asset Units
        self.entry_price = 0.0
        self.max_steps = len(self.df) - window_size - 1
        
        self.history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start at window_size to have enough history
        self.current_step = self.window_size
        self.balance = INITIAL_BALANCE
        self.position = 0.0
        self.entry_price = 0.0
        self.hold_steps = 0
        self.last_action = -1
        self.action_streak = 0
        self.history = []
        
        return self._get_obs(), {}

    def _get_obs(self):
        # 1. Market Obs (Normalized)
        end = self.current_step
        start = end - self.window_size
        market_obs = self.df.iloc[start:end][self.market_features].values.astype(np.float32)
        
        # 2. Account Obs (Normalized)
        current_price = self.df.iloc[self.current_step]['close']
        
        # A. Balance Ratio (Relative to Initial)
        balance_norm = self.balance / INITIAL_BALANCE
        
        # B. Leverage / Exposure
        # Value = Units * Price. Leverage = Value / Equity.
        # Simple Proxy: Units * Price / Initial Balance (keeps scale consistent)
        position_value = self.position * current_price
        leverage = position_value / INITIAL_BALANCE
        
        # C. Unrealized PnL % (Log Return distance from Entry)
        pnl_pct = 0.0
        if self.position != 0 and self.entry_price > 0:
            # If Long: log(Current/Entry). If Short: log(Entry/Current) -> -log(Current/Entry)
            if self.position > 0:
                pnl_pct = np.log(current_price / self.entry_price)
            else:
                pnl_pct = np.log(self.entry_price / current_price)
                
        # D. Time progress? Or Hold duration? 
        # For now, 0.0 placeholder or maybe 'In Trade' duration.
        # Let's use 'In Trade' flag (1.0 or 0.0)
        in_trade = 1.0 if self.position != 0 else 0.0
        
        account_obs = np.array([
            balance_norm, # ~1.0
            leverage,     # ~ -1.0 to 1.0
            pnl_pct,      # ~ -0.05 to 0.05
            in_trade      # 0 or 1
        ], dtype=np.float32)
        
        return {
            'market': market_obs,
            'account': account_obs
        }

    def step(self, action):
        done = False
        truncated = False
        current_price = self.df.iloc[self.current_step]['close']
        prev_equity = self._get_equity(current_price)
        
        reward = 0.0
        trade_fee = 0.0
        prev_position = self.position  # Track for Short bonus in reward
        
        # Execute Action
        # 0: IDLE, 1: LONG, 2: SHORT, 3: CLOSE
        
        if action == 1: # OPEN LONG
            if self.position == 0:
                # Open Long
                size = self.balance / current_price # All-in 1x for simplicity or fixed size? 
                # Let's use fixed fractional sizing or full equity?
                # For RL, full equity usage simplifies "direction" learning.
                cost = self.balance * (COMMISSION_RATE + SLIPPAGE)
                self.position = (self.balance - cost) / current_price
                self.balance -= cost # Deduct fee from cash (simplified)
                self.entry_price = current_price
                trade_fee = cost
            elif self.position < 0:
                # Flip Short to Long
                # Close Short
                value = abs(self.position) * current_price
                pnl = (self.entry_price - current_price) / self.entry_price * value
                fee_close = value * (COMMISSION_RATE + SLIPPAGE)
                self.balance = self.balance + pnl - fee_close
                self.position = 0
                
                # Open Long
                cost = self.balance * (COMMISSION_RATE + SLIPPAGE)
                self.position = (self.balance - cost) / current_price
                self.balance -= cost
                self.entry_price = current_price
                trade_fee = fee_close + cost

        elif action == 2: # OPEN SHORT
             if self.position == 0:
                # Open Short
                cost = self.balance * (COMMISSION_RATE + SLIPPAGE)
                self.position = -((self.balance - cost) / current_price) # Negative size
                self.balance -= cost
                self.entry_price = current_price
                trade_fee = cost
             elif self.position > 0:
                # Flip Long to Short
                # Close Long
                value = self.position * current_price
                pnl = (current_price - self.entry_price) / self.entry_price * value
                fee_close = value * (COMMISSION_RATE + SLIPPAGE)
                self.balance = self.balance + pnl - fee_close
                self.position = 0
                
                # Open Short
                cost = self.balance * (COMMISSION_RATE + SLIPPAGE)
                self.position = -((self.balance - cost) / current_price)
                self.balance -= cost
                self.entry_price = current_price
                trade_fee = fee_close + cost
                
        elif action == 3: # CLOSE
            if self.position != 0:
                value = abs(self.position) * current_price
                pnl = 0
                if self.position > 0:
                     pnl = (current_price - self.entry_price) / self.entry_price * value
                else:
                     pnl = (self.entry_price - current_price) / self.entry_price * value
                
                fee_close = value * (COMMISSION_RATE + SLIPPAGE)
                self.balance = self.balance + pnl - fee_close
                self.position = 0
                self.entry_price = 0
                trade_fee = fee_close
        
        # Step Forward
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
            
        # Calculate Reward
        new_price = self.df.iloc[self.current_step]['close']
        new_equity = self._get_equity(new_price)
        
        # === REWARD V4: Survival Gene (Short Discovery + Volatility Hunger) ===
        # Base: Log Return of Equity
        reward = np.log(new_equity / prev_equity)
        
        # --- Current volatility (for context-aware rewards) ---
        recent_returns = self.df['log_ret'].iloc[max(0, self.current_step-12):self.current_step]
        volatility = recent_returns.std() if len(recent_returns) > 1 else 0.0
        
        # 1. REALIZED PnL BONUS (Short gets 1.5x multiplier!)
        if action == 3 and trade_fee > 0:  # Actually closed a position
            if new_equity > prev_equity:
                base_bonus = 0.01
                # SHORT BONUS: 1.5x reward for profitable Short closes
                if prev_position < 0:  # Was Short
                    base_bonus *= 1.5
                reward += base_bonus
            else:
                reward -= 0.005  # Penalty for closing at a loss
        
        # 2. HOLD PENALTY + TIME DECAY
        if self.position != 0:
            self.hold_steps += 1
            # Time decay: every step in a position costs a tiny amount
            reward -= 0.0001  # "Time is money" — forces quick decisive trades
            # Escalating penalty after 100 steps
            if self.hold_steps > 100:
                reward -= 0.001 * (self.hold_steps - 100) / 50
        else:
            self.hold_steps = 0
        
        # 3. VOLATILITY-IDLE PENALTY: Punish sitting still during big moves
        if action == 0 and self.position == 0 and volatility > 0.003:
            reward -= 0.002 * (volatility / 0.003)  # Scales with how volatile it is
        
        # 4. TRADE ENTRY: Bonus for opening, extra for Shorts
        if action in (1, 2) and self.position != 0 and trade_fee > 0:
            entry_bonus = 0.001
            if action == 2:  # SHORT entry gets 1.5x bonus
                entry_bonus *= 1.5
            reward += entry_bonus
        
        # 5. ANTI-MONOTONY (from V3)
        if action == self.last_action:
            self.action_streak += 1
        else:
            self.action_streak = 0
            reward += 0.0005
        self.last_action = action
        
        if self.action_streak > 20:
            reward -= 0.001 * (self.action_streak - 20) / 10
        
        # Info
        info = {
            'equity': new_equity,
            'balance': self.balance,
            'position': self.position,
            'step_fee': trade_fee
        }
        
        # Early Stopping if Ruin
        if new_equity < INITIAL_BALANCE * 0.1: # 90% Drawdown
            done = True
            reward -= 1.0 # Heavy Penalty for ruin
            
        return self._get_obs(), reward, done, truncated, info

    def _get_equity(self, price):
        equity = self.balance
        if self.position != 0:
            value = abs(self.position) * price
            pnl = 0
            if self.position > 0:
                pnl = (price - self.entry_price) / self.entry_price * value
            else:
                pnl = (self.entry_price - price) / self.entry_price * value
            
            # Note: Balance already had entry fee deducted. 
            # Equity is Balance (which is effectively margin) + PnL.
            # But wait, if I deducted fee from balance at entry, balance is 'Cash remaining'.
            # Equity = Cash + Position Value - Debt?
            # Simplified:
            # Entry: Balance 1000. Buy 1 ETH @ 1000. Fee 1.
            # Balance = 0. Position = 1 ETH.
            # Equity = 0 + 1 * 1000 = 1000. (Minus fee? I deducted fee from balance BEFORE buying size).
            # So I bought 0.999 ETH.
            # Equity = 0 + 0.999 * 1000 = 999.
            # Correct.
            
            # What about closing fee? Equity should estimate closing fee?
            # Standard Gym envs usually don't penalize exit fee until exit.
            # But "Mark to Market" equity is usually Position Value.
            equity += pnl
        
        return equity

    def render(self):
        pass
