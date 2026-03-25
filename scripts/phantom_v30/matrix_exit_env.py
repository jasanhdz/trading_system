import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

class MatrixExitEnv(gym.Env):
    """
    Phantom Exit Agent — Kamikaze V33.51
    SYNCHRONIZED with TradingService.ts live parameters.
    
    The agent spawns inside an ALREADY OPEN trade (Long or Short).
    Its only job is to decide when to CLOSE the trade based on internal health metrics.
    
    Live Bot Parity:
      - 20x Leverage
      - -50% ROE Hard Stop (-2.5% price)
      - +25% ROE Take Profit (+1.25% price)
      - 2 hour max trade duration (24 x 5min candles)
      - 0.04% Binance Futures taker fee
    """
    def __init__(self, tensors, leverage=20, max_steps=24, fee=0.0004):
        super(MatrixExitEnv, self).__init__()
        
        # Load CPU Tensors
        self.close = tensors['close'].numpy()
        self.atr = tensors['atr'].numpy()
        self.cvd_z = tensors.get('cvd_z', torch.zeros_like(tensors['close'])).numpy()
        self.cvd_roc = tensors.get('cvd_roc', torch.zeros_like(tensors['close'])).numpy()
        self.n_candles = tensors['n_candles']
        
        self.leverage = leverage
        self.max_steps = max_steps   # 24 velas x 5min = 2 horas (synced with YAML)
        self.fee = fee               # 0.04% real Binance Futures taker fee
        
        # Action: 0 = HOLD, 1 = CLOSE
        self.action_space = spaces.Discrete(2)
        
        # Obs: [current_pnl, mfe, mae, time_decay, atr_normalized, cvd_z, cvd_roc]
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(7,), dtype=np.float32)
        
        # State
        self.current_step = 0
        self.entry_step = 0
        self.entry_price = 0.0
        self.side = 1.0  # 1 for Long, -1 for Short
        self.steps_taken = 0
        self.mfe = 0.0
        self.mae = 0.0
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            
        # Spawn randomly in the timeline, leaving enough room for max_steps
        self.entry_step = np.random.randint(0, self.n_candles - self.max_steps - 1)
        self.current_step = self.entry_step
        self.entry_price = self.close[self.entry_step]
        
        # 50% chance of Long or Short
        self.side = 1.0 if np.random.rand() > 0.5 else -1.0
        
        self.steps_taken = 0
        # Initialize MFE/MAE at 0 (gross PnL, no fees baked in)
        self.mfe = 0.0
        self.mae = 0.0
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        current_price = self.close[self.current_step]
        
        # Gross leveraged PnL (no fees here — fees apply only at terminal state)
        raw_pnl_pct = ((current_price - self.entry_price) / self.entry_price) * self.side
        current_pnl = raw_pnl_pct * self.leverage
        
        # Update MFE / MAE
        if current_pnl > self.mfe:
            self.mfe = current_pnl
        if current_pnl < self.mae:
            self.mae = current_pnl
            
        time_decay = self.steps_taken / self.max_steps
        
        # Normalized Volatility (ATR % of entry price * leverage)
        current_atr = self.atr[self.current_step]
        atr_norm = (current_atr / self.entry_price) * self.leverage
        
        # CVD microstructure context
        cvd_z_val = float(self.cvd_z[self.current_step])
        cvd_roc_val = float(self.cvd_roc[self.current_step])
        
        obs = np.array([current_pnl, self.mfe, self.mae, time_decay, atr_norm, cvd_z_val, cvd_roc_val], dtype=np.float32)
        return obs

    def step(self, action):
        self.current_step += 1
        self.steps_taken += 1
        
        # Calculate current state metrics
        current_price = self.close[self.current_step]
        raw_pnl_pct = ((current_price - self.entry_price) / self.entry_price) * self.side
        # Net PnL: fee applied once at entry + once at exit = 2x fee
        current_pnl = (raw_pnl_pct * self.leverage) - (self.fee * 2 * self.leverage)
        
        done = False
        reward = 0.0
        
        # ACTION 1: CLOSE TRADE (agent's tactical decision)
        if action == 1:
            done = True
            reward = current_pnl
            return self._get_obs(), reward, done, False, {}
            
        # ACTION 0: HOLD TRADE
        # MECHANICAL BRACKETS — Synchronized with TradingService.ts Kamikaze V33.5
        # 20x Leverage: -50% ROE SL, +25% ROE TP
        HARD_STOP = -0.50    # -2.5% price movement -> -50% ROE @ 20x
        TAKE_PROFIT = 0.25   # +1.25% price movement -> +25% ROE @ 20x
        
        if current_pnl <= HARD_STOP:
            done = True
            reward = HARD_STOP  # Painful penalty for letting bracket liquidate
            
        elif current_pnl >= TAKE_PROFIT:
            done = True
            reward = TAKE_PROFIT  # Bracket captured profit, but agent didn't decide
            
        elif self.steps_taken >= self.max_steps:
            done = True
            reward = current_pnl  # Force exit on 2-hour time limit
            
        elif self.current_step >= self.n_candles - 1:
            done = True
            reward = current_pnl
            
        else:
            # PnL-aware hold reward: encourage holding winners, penalize holding losers
            if current_pnl > 0:
                reward = 0.001   # Small positive incentive to let winners run
            else:
                # Proportional penalty: deeper loss = more urgency to close
                reward = current_pnl * 0.01  # e.g., -0.05 PnL -> -0.0005 penalty
            
        return self._get_obs(), reward, done, False, {}
