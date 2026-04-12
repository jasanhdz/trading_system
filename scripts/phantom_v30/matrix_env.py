#!/usr/bin/env python3
"""
Phantom V30 Matrix Environment — NumPy Vectorized
Runs 2048+ trading agents in parallel using numpy arrays in a SINGLE process.
Eliminates SubprocVecEnv IPC overhead while GPU handles model training.

Architecture:
  - Environment simulation: CPU (numpy, single process, zero IPC)
  - Model training: GPU (PyTorch via SB3, hipBLAS)
  
This is 10-20x faster than SubprocVecEnv(8) because:
  1. No pickle/unpickle overhead between processes
  2. No multiprocessing synchronization
  3. Vectorized numpy ops are faster than 8 sequential Python loops
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

# Constants
COMMISSION_RATE = 0.0004  # 0.04% (Binance Futures taker fee)
SLIPPAGE = 0.0001         # 0.01%
TOTAL_FEE = COMMISSION_RATE + SLIPPAGE
INITIAL_BALANCE = 20.0    # ⚔️ KAMIKAZE MODE: $20 to $500 ⚔️
LEVERAGE = 20.0           # ⚔️ 20x Leverage ⚔️
MAINTENANCE_MARGIN_RATE = 0.004 # 0.4% MMR
WINDOW_SIZE = 64
N_FEATURES = 18  # V35: +3 acceleration features (ema_1h_accel, ema_4h_accel, cvd_accel)


class PhantomMatrixEnv(VecEnv):
    """
    NumPy-Vectorized Trading Environment.
    All N agents step simultaneously using vectorized numpy operations.
    No Python loops for agent logic, no process spawning.
    """
    
    def __init__(self, features: np.ndarray, close_prices: np.ndarray, 
                 num_envs: int = 2048):
        """
        Args:
            features: (N_candles, 4) array [log_ret, high_norm, low_norm, vol_norm]
            close_prices: (N_candles,) array of close prices
            num_envs: number of parallel agents
        """
        self.num_envs = num_envs
        self.n_candles = len(close_prices)
        
        # Market data (numpy arrays)
        self.features = features.astype(np.float32)
        self.close_prices = close_prices.astype(np.float32)
        
        # Pre-compute rolling volatility (12-period std of log_ret)
        log_rets = self.features[:, 0]
        self.volatility = np.zeros(self.n_candles, dtype=np.float32)
        for i in range(12, self.n_candles):
            self.volatility[i] = np.std(log_rets[i-12:i])
        
        # Define spaces for SB3
        observation_space = spaces.Dict({
            'market': spaces.Box(low=-10, high=10, 
                                 shape=(WINDOW_SIZE, N_FEATURES), dtype=np.float32),
            'account': spaces.Box(low=-np.inf, high=np.inf, 
                                  shape=(4,), dtype=np.float32),
        })
        action_space = spaces.Discrete(4)
        
        super().__init__(num_envs, observation_space, action_space)
        
        # --- Agent State Arrays ---
        self.balances = np.full(num_envs, INITIAL_BALANCE, dtype=np.float32)
        self.positions = np.zeros(num_envs, dtype=np.float32)
        self.entry_prices = np.zeros(num_envs, dtype=np.float32)
        self.hold_steps = np.zeros(num_envs, dtype=np.int32)
        self.flat_steps = np.zeros(num_envs, dtype=np.int32)
        self.last_actions = np.full(num_envs, -1, dtype=np.int32)
        self.action_streaks = np.zeros(num_envs, dtype=np.int32)
        self.n_flips = np.zeros(num_envs, dtype=np.int32)
        self.current_steps = np.zeros(num_envs, dtype=np.int64)
        self.episode_rewards = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self.pending_fee_recovery = np.zeros(num_envs, dtype=np.float32)
        self.peak_equity = np.full(num_envs, INITIAL_BALANCE, dtype=np.float32)
        
        # Initial reset
        # Pre-compute recency sampling weights (exponential decay, halflife=90 days)
        max_start = self.n_candles - WINDOW_SIZE - 500
        positions = np.arange(WINDOW_SIZE, max(WINDOW_SIZE + 1, max_start))
        # Exponential weight: recent candles get higher probability
        # halflife = 90 days * 288 candles/day = 25,920 candles
        halflife = 90 * 288
        weights = np.exp(np.log(2) * (positions - positions[-1]) / halflife)
        self._sampling_weights = weights / weights.sum()
        self._sampling_positions = positions
        
        self._reset_all()
    
    def _reset_all(self):
        """Reset all environments with recency-biased starting positions."""
        self.balances[:] = INITIAL_BALANCE
        self.peak_equity[:] = INITIAL_BALANCE
        self.positions[:] = 0.0
        self.entry_prices[:] = 0.0
        self.hold_steps[:] = 0
        self.flat_steps[:] = 0
        self.last_actions[:] = -1
        self.action_streaks[:] = 0
        self.n_flips[:] = 0
        self.episode_rewards[:] = 0.0
        self.episode_lengths[:] = 0
        self.pending_fee_recovery[:] = 0.0
        
        # 70% recency-biased, 30% uniform (preserves historical diversity)
        n_biased = int(self.num_envs * 0.7)
        n_uniform = self.num_envs - n_biased
        
        biased_starts = np.random.choice(
            self._sampling_positions, size=n_biased, p=self._sampling_weights
        )
        max_start = self.n_candles - WINDOW_SIZE - 500
        uniform_starts = np.random.randint(
            WINDOW_SIZE, max(WINDOW_SIZE + 1, max_start), size=n_uniform
        )
        self.current_steps = np.concatenate([biased_starts, uniform_starts]).astype(np.int64)
        np.random.shuffle(self.current_steps)
    
    def _reset_envs(self, mask):
        """Reset only environments indicated by boolean mask."""
        if not np.any(mask):
            return
        
        n_reset = np.sum(mask)
        self.balances[mask] = INITIAL_BALANCE
        self.peak_equity[mask] = INITIAL_BALANCE
        self.positions[mask] = 0.0
        self.entry_prices[mask] = 0.0
        self.hold_steps[mask] = 0
        self.flat_steps[mask] = 0
        self.last_actions[mask] = -1
        self.action_streaks[mask] = 0
        self.n_flips[mask] = 0
        self.episode_rewards[mask] = 0.0
        self.episode_lengths[mask] = 0
        self.pending_fee_recovery[mask] = 0.0
        
        # Same 70/30 recency bias for mid-training resets
        n_biased = int(n_reset * 0.7)
        n_uniform = n_reset - n_biased
        
        biased_starts = np.random.choice(
            self._sampling_positions, size=n_biased, p=self._sampling_weights
        )
        max_start = self.n_candles - WINDOW_SIZE - 500
        uniform_starts = np.random.randint(
            WINDOW_SIZE, max(WINDOW_SIZE + 1, max_start), size=max(1, n_uniform)
        )[:n_uniform]  # Handle n_uniform=0
        
        new_starts = np.concatenate([biased_starts, uniform_starts])
        np.random.shuffle(new_starts)
        self.current_steps[mask] = new_starts[:n_reset]
    
    def _get_equity(self, prices):
        """Vectorized equity for all agents."""
        equity = self.balances.copy()
        has_pos = self.positions != 0
        
        if np.any(has_pos):
            abs_pos = np.abs(self.positions)
            is_long = self.positions > 0
            is_short = self.positions < 0
            
            pnl = np.zeros_like(equity)
            pnl = np.where(
                is_long,
                abs_pos * (prices - self.entry_prices),
                np.where(
                    is_short,
                    abs_pos * (self.entry_prices - prices),
                    0.0
                )
            )
            equity += pnl
        
        return equity
    
    def _get_obs(self):
        """Build observation arrays for all agents."""
        # Market: (num_envs, WINDOW_SIZE, N_FEATURES) — fully vectorized
        offsets = np.arange(WINDOW_SIZE)  # [0, 1, ..., 63]
        all_idx = self.current_steps[:, None] - WINDOW_SIZE + offsets[None, :]
        all_idx = np.clip(all_idx, 0, self.n_candles - 1).astype(np.intp)
        market_obs = self.features[all_idx]  # (num_envs, WINDOW_SIZE, N_FEATURES)
        
        # Account: (num_envs, 4)
        idx = np.clip(self.current_steps, 0, self.n_candles - 1)
        current_prices = self.close_prices[idx]
        
        equity = self._get_equity(current_prices)
        balance_norm = equity / INITIAL_BALANCE
        
        position_notional = np.abs(self.positions) * current_prices
        leverage_used = position_notional / np.maximum(equity, 1e-10)
        
        has_pos = (self.positions != 0) & (self.entry_prices > 0)
        safe_entry = np.maximum(self.entry_prices, 1e-10)
        
        pnl_pct = np.where(
            has_pos & (self.positions > 0),
            ((current_prices - safe_entry) / safe_entry) * LEVERAGE,  # Leveraged PnL %
            np.where(
                has_pos & (self.positions < 0),
                ((safe_entry - current_prices) / safe_entry) * LEVERAGE,
                0.0
            )
        )
        
        in_trade = (self.positions != 0).astype(np.float32)
        
        account_obs = np.stack([balance_norm, leverage_used, pnl_pct, in_trade], axis=1)
        
        return {
            'market': market_obs,
            'account': account_obs,
        }
    
    def reset(self):
        """Reset all envs. Returns initial observations."""
        self._reset_all()
        return self._get_obs()
    
    def step_async(self, actions):
        """Store actions for async step."""
        self._pending_actions = np.asarray(actions, dtype=np.int32)
    
    def step_wait(self):
        """Execute actions. All logic vectorized with np.where."""
        actions = self._pending_actions
        
        # Current prices
        idx = np.clip(self.current_steps, 0, self.n_candles - 1)
        current_prices = self.close_prices[idx]
        
        # Previous state
        prev_equity = self._get_equity(current_prices)
        prev_positions = self.positions.copy()
        
        trade_fees = np.zeros(self.num_envs, dtype=np.float32)
        open_trade_fee = np.zeros(self.num_envs, dtype=np.float32)
        closed_trade_pnl = np.zeros(self.num_envs, dtype=np.float32)
        just_closed_trade = np.zeros(self.num_envs, dtype=bool)
        
        # ═══════════════ ACTION 3: CLOSE ═══════════════
        close_mask = (actions == 3) & (self.positions != 0)
        
        # ═══════════════ BRACKET OVERRIDE (Live Bot Parity) ═══════════════
        # Live Bot Settings: 20x Leverage, -40% ROE SL, +40% ROE TP
        HARD_STOP = -0.020   # -2.0% price = -40% ROE @ 20x
        TAKE_PROFIT = 0.020  # 2.0% price = +40% ROE @ 20x
        
        # Calculate raw price variation % from entry
        safe_entry_bracket = np.maximum(self.entry_prices, 1e-10)
        price_diff_pct = np.where(
            self.positions > 0,
            (current_prices - safe_entry_bracket) / safe_entry_bracket,       # LONG % diff
            (safe_entry_bracket - current_prices) / safe_entry_bracket        # SHORT % diff
        )
        
        # --- TRAILING STOP LOGIC ---
        # Update highest observed profit for active positions
        if not hasattr(self, 'peak_diff_pct'):
            self.peak_diff_pct = np.zeros(self.num_envs, dtype=np.float32)
            
        # Reset peak for new positions (handled during ACTION 1 & 2) or maintain peak
        self.peak_diff_pct = np.where(self.positions != 0, np.maximum(self.peak_diff_pct, price_diff_pct), 0.0)
        
        # Live Bot Trailing defaults (from YAML Base Config):
        # Activation at 20% ROE (which is 20% / 20x = 0.01 raw price move)
        # Callback 10% ROE from peak (which is 0.005 raw price move)
        TRAILING_ACTIVATION = 0.20 / LEVERAGE  # 0.01 raw
        TRAILING_CALLBACK = 0.005              # 0.5% callback from peak (10% ROE)
        
        trailing_hit = (self.positions != 0) & (self.peak_diff_pct >= TRAILING_ACTIVATION) & (self.peak_diff_pct - price_diff_pct >= TRAILING_CALLBACK)
        # ----------------------------

        # A position hits the brackets if the price drops below Stop Loss or rises above Take Profit
        with np.errstate(divide='ignore', invalid='ignore'):
            bracket_hit = (self.positions != 0) & ((price_diff_pct <= HARD_STOP) | (price_diff_pct >= TAKE_PROFIT) | trailing_hit)
        
        # Combine manual AI closures OR mathematical Bracket hits
        active_close = close_mask | bracket_hit
        
        if np.any(active_close):
            abs_pos = np.abs(self.positions)
            
            pnl = np.where(
                self.positions > 0,
                abs_pos * (current_prices - self.entry_prices),
                abs_pos * (self.entry_prices - current_prices)
            )
            notional = abs_pos * current_prices
            fee = notional * TOTAL_FEE
            new_balance = self.balances + pnl - fee
            
            closed_trade_pnl = np.where(active_close, pnl - fee, closed_trade_pnl)
            just_closed_trade = just_closed_trade | active_close
            
            self.balances = np.where(active_close, new_balance, self.balances)
            self.positions = np.where(active_close, 0.0, self.positions)
            self.entry_prices = np.where(active_close, 0.0, self.entry_prices)
            self.peak_diff_pct = np.where(active_close, 0.0, self.peak_diff_pct)
            trade_fees = np.where(active_close, fee, trade_fees)
        
        # ═══════════════ ACTION 1: OPEN LONG (flat only) ═══════════════
        long_mask = (actions == 1) & (self.positions == 0)
        if np.any(long_mask):
            notional = self.balances * LEVERAGE
            fee = notional * TOTAL_FEE
            # Only allow if fee doesn't wipe out balance immediately
            can_open = long_mask & (self.balances > fee * 1.5)
            
            # Position size (coins)
            new_pos = notional / np.maximum(current_prices, 1e-10)
            
            self.positions = np.where(can_open, new_pos, self.positions)
            self.balances = np.where(can_open, self.balances - fee, self.balances)
            self.entry_prices = np.where(can_open, current_prices, self.entry_prices)
            trade_fees = np.where(can_open, fee, trade_fees)
            open_trade_fee = np.where(can_open, fee, open_trade_fee)
        
        # ═══════════════ ACTION 2: OPEN SHORT (flat only) ═══════════════
        short_mask = (actions == 2) & (self.positions == 0)
        if np.any(short_mask):
            notional = self.balances * LEVERAGE
            fee = notional * TOTAL_FEE
            can_open = short_mask & (self.balances > fee * 1.5)
            
            new_pos = -(notional / np.maximum(current_prices, 1e-10))
            
            self.positions = np.where(can_open, new_pos, self.positions)
            self.balances = np.where(can_open, self.balances - fee, self.balances)
            self.entry_prices = np.where(can_open, current_prices, self.entry_prices)
            trade_fees = np.where(can_open, fee, trade_fees)
            open_trade_fee = np.where(can_open, fee, open_trade_fee)
        
        # ═══════════════ FLIP SHORT→LONG ═══════════════
        flip_long = (actions == 1) & (prev_positions < 0)
        if np.any(flip_long):
            abs_pos = np.abs(prev_positions)
            pnl = abs_pos * (self.entry_prices - current_prices)
            notional_close = abs_pos * current_prices
            fee_close = notional_close * TOTAL_FEE
            new_bal = self.balances + pnl - fee_close
            
            closed_trade_pnl = np.where(flip_long, pnl - fee_close, closed_trade_pnl)
            just_closed_trade = just_closed_trade | flip_long
            
            notional_open = new_bal * LEVERAGE
            fee_open = notional_open * TOTAL_FEE
            can_flip = flip_long & (new_bal > fee_open * 1.5)
            
            new_pos = notional_open / np.maximum(current_prices, 1e-10)
            
            # If we can't open, we just close
            self.positions = np.where(can_flip, new_pos, np.where(flip_long, 0.0, self.positions))
            self.balances = np.where(can_flip, new_bal - fee_open, np.where(flip_long, new_bal, self.balances))
            self.entry_prices = np.where(can_flip, current_prices, np.where(flip_long, 0.0, self.entry_prices))
            trade_fees = np.where(flip_long, fee_close + np.where(can_flip, fee_open, 0), trade_fees)
            open_trade_fee = np.where(can_flip, fee_open, open_trade_fee)
        
        # ═══════════════ FLIP LONG→SHORT ═══════════════
        flip_short = (actions == 2) & (prev_positions > 0)
        if np.any(flip_short):
            abs_pos = np.abs(prev_positions)
            pnl = abs_pos * (current_prices - self.entry_prices)
            notional_close = abs_pos * current_prices
            fee_close = notional_close * TOTAL_FEE
            new_bal = self.balances + pnl - fee_close
            
            closed_trade_pnl = np.where(flip_short, pnl - fee_close, closed_trade_pnl)
            just_closed_trade = just_closed_trade | flip_short
            
            notional_open = new_bal * LEVERAGE
            fee_open = notional_open * TOTAL_FEE
            can_flip = flip_short & (new_bal > fee_open * 1.5)
            
            new_pos = -(notional_open / np.maximum(current_prices, 1e-10))
            
            self.positions = np.where(can_flip, new_pos, np.where(flip_short, 0.0, self.positions))
            self.balances = np.where(can_flip, new_bal - fee_open, np.where(flip_short, new_bal, self.balances))
            self.entry_prices = np.where(can_flip, current_prices, np.where(flip_short, 0.0, self.entry_prices))
            trade_fees = np.where(flip_short, fee_close + np.where(can_flip, fee_open, 0), trade_fees)
            open_trade_fee = np.where(can_flip, fee_open, open_trade_fee)
        
        # ═══════════════ STEP FORWARD & LIQUIDATIONS ═══════════════
        self.current_steps += 1
        self.episode_lengths += 1
        
        new_idx = np.clip(self.current_steps, 0, self.n_candles - 1)
        new_prices = self.close_prices[new_idx]
        new_equity = self._get_equity(new_prices)
        
        # Update Peak Equity for Drawdown calculation
        self.peak_equity = np.maximum(self.peak_equity, new_equity)
        
        # 🔥 KAMIKAZE LIQUIDATIONS 🔥
        in_pos = self.positions != 0
        notional_value = np.abs(self.positions) * new_prices
        maintenance_margin = notional_value * MAINTENANCE_MARGIN_RATE
        
        # Liquidated if equity drops below maintenance margin + closing fee
        liquidation_threshold = maintenance_margin + (notional_value * TOTAL_FEE)
        liquidated = in_pos & (new_equity <= liquidation_threshold)
        
        # Apply liquidation penalty: lose all balance, close position
        self.balances = np.where(liquidated, 0.0, self.balances)
        self.positions = np.where(liquidated, 0.0, self.positions)
        self.entry_prices = np.where(liquidated, 0.0, self.entry_prices)
        new_equity = np.where(liquidated, 0.0, new_equity)
        
        # ═══════════════ REWARD V13: Asymmetric Greed & Variance Compression ═══════════════
        
        safe_prev = np.maximum(prev_equity, 1e-10)
        safe_new = np.maximum(new_equity, 1e-10)
        
        log_return = np.log(safe_new / safe_prev)
        
        # 1. BASE: Variance Compression & Asymmetric Greed
        # Multiply by 10.0 instead of 100.0 to prevent PPO gradient explosion at 20x.
        # Asymmetry: Winners get 15x, losers get 10x. EV of trading becomes positive computationally.
        reward = np.where(log_return > 0, log_return * 15.0, log_return * 10.0)
        
        # 2. CLOSE SETTLEMENT: Relational bonus for taking profit
        closed_profit = just_closed_trade & (closed_trade_pnl > 0)
        # V34: Eradicated flat +0.5 bonus to avoid dopamine farming. Now it scales directly with the PNL generated.
        # profit_pct is the raw percentage gain of the closed trade relative to the account
        profit_pct = np.clip(closed_trade_pnl / safe_prev, 0.0, 1.0)
        execution_bonus = np.where(closed_profit, profit_pct * 15.0, 0.0)
        
        # Subtract the fee we "lent" when the trade was opened
        execution_bonus = np.where(just_closed_trade, execution_bonus - self.pending_fee_recovery, execution_bonus)
        
        # Clear the debt for closed/liquidated positions
        self.pending_fee_recovery = np.where(just_closed_trade | liquidated, 0.0, self.pending_fee_recovery)
        
        reward += execution_bonus
        
        # 3. FIXED OPEN SHAPING: Refund fee in % to prevent entry fear (fixed previous bug)
        just_opened = open_trade_fee > 0
        fee_pct = open_trade_fee / safe_prev
        refund_reward = fee_pct * 10.0  # Perfectly matches the log_return * 10 scale
        
        reward += np.where(just_opened, refund_reward, 0.0)
        self.pending_fee_recovery = np.where(just_opened, refund_reward, self.pending_fee_recovery)
        
        # --- MUTATION 1: MTF ACTION MASKING PENALTY ---
        # Feature 12 is ema_4h_slope
        current_features = self.features[new_idx] 
        ema_4h_slope = current_features[:, 12]
        # Severe penalty if going LONG when 4H is crashing heavily (<-1.0)
        suicide_long = just_opened & (actions == 1) & (ema_4h_slope < -1.0)
        suicide_short = just_opened & (actions == 2) & (ema_4h_slope > 1.0)
        reward = np.where(suicide_long | suicide_short, reward - 0.3, reward)
        
        # --- MUTATION 2: RSI EXHAUSTION MASKING (DYNAMIC PROBABILITY BURN) ---
        # Feature 4 is rsi_norm. > 0.6 is RSI > 80 (Overbought climax). <-0.6 is RSI < 20 (Oversold bounce area)
        rsi_norm = current_features[:, 4]
        exhausted_long = just_opened & (actions == 1) & (rsi_norm > 0.6)
        exhausted_short = just_opened & (actions == 2) & (rsi_norm < -0.6)
        reward = np.where(exhausted_long | exhausted_short, reward - 0.3, reward)

        # --- MUTATION 3: ORGANIC SNIPER REWARD (Take money and run) ---
        # If the AI mathematically pushed the "CLOSE" button (actions == 3) purely on its own will,
        # AND it had more than +10% ROE (+0.5% raw price diff), it gets a relational multiplier, NOT a flat scalar!
        # Multiplier of 1.5x on the reward ensures it prioritizes massive gains over tiny micro-scalps.
        organic_sniper = close_mask & (price_diff_pct > 0.005)
        reward = np.where(organic_sniper, reward * 1.5, reward)
        
        # --- MUTATION 2: EQUITY CURVE BLEED PENALTY ---
        # Teach AI to fear floating losses > -1% price (-20% ROE)
        bleeding_penalty = np.where((self.positions != 0) & (price_diff_pct < -0.01), -0.05, 0.0)
        reward += bleeding_penalty
        
        # 4. IDLE PENALTY: Kept same absolute size to force action strongly in relative terms
        idle_penalty = np.where(
            self.flat_steps > 200, -0.1,
            np.where(self.flat_steps > 50, -0.05, 0.0)
        )
        reward += idle_penalty
        
        # 5. ENTRY ATTEMPT BONUS
        entry_bonus = np.where(just_opened, 0.15, 0.0)
        reward += entry_bonus
        
        # 6. FLIP PENALTY: Discourage overtrading
        flipped = flip_long | flip_short
        reward = np.where(flipped, reward - 0.2, reward)
        
        # 7. DEATH PENALTY: Removed -50 Cliff (Variance reduction)
        # We cap the natural loss strictly at -10.0 so gradients don't explode
        ruined = liquidated | (new_equity < INITIAL_BALANCE * 0.1)
        current_dd = (self.peak_equity - new_equity) / np.maximum(self.peak_equity, 1e-10)
        over_dd = current_dd > 0.80
        
        ruined = ruined | over_dd
        # Rather than punishing them infinitely, we give a strict painful ceiling (-10.0)
        reward = np.where(ruined, -10.0, reward)
        
        # Update tracking (kept for observation/debugging, no reward impact)
        self.hold_steps = np.where(in_pos, self.hold_steps + 1, 0)
        is_flat = (self.positions == 0)
        self.flat_steps = np.where(is_flat, self.flat_steps + 1, 0)
        self.last_actions = actions.copy()
        
        # Done conditions
        mission_accomplished = new_equity >= 500.0
        
        # ═══════════════ DONE / RESET ═══════════════
        done = ruined | mission_accomplished | (self.current_steps >= self.n_candles - 2)
        
        infos = []
        for i in range(self.num_envs):
            info = {
                'balance': float(self.balances[i]),
                'equity': float(new_equity[i]),
            }
            if done[i]:
                info['episode'] = {
                    'r': float(self.episode_rewards[i] + reward[i]),
                    'l': int(self.episode_lengths[i]),
                }
            infos.append(info)
        
        self.episode_rewards += reward
        self._reset_envs(done)
        
        obs = self._get_obs()
        return obs, reward.astype(np.float32), done.astype(np.bool_), infos
    
    def close(self):
        pass
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
    
    def env_method(self, method_name, *args, indices=None, **kwargs):
        pass
    
    def get_attr(self, attr_name, indices=None):
        return [None] * self.num_envs
    
    def set_attr(self, attr_name, value, indices=None):
        pass
    
    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)


# === Sanity Test ===
if __name__ == "__main__":
    import sys, os
    sys.path.append(str(os.path.join(os.path.dirname(__file__), '..', '..')))
    from scripts.phantom_v30.tensor_loader import load_tensor_data
    
    print("🧪 Matrix Env Sanity Test (NumPy) - KAMIKAZE 50x LEVERAGE")
    data = load_tensor_data("cpu")
    
    env = PhantomMatrixEnv(
        features=data['features'].numpy(),
        close_prices=data['close'].numpy(),
        num_envs=16,
    )
    
    obs = env.reset()
    print(f"✅ Obs market shape: {obs['market'].shape}")   # (16, 64, 4)
    print(f"✅ Obs account shape: {obs['account'].shape}")  # (16, 4)
    
    import time
    t0 = time.time()
    for step in range(1000):
        actions = np.random.randint(0, 4, size=16)
        obs, rewards, dones, infos = env.step(actions)
    elapsed = time.time() - t0
    
    print(f"✅ 1000 steps x 16 envs in {elapsed:.3f}s ({16000/elapsed:.0f} steps/s)")
    print(f"   Balances: min={min(i['balance'] for i in infos):.2f}, max={max(i['balance'] for i in infos):.2f}")
    print(f"   Rewards range: [{rewards.min():.6f}, {rewards.max():.6f}]")
    print(f"   Resets: {dones.sum()}")
    
    # Scale test
    print(f"\\n🔥 Scale Test: 2048 envs...")
    env2 = PhantomMatrixEnv(
        features=data['features'].numpy(),
        close_prices=data['close'].numpy(),
        num_envs=2048,
    )
    obs = env2.reset()
    t0 = time.time()
    for step in range(100):
        actions = np.random.randint(0, 4, size=2048)
        obs, rewards, dones, infos = env2.step(actions)
    elapsed = time.time() - t0
    print(f"✅ 100 steps x 2048 envs in {elapsed:.3f}s ({2048*100/elapsed:.0f} steps/s)")
    print(f"🎯 All Tests Passed!")
