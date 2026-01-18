#!/usr/bin/env python3
"""
Project Wraith: Deep Q-Learning Training (The Sniper)
Target: Train an agent to filter 'Distribution Top' candidates.
Reward: PnL - Drawdown Penalty.
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from pathlib import Path
import logging

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from data.storage.database_manager import DatabaseManager
from utils.logger import setup_logger
# Import detection logic
from scripts.detect_distribution_tops import calculate_physics_features, detect_wraith_setups

logger = setup_logger("wraith_dqn")

# Config
DB_URL = "sqlite:///data/binance_candles.db"
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
EPISODES = 150 # Optimized from 500
BATCH_SIZE = 32
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
MEMORY_SIZE = 2000
LR = 0.001

# Trading Config
TP_PCT = 0.04  # 4% Target (Cascades)
SL_PCT = 0.01  # 1% Stop (Tight)
HORIZON = 48   # 4 Hours (5m * 48)
DRAWDOWN_PENALTY_FACTOR = 2.0 # Penalize DD 2x more than PnL gain

class WraithEnvironment:
    def __init__(self, df, candidates):
        self.df = df
        self.candidates = candidates
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        return self._get_state(self.current_step)
        
    def _get_state(self, step_idx):
        if step_idx >= len(self.candidates):
            return None
            
        idx = self.candidates.index[step_idx]
        row = self.candidates.iloc[step_idx]
        
        # State Vector (Normalized roughly)
        state = [
            row['dist_to_ema'] * 100,      # % Distance
            row['velocity_sm'] / row['close'] * 1000, # Norm Velocity
            row['acceleration_sm'] / row['close'] * 1000, # Norm Accel
            row['volatility_z'],           # Z-Score
            row['bb_dist'] * 100,          # % Dist to BB
            (row['volume'] / (row['vol_sm'] + 1e-8)) - 1.0 # Vol Ratio
        ]
        return np.array(state, dtype=np.float32)
        
    def step(self, action):
        """
        Action 0: PASS
        Action 1: SHORT
        """
        done = False
        reward = 0
        
        idx = self.candidates.index[self.current_step]
        entry_price = self.df.loc[idx, 'close']
        
        # Calculate Outcome
        if action == 1: # SHORT
            # Simulate Future
            # Get next HORIZON candles
            # We need integer location
            loc = self.df.index.get_loc(idx)
            future = self.df.iloc[loc+1 : loc+HORIZON+1]
            
            if future.empty:
                reward = 0
            else:
                # Check SL/TP
                sl_price = entry_price * (1 + SL_PCT) # Short SL is above
                tp_price = entry_price * (1 - TP_PCT) # Short TP is below
                
                outcome_pnl = 0
                max_dd = 0
                
                hit_sl = False
                hit_tp = False
                
                for _, f_row in future.iterrows():
                    # Calculate Drawdown (Price moving UP against Short)
                    dd = (f_row['high'] - entry_price) / entry_price
                    if dd > max_dd:
                        max_dd = dd
                        
                    if f_row['high'] >= sl_price:
                        outcome_pnl = -SL_PCT
                        hit_sl = True
                        break
                    
                    if f_row['low'] <= tp_price:
                        outcome_pnl = TP_PCT
                        hit_tp = True
                        break
                
                if not hit_sl and not hit_tp:
                    # Exit at end of horizon
                    exit_price = future.iloc[-1]['close']
                    outcome_pnl = (entry_price - exit_price) / entry_price
                
                # Reward Function: PnL - (DD * Penalty)
                # If SL hit, max_dd is likely SL_PCT.
                # If TP hit, max_dd might be small.
                
                # We want to encourage "Clean Drops" (Low DD)
                reward = (outcome_pnl * 100) - (max_dd * 100 * DRAWDOWN_PENALTY_FACTOR)
                
        else: # PASS
            # Check if we missed a big drop
            # Calculate potential PnL
            loc = self.df.index.get_loc(idx)
            future = self.df.iloc[loc+1 : loc+HORIZON+1]
            if not future.empty:
                min_price = future['low'].min()
                max_drop = (entry_price - min_price) / entry_price
                
                if max_drop > 0.03: # Missed a 3% drop
                    reward = -0.5 # Small regret penalty
                else:
                    reward = 0.1 # Good pass (avoided noise)
        
        self.current_step += 1
        next_state = self._get_state(self.current_step)
        
        if self.current_step >= len(self.candidates) - 1:
            done = True
            
        return next_state, reward, done

class WraithNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WraithNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.model = WraithNet(input_dim, output_dim)
        self.target_model = WraithNet(input_dim, output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.input_dim = input_dim
        
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
            
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states)) # Handle None?
        # Filter out None next_states (terminal)
        
        # Simple handling: if done, target is reward.
        # We need to mask next_states.
        
        # Compute Q(s, a)
        current_q = self.model(states).gather(1, actions)
        
        # Compute Q(s', a') from target
        # Handle terminal states manually
        next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
        
        # If done, next_q should be 0
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        target_q = rewards + (GAMMA * next_q * (1 - dones_t))
        
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
            
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

def main():
    logger.info("🦅 Initializing Wraith DQN Training...")
    
    # 1. Load Data
    db_manager = DatabaseManager(DB_URL)
    df = db_manager.get_ohlcv_data(SYMBOL, TIMEFRAME, limit=5000) # More data for training
    
    # Ensure timestamp column
    if 'timestamp' not in df.columns:
        df = df.reset_index()
        
    # 2. Detect Candidates (The Environment)
    logger.info("Detecting candidates...")
    df = calculate_physics_features(df)
    candidates = detect_wraith_setups(df)
    
    logger.info(f"Found {len(candidates)} candidates for training.")
    if len(candidates) < 50:
        logger.error("Not enough candidates to train.")
        return
        
    # 3. Initialize Environment & Agent
    env = WraithEnvironment(df, candidates)
    agent = DQNAgent(input_dim=6, output_dim=2)
    
    # 4. Training Loop
    logger.info(f"Starting training for {EPISODES} episodes...")
    
    best_reward = -float('inf')
    model_path = Path("models/wraith_dqn")
    model_path.mkdir(parents=True, exist_ok=True)
    
    for e in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if state is None: break
            
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            # If next_state is None (end of list), treat as done
            if next_state is None:
                done = True
                # Use dummy state for replay buffer shape consistency or handle in replay
                # For simplicity, we just don't add if next_state is None, or add zero vector
                next_state = np.zeros(agent.input_dim) 
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.replay()
            
        if e % TARGET_UPDATE == 0:
            agent.update_target()
            
        logger.info(f"Episode {e+1}/{EPISODES} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")
        
        # Save Best Model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.model.state_dict(), model_path / "wraith_net_best.pth")
            logger.info(f"⭐ New Best Reward: {best_reward:.2f} (Saved)")
            
        # Periodic Save
        if (e+1) % 10 == 0:
            torch.save(agent.model.state_dict(), model_path / "wraith_net.pth")
            
    # 5. Save Final Model
    torch.save(agent.model.state_dict(), model_path / "wraith_net.pth")
    logger.info(f"Wraith Model saved to {model_path}")

if __name__ == "__main__":
    main()
