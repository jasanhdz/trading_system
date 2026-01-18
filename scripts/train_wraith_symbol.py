#!/usr/bin/env python3
"""
Project Wraith: Symbol-Specific Training Pipeline
Target: Train independent DQN models for each symbol.
Usage: python train_wraith_symbol.py ETH/USDT
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
import os
from pathlib import Path
from collections import deque

sys.path.append(str(Path(__file__).parent.parent))

from data.storage.database_manager import DatabaseManager
from scripts.detect_distribution_tops import calculate_physics_features, detect_wraith_setups

# Config
DB_URL = "sqlite:///data/binance_candles.db"
TIMEFRAME = "5m"

# Training Config
EPISODES = 150
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LR = 0.001
MEMORY_SIZE = 10000
TARGET_UPDATE = 10

# Simulation
HORIZON = 48  # 4 hours
SL_PCT = 0.01
TP_PCT = 0.04
LEVERAGE = 10

# AMD ROCm GPU Detection
def get_device():
    if torch.cuda.is_available():  # In ROCm, torch.cuda works for HIP
        device = torch.device("cuda")
        print(f"🔥 Using AMD GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU for training")
    return device

device = get_device()

class WraithNet(nn.Module):
    def __init__(self, input_dim=6, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

class SymbolEnvironment:
    def __init__(self, symbol, candidates, df):
        self.symbol = symbol
        self.candidates = candidates
        self.df = df
        self.current_idx = 0
        
    def reset(self):
        self.current_idx = 0
        return self._get_state()
        
    def _get_state(self):
        if self.current_idx >= len(self.candidates):
            return None
        row = self.candidates.iloc[self.current_idx]
        return np.array([
            row['dist_to_ema'] * 100,
            row['velocity_sm'] / row['close'] * 1000,
            row['acceleration_sm'] / row['close'] * 1000,
            row['volatility_z'],
            row['bb_dist'] * 100,
            (row['volume'] / (row['vol_sm'] + 1e-8)) - 1.0
        ], dtype=np.float32)
        
    def step(self, action):
        if self.current_idx >= len(self.candidates):
            return None, 0, True
            
        row = self.candidates.iloc[self.current_idx]
        cand_idx = self.candidates.index[self.current_idx]
        
        if action == 0:  # PASS
            reward = 0
        else:  # SHOOT
            entry_price = row['close']
            sl_price = entry_price * (1 + SL_PCT)
            tp_price = entry_price * (1 - TP_PCT)
            
            loc = self.df.index.get_loc(cand_idx)
            future = self.df.iloc[loc+1 : loc+HORIZON+1]
            
            exit_price = entry_price
            max_dd = 0
            
            for _, fut in future.iterrows():
                if fut['high'] >= sl_price:
                    exit_price = sl_price
                    break
                if fut['low'] <= tp_price:
                    exit_price = tp_price
                    break
                dd = (fut['high'] - entry_price) / entry_price
                if dd > max_dd:
                    max_dd = dd
                exit_price = fut['close']
            
            pnl = (entry_price - exit_price) / entry_price * LEVERAGE
            reward = pnl - (2 * max_dd * LEVERAGE)
        
        self.current_idx += 1
        next_state = self._get_state()
        done = next_state is None
        
        return next_state, reward, done

class DQNAgent:
    def __init__(self):
        self.policy_net = WraithNet().to(device)
        self.target_net = WraithNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.policy_net(state_t).argmax().item()
            
    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return 0
            
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        
        # Handle None next_states
        non_final_mask = torch.tensor([s is not None for s in next_states], device=device)
        non_final_next = torch.FloatTensor(np.array([s for s in next_states if s is not None])).to(device)
        
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1))
        
        next_q = torch.zeros(BATCH_SIZE, device=device)
        if len(non_final_next) > 0:
            next_q[non_final_mask] = self.target_net(non_final_next).max(1)[0].detach()
        
        expected_q = rewards_t + GAMMA * next_q
        
        loss = nn.MSELoss()(q_values.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

def train_for_symbol(symbol):
    print(f"\n{'='*60}")
    print(f"🦅 TRAINING WRAITH FOR: {symbol}")
    print(f"{'='*60}")
    
    # Create model directory
    symbol_clean = symbol.replace('/', '_').replace(':', '_')
    model_dir = f"models/wraith_{symbol_clean.lower()}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Load Data
    print(f"📊 Loading data...")
    db_manager = DatabaseManager(DB_URL)
    df = db_manager.get_ohlcv_data(symbol, TIMEFRAME, limit=50000)
    
    if df.empty or len(df) < 1000:
        print(f"❌ Insufficient data for {symbol}: {len(df)} candles")
        return None
    
    if 'timestamp' not in df.columns:
        df = df.reset_index()
    
    print(f"📊 Loaded {len(df)} candles")
    
    # Physics Features
    print(f"🔬 Calculating physics features...")
    df = calculate_physics_features(df)
    candidates = detect_wraith_setups(df)
    print(f"🎯 Found {len(candidates)} candidates")
    
    if len(candidates) < 100:
        print(f"❌ Not enough candidates for training: {len(candidates)}")
        return None
    
    # Train DQN
    print(f"🧠 Training DQN...")
    agent = DQNAgent()
    env = SymbolEnvironment(symbol, candidates, df)
    
    best_reward = -float('inf')
    rewards_history = []
    
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        
        while state is not None:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step()
            
            total_reward += reward
            state = next_state
            
        agent.decay_epsilon()
        
        if episode % TARGET_UPDATE == 0:
            agent.update_target()
            
        rewards_history.append(total_reward)
        
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.policy_net.state_dict(), f"{model_dir}/wraith_net_best.pth")
            
        if episode % 10 == 0:
            avg = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else total_reward
            print(f"  Episode {episode}: Reward={total_reward:.2f} | Best={best_reward:.2f} | ε={agent.epsilon:.3f}")
    
    # Save final model
    torch.save(agent.policy_net.state_dict(), f"{model_dir}/wraith_net.pth")
    
    print(f"\n✅ Training complete for {symbol}")
    print(f"📁 Model saved to: {model_dir}/")
    print(f"🏆 Best Reward: {best_reward:.2f}")
    
    return model_dir

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_wraith_symbol.py SYMBOL")
        print("Example: python train_wraith_symbol.py ETH/USDT")
        print("         python train_wraith_symbol.py SOL/USDT")
        sys.exit(1)
    
    symbol = sys.argv[1]
    train_for_symbol(symbol)

if __name__ == "__main__":
    main()
