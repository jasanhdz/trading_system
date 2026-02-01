#!/usr/bin/env python3
"""
THE DOJO: The Grizzly (Crash Specialist) 🦖
Optimized for High Volatility / Panic Regimes.
Dataset: dataset_bear_crash.csv
Traits: Wide Stops, Aggressive Targets, Risk-Taker.
"""
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from pathlib import Path
from sklearn.model_selection import train_test_split

# Fix path to include project root (3 levels up from scripts/phantom_bear_legion/train_grizzly.py)
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import Feature Calculation if needed (for missing columns)
from scripts.phantom_v9.detect_phantom_tops import calculate_phantom_dna

# Config
DATASET_PATH = Path(__file__).parent / "data/dataset_bear_crash.csv"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_NAME = "grizzly_v1.pth"

# GRIZZLY PARAMETERS (Aggressive)
EPOCHS = 200
BATCH_SIZE = 64
GAMMA = 0.90
EPSILON_START = 1.0
EPSILON_END = 0.01 
EPSILON_DECAY = 0.99
MEMORY_SIZE = 20000
LR = 0.0005 # Slower learning rate for deep training

# TRADING PHYSICS (High Volatility)
SL_PCT = 0.030 # 3.0% Wide Stop
TP_PCT = 0.060 # 6.0% Deep Target
HORIZON = 48   # 4 Hours max for the crash to unfold
DRAWDOWN_PENALTY = 0.5 # Lenient on drawdown (Volatility is expected)

class BearLegionNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BearLegionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256) # Wider network
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) # Higher dropout to prevent overfitting small dataset

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.dev = torch.device("cpu")
        self.model = BearLegionNet(input_dim, output_dim).to(self.dev)
        self.target_model = BearLegionNet(input_dim, output_dim).to(self.dev)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self):
        if len(self.memory) < BATCH_SIZE: return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.dev)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.dev)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.dev)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.dev)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.dev)
        
        current_q = self.model(states).gather(1, actions)
        next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
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
    print(f"🦖 THE DOJO: TRAINING GRIZZLY (CRASH SPECIALIST)...")
    
    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found: {DATASET_PATH}")
        return
        
    df = pd.read_csv(DATASET_PATH)
    if 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Check Features
    needed_cols = ['velocity', 'bear_trap', 'vol_z']
    if not all(col in df.columns for col in needed_cols):
         print("🧬 Calculating DNA (Re-hydrating features)...")
         df = calculate_phantom_dna(df)
    
    df.fillna(0, inplace=True)
    
    # Train/Test Split (Time-ordered)
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)
    print(f"   Feeding Ground: {len(train_df)} candles (Validation: {len(val_df)})")
    
    # Environment (Simplified for script)
    # We define Step logic inside Training Loop or Class? Standard Class is better.
    
    class GrizzlyEnv:
        def __init__(self, df_data):
            self.df = df_data.reset_index(drop=True)
            self.current_step = 0
            
        def reset(self):
            # Start at random point, but with enough runway
            if len(self.df) < HORIZON + 50: 
                start_idx = 0
            else:
                start_idx = random.randint(0, len(self.df) - HORIZON - 5)
            
            self.start_idx = start_idx
            self.current_step = 0
            return self._get_state(start_idx)
            
        def _get_state(self, idx):
            if idx >= len(self.df): return np.zeros(12)
            row = self.df.iloc[idx]
            state = [
                row.get('velocity', 0) / row['close'] * 10000,
                row.get('acceleration', 0) / row['close'] * 10000,
                row.get('cvd_slope', 0) / 1e6,
                row.get('bear_trap', 0),
                row.get('vol_z', 0),
                row.get('volume_ratio', 1),
                row.get('dist_ema_20', 0) * 100,
                row.get('dist_ema_200', 0) * 100,
                row.get('staleness', 0) / 50.0,
                row.get('weakness_score', 0),
                row.get('is_fakeout', 0),
                0.0
            ]
            return np.array(state, dtype=np.float32)

        def step(self, action):
            # We move sequentially in the batch? No, Episode = 1 Trade Opportunity?
            # Standard DQN usually steps through time. 
            # Let's simplify: 1 Episode = 1 Random Start -> Trade -> Outcome -> Done.
            # This is "Contextual Bandits" style, perfect for "Sniper".
            # If we step through time, we learn "Wait... Wait... Fire".
            # Given we have 18k candles, let's do continuous stepping for a small window?
            # Let's stick to the "Sniper" logic: 1 Step per episode? 
            # No, train_specialist uses chunks. Let's use chunks.
            
            idx = self.start_idx + self.current_step
            if idx >= len(self.df) - HORIZON - 1:
                return np.zeros(12), 0, True
            
            row = self.df.iloc[idx]
            reward = 0
            done = False
            
            if action == 1: # FIRE SHORT
                entry = row['close']
                sl_price = entry * (1 + SL_PCT)
                tp_price = entry * (1 - TP_PCT)
                
                # Look Forward
                future = self.df.iloc[idx+1 : idx+HORIZON+1]
                max_high = future['high'].max()
                min_low = future['low'].min()
                exit_close = future.iloc[-1]['close']
                
                if max_high >= sl_price:
                    # SL Hit (Check DD Penalty)
                    dd = (max_high - entry)/entry
                    reward = -1.0 - (dd * 10 * DRAWDOWN_PENALTY) # Base -1 + DD
                elif min_low <= tp_price:
                    # TP Hit (Jackpot)
                    reward = 2.0 # TP is 6%, SL is 3%. R:R 2:1. Reward 2.0.
                else:
                    # Time Exit
                    pnl = (entry - exit_close)/entry
                    reward = pnl * 10 # Scale up pnl
                
                # Trade Ends Episode
                done = True 
            
            else: # WAIT / PASS
                # Check what we missed
                future_low = self.df.iloc[idx+1 : idx+HORIZON+1]['low'].min()
                potential_drop = (row['close'] - future_low) / row['close']
                
                if potential_drop > TP_PCT:
                    reward = -0.5 # Regret
                else:
                    reward = 0.1 # Patience rewarded
                
                # Continue if not end of chunk
                self.current_step += 1
                if self.current_step > 96: # Max 8 hours of waiting per episode
                    done = True
            
            next_state = self._get_state(self.start_idx + self.current_step)
            return next_state, reward, done

    # Init Training
    env = GrizzlyEnv(train_df)
    agent = DQNAgent(12, 2)
    
    best_rew = -float('inf')
    
    print(f"   Training for {EPOCHS} Epochs...")
    
    for e in range(EPOCHS):
        state = env.reset()
        total_rew = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, r, done = env.step(action)
            agent.remember(state, action, r, next_state, done)
            state = next_state
            total_rew += r
            agent.replay()
        
        if e % 10 == 0:
            agent.update_target()
            print(f"   Ep {e}: Reward {total_rew:.2f} | Eps {agent.epsilon:.2f}")
            
        if total_rew > best_rew:
            best_rew = total_rew
            torch.save(agent.model.state_dict(), MODEL_DIR / MODEL_NAME)
            
    print(f"✅ GRIZZLY TRAINED. Saved to {MODEL_DIR / MODEL_NAME}")

if __name__ == "__main__":
    main()
