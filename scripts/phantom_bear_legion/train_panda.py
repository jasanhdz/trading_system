#!/usr/bin/env python3
"""
THE DOJO: The Panda (Grinder Specialist) 🐼
Optimized for Low Volatility / Slow Bleed Regimes.
Dataset: dataset_bear_grinder.csv
Traits: Tight Stops, Precision, Patience.
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

# Fix path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from scripts.phantom_v9.detect_phantom_tops import calculate_phantom_dna

# Config
DATASET_PATH = Path(__file__).parent / "data/dataset_bear_grinder.csv"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_NAME = "panda_v1.pth"

# PANDA PARAMETERS (Conservative)
EPOCHS = 100
BATCH_SIZE = 64
GAMMA = 0.95 # Higher Gamma (Longer term grinding)
EPSILON_START = 1.0
EPSILON_END = 0.01 
EPSILON_DECAY = 0.99
MEMORY_SIZE = 50000 # Full Memory
BATCH_SIZE = 128 # Larger batch for GPU
LR = 0.001 # Standard Rate

# TRADING PHYSICS (Low Volatility)
SL_PCT = 0.010 # 1.0% Tight Stop
TP_PCT = 0.020 # 2.0% Modest Target
HORIZON = 96   # 8 Hours (Grinding takes time)
DRAWDOWN_PENALTY = 2.0 # Strict Penalty on Drawdown

class BearLegionNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BearLegionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256) # Boost width slightly for GPU
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) 

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, input_dim, output_dim):
        # DETECT ROCm / CUDA
        if torch.cuda.is_available():
            self.dev = torch.device("cuda")
            print(f"⚡ GPU DETECTED: {torch.cuda.get_device_name(0)}")
        else:
            self.dev = torch.device("cpu")
            print("⚠️ GPU NOT FOUND: Using CPU (Slow)")
            
        self.model = BearLegionNet(input_dim, output_dim).to(self.dev)
        self.target_model = BearLegionNet(input_dim, output_dim).to(self.dev)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        
    def act(self, state):
        if random.random() < self.epsilon: return random.randint(0, 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.dev)
        with torch.no_grad(): q = self.model(state_t)
        return torch.argmax(q).item()
        
    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))
        
    def replay(self):
        if len(self.memory) < BATCH_SIZE: return
        batch = random.sample(self.memory, BATCH_SIZE)
        s, a, r, ns, d = zip(*batch)
        s = torch.FloatTensor(np.array(s)).to(self.dev)
        a = torch.LongTensor(a).unsqueeze(1).to(self.dev)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.dev)
        ns = torch.FloatTensor(np.array(ns)).to(self.dev)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.dev)
        
        q = self.model(s).gather(1, a)
        nq = self.target_model(ns).max(1)[0].unsqueeze(1)
        tq = r + (GAMMA * nq * (1-d))
        
        loss = self.criterion(q, tq)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > EPSILON_END: self.epsilon *= EPSILON_DECAY
            
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

def main():
    print(f"🐼 THE DOJO: TRAINING PANDA (GRINDER SPECIALIST)...")
    
    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found: {DATASET_PATH}")
        return
        
    df = pd.read_csv(DATASET_PATH)
    if 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    needed = ['velocity', 'bear_trap', 'vol_z']
    if not all(c in df.columns for c in needed):
         print("🧬 Calculating DNA...")
         df = calculate_phantom_dna(df)
    df.fillna(0, inplace=True)
    
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)
    
    # train_df = train_df.iloc[-5000:].copy() # REMOVED LITE MODE
    print(f"   Bamboo Grove (Full): {len(train_df)} candles")

    # Environment (Panda Logic)
    class PandaEnv:
        def __init__(self, df_data):
            self.df = df_data.reset_index(drop=True)
            self.current_step = 0
            
        def reset(self):
            if len(self.df) < HORIZON + 50: start = 0
            else: start = random.randint(0, len(self.df) - HORIZON - 5)
            self.start_idx = start
            self.current_step = 0
            return self._get_state(start)
            
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
            idx = self.start_idx + self.current_step
            if idx >= len(self.df) - HORIZON - 1: return np.zeros(12), 0, True
            
            row = self.df.iloc[idx]
            reward = 0
            done = False
            
            if action == 1: # FIRE SHORT
                entry = row['close']
                sl_price = entry * (1 + SL_PCT)
                tp_price = entry * (1 - TP_PCT)
                
                future = self.df.iloc[idx+1 : idx+HORIZON+1]
                max_high = future['high'].max()
                min_low = future['low'].min()
                exit_close = future.iloc[-1]['close']
                
                if max_high >= sl_price:
                    # SL Hit
                    dd = (max_high - entry)/entry
                    reward = -1.0 - (dd * 100 * DRAWDOWN_PENALTY) # Severe Penalty
                elif min_low <= tp_price:
                    # TP Hit
                    reward = 1.0 # Standard Win
                else:
                    # Time Exit
                    pnl = (entry - exit_close)/entry
                    if pnl > 0: reward = 0.5 # Small win
                    else: reward = -0.5 # Small loss
                
                done = True 
            
            else: # PASS
                self.current_step += 1
                if self.current_step > 96: done = True
                reward = 0.05 # Small reward for patience
            
            next_state = self._get_state(self.start_idx + self.current_step)
            return next_state, reward, done

    env = PandaEnv(train_df)
    agent = DQNAgent(12, 2)
    best_rew = -float('inf')
    
    print(f"   Training for {EPOCHS} Epochs...")
    for e in range(EPOCHS):
        if e == 0: print("   Loop Started...")
        state = env.reset()
        total_rew = 0
        done = False
        while not done:
            a = agent.act(state)
            ns, r, d = env.step(a)
            agent.remember(state, a, r, ns, d)
            state = ns
            total_rew += r
            agent.replay()
            
        if e % 1 == 0: 
            agent.update_target()
            print(f"   Ep {e}: Reward {total_rew:.2f} | Eps {agent.epsilon:.2f}")
            
        if total_rew > best_rew:
            best_rew = total_rew
            torch.save(agent.model.state_dict(), MODEL_DIR / MODEL_NAME)
            
    print(f"✅ PANDA TRAINED. Saved to {MODEL_DIR / MODEL_NAME}")

if __name__ == "__main__":
    main()
