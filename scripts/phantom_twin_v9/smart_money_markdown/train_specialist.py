#!/usr/bin/env python3
"""
Phantom V11: The Twin Specialist (Standard Mode / Steroid Data)
Trained on "Steroid" Filtered Data (High Precision).
Goal: Learn to trade ONLY the best signals.
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

# Fix path to include project root
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Config
DATASET_PATH = ROOT_DIR / "data/dataset_steroid.csv" # UPDATED to Steroid
MODEL_NAME = "phantom_v11_twin"
LOG_DIR = ROOT_DIR / "logs/v11_twin"

# STANDARD CONFIG (Reverted from Sniper)
EPOCHS = 100 
BATCH_SIZE = 64
GAMMA = 0.90 # Lower Gamma for discontinuous episodes
EPSILON_START = 1.0
EPSILON_END = 0.01 
EPSILON_DECAY = 0.995
MEMORY_SIZE = 20000
LR = 0.001 # Standard Learning Rate

# Trading Config (Aggressive, inherited from V9)
SL_PCT = 0.035
TP_PCT = 0.06
HORIZON = 48
DRAWDOWN_PENALTY = 1.0 # Standard Penalty (Data is already filtered)

class PhantomNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PhantomNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.dev = torch.device("cpu") # Force CPU
        print(f"🔌 Device: {self.dev}")
        
        self.model = PhantomNet(input_dim, output_dim).to(self.dev)
        self.target_model = PhantomNet(input_dim, output_dim).to(self.dev)
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
    print(f"溺 Initializing Phantom V11: The Twin Specialist (Steroid Data)...")
    
    # 1. Load Dataset
    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found: {DATASET_PATH}")
        return
        
    df = pd.read_csv(DATASET_PATH)
    # Check if we need to parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. SPLIT TRAIN/TEST
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)
    
    print(f"👻 Ghost Squad (Steroid Division):")
    print(f"   Training Data: {len(train_df)} candles")
    print(f"   Validation Data: {len(val_df)} candles")
    
    # 3. Features are ALREADY calculated in refine_dataset.py? 
    # Yes, refine_dataset called calculate_phantom_dna.
    # But let's verify if columns exist.
    needed_cols = ['velocity', 'bear_trap', 'vol_z']
    if not all(col in df.columns for col in needed_cols):
         print("🧬 Calculating DNA (Missing Columns)...")
         from scripts.phantom_v9.detect_phantom_tops import calculate_phantom_dna
         train_df = calculate_phantom_dna(train_df)
    
    train_df.fillna(0, inplace=True) 
    
    # 4. Environment Wrapper (Updated for Pre-Calculated Rewards)
    class TwinEnv:
        def __init__(self, df_full, chunk_size=1000): # Smaller chunks or just sequential
            self.df_full = df_full
            self.chunk_size = chunk_size
            self.reset()
            
        def reset(self):
            if len(self.df_full) <= self.chunk_size:
                 start_idx = 0
            else:
                 start_idx = random.randint(0, len(self.df_full) - self.chunk_size)
                 
            self.current_chunk = self.df_full.iloc[start_idx : start_idx + self.chunk_size]
            self.current_step = 0
            return self._get_state(0)
            
        def _get_state(self, step_idx):
            if step_idx >= len(self.current_chunk): return None
            row = self.current_chunk.iloc[step_idx]
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
                row.get('reserved', 0)
            ]
            return np.array(state, dtype=np.float32)

        def step(self, action):
            reward = 0
            done = False
            
            if self.current_step >= len(self.current_chunk):
                 return np.zeros(12), 0, True

            row = self.current_chunk.iloc[self.current_step]
            entry_price = row['close']
            
            # --- REWARD CALCULATION (STEROID MODE) ---
            if 'future_max_high' in row:
                # Use Pre-Calculated Ground Truth (Handles discontinuity)
                max_high = row['future_max_high']
                min_low = row['future_min_low']
                close_exit = row['future_close_exit']
            else:
                # Fallback to Look-ahead (Slow / Danger of discontinuity)
                future = self.current_chunk.iloc[self.current_step+1 : self.current_step+HORIZON+1]
                if len(future) < 1:
                    max_high = entry_price
                    min_low = entry_price
                    close_exit = entry_price
                else:
                    max_high = future['high'].max()
                    min_low = future['low'].min()
                    close_exit = future.iloc[-1]['close']

            if action == 1: # SHORT
                sl_price = entry_price * (1 + SL_PCT)
                tp_price = entry_price * (1 - TP_PCT)
                
                max_dd = (max_high - entry_price) / entry_price
                if max_dd < 0: max_dd = 0
                
                if max_high >= sl_price:
                    # SL Hit
                    reward = -SL_PCT * 100 - (max_dd * 100 * DRAWDOWN_PENALTY)
                elif min_low <= tp_price:
                    # TP Hit
                    reward = TP_PCT * 100
                else:
                    # Time Exit
                    pnl = (entry_price - close_exit) / entry_price
                    reward = (pnl * 100) - (max_dd * 100 * DRAWDOWN_PENALTY)
            else: # PASS
                # Reward for passing?
                # If trade was good -> Regret (-5)
                # If trade was bad -> Relief (+0.1)
                drop = (entry_price - min_low) / entry_price
                if drop > TP_PCT:
                    reward = -5 # Regret missing big dump
                else:
                    reward = 0.1 # Good job waiting
            
            self.current_step += 1
            if self.current_step >= len(self.current_chunk) - 1:
                done = True
                next_state = np.zeros(12)
            else:
                next_state = self._get_state(self.current_step)
                
            return next_state, reward, done

    env = TwinEnv(train_df)
    agent = DQNAgent(input_dim=12, output_dim=2)
    
    best_reward = -float('inf')
    model_path = ROOT_DIR / "models" / MODEL_NAME
    model_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting Training ({EPOCHS} Episodes) on {agent.dev}...")
    
    for e in range(EPOCHS):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if state is None: break
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            if next_state is None: next_state = np.zeros(12)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()
            
        if e % 5 == 0: agent.update_target()
        
        print(f"Episode {e+1}/{EPOCHS} - Reward: {total_reward:.2f} - Eps: {agent.epsilon:.2f}")
        
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.model.state_dict(), model_path / "phantom_v11_best.pth")
            
    torch.save(agent.model.state_dict(), model_path / "phantom_v11_final.pth")
    
    print("\n✅ Training Complete (Standard Mode / Steroid Data).")

if __name__ == "__main__":
    main()
