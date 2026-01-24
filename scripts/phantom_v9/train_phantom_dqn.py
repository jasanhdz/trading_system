#!/usr/bin/env python3
"""
Phantom V9: DQN Training
Uses Wraith's Brain Logic with Phantom's 12-Dimension Input.
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

# Fix path to include project root
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import from the new strategy folder
from scripts.phantom_v9.detect_phantom_tops import calculate_phantom_dna, detect_eth_setups
from data.storage.database_manager import DatabaseManager

logger = logging.getLogger("phantom_dqn")

# Config
DB_URL = "sqlite:///data/binance_candles.db"
SYMBOL = "ETH/USDT"
TIMEFRAME = "5m"
EPISODES = 150
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
MEMORY_SIZE = 5000
LR = 0.001

# Trading Config (ETH es más salvaje, necesitamos stops más anchos)
SL_PCT = 0.025   # 2.5%
TP_PCT = 0.05    # 5.0%
HORIZON = 48     # 4 horas
DRAWDOWN_PENALTY = 2.0

class PhantomEnv:
    def __init__(self, df, candidates):
        self.df = df
        self.candidates = candidates
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self._get_state(0)

    def _get_state(self, step_idx):
        if step_idx >= len(self.candidates): return None
        idx = self.candidates.index[step_idx]
        row = self.candidates.iloc[step_idx]
        
        # Las 12 Columnas del ADN de Phantom (Normalizadas)
        state = [
            row['velocity'] / row['close'] * 10000,
            row['acceleration'] / row['close'] * 10000,
            row['cvd_slope'] / 1e6, # Normalizar CVD slope es difícil, usar escala fija
            row['bear_trap'],
            row['vol_z'],
            row['volume_ratio'],
            row['dist_ema_20'] * 100,
            row['dist_ema_200'] * 100,
            row['staleness'] / 50.0, # Normalizar 0-50 velas
            row['weakness_score'],
            row['is_fakeout'],
            row['reserved']
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False
        
        idx = self.candidates.index[self.current_step]
        entry_price = self.df.loc[idx, 'close']
        
        if action == 1: # SHORT
            loc = self.df.index.get_loc(idx)
            future = self.df.iloc[loc+1 : loc+HORIZON+1]
            
            if future.empty:
                reward = 0
            else:
                sl_price = entry_price * (1 + SL_PCT)
                tp_price = entry_price * (1 - TP_PCT)
                
                max_dd = 0
                hit_sl = False
                hit_tp = False
                
                for _, f in future.iterrows():
                    dd = (f['high'] - entry_price) / entry_price
                    if dd > max_dd: max_dd = dd
                    
                    if f['high'] >= sl_price:
                        reward = -SL_PCT * 100 - (max_dd * 100 * DRAWDOWN_PENALTY)
                        hit_sl = True
                        break
                    if f['low'] <= tp_price:
                        reward = TP_PCT * 100
                        hit_tp = True
                        break
                
                if not hit_sl and not hit_tp:
                    exit_p = future.iloc[-1]['close']
                    pnl = (entry_price - exit_p) / entry_price
                    reward = (pnl * 100) - (max_dd * 100 * DRAWDOWN_PENALTY)
        else: # PASS
            # Chequeo de arrepentimiento (Regret)
            loc = self.df.index.get_loc(idx)
            future = self.df.iloc[loc+1 : loc+HORIZON+1]
            if not future.empty:
                min_p = future['low'].min()
                drop = (entry_price - min_p) / entry_price
                if drop > TP_PCT:
                    reward = -10 # Castigo fuerte por no tomar un trade ganador
                else:
                    reward = 0.5 # Recompensa pequeña por ahorrar comisión

        self.current_step += 1
        next_state = self._get_state(self.current_step)
        
        if self.current_step >= len(self.candidates) - 1:
            done = True
            
        return next_state, reward, done

class PhantomNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PhantomNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128) # Capa más grande para 12 inputs
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) # Regularización

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.model = PhantomNet(input_dim, output_dim)
        self.target_model = PhantomNet(input_dim, output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        
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
        if len(self.memory) < BATCH_SIZE: return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        
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
    print("🧠 Initializing Phantom V9 Training...")
    db = DatabaseManager(DB_URL)
    df = db.get_ohlcv_data(SYMBOL, TIMEFRAME, limit=20000)
    if 'timestamp' not in df.columns: df = df.reset_index()
    
    df = calculate_phantom_dna(df)
    candidates = detect_eth_setups(df)
    
    if len(candidates) < 50:
        print("Not enough candidates. Expand filter logic.")
        return

    env = PhantomEnv(df, candidates)
    agent = DQNAgent(input_dim=12, output_dim=2)
    
    best_reward = -float('inf')
    model_path = Path("models/phantom_v9")
    model_path.mkdir(parents=True, exist_ok=True)
    
    for e in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if state is None: break
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            if next_state is None:
                next_state = np.zeros(12)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()
            
        if e % TARGET_UPDATE == 0: agent.update_target()
        
        print(f"Episode {e+1}/{EPISODES} - Reward: {total_reward:.2f} - Eps: {agent.epsilon:.2f}")
        
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.model.state_dict(), model_path / "phantom_v9_best.pth")
            
    torch.save(agent.model.state_dict(), model_path / "phantom_v9_final.pth")
    print("Phantom V9 Training Complete.")

if __name__ == "__main__":
    main()
