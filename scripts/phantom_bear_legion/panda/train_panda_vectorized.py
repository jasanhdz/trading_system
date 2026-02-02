#!/usr/bin/env python3
"""
PHANTOM LEGION: Panda V1 (The Speedster)
Optimized Training using NUMPY VECTORIZATION.
Replaces slow DataFrame.iloc[] loops with pure array math.
Dataset: dataset_bear_grinder.csv
"""
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Fix path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Imports
from scripts.phantom_v9.detect_phantom_tops import calculate_phantom_dna

# Config
DATASET_PATH = Path(__file__).parent.parent / "data/dataset_bear_grinder.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_NAME = "panda_v1_vectorized.pth"

# Parameters
EPOCHS = 100
BATCH_SIZE = 64
LR = 0.0005
LEVERAGE = 20.0
SL_PCT = 0.010 # Panda Tight SL (1%)
TP_PCT = 0.020 # Panda Tight TP (2%)
HORIZON = 96   # 8 Hours
FIXED_MARGIN = 20.0
WEAKNESS_PENALTY = 0.5 

class PandaNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PandaNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

class VectorizedEnv:
    """
    The High-Speed Environment.
    Uses pre-calculated Numpy arrays instead of DataFrame slicing.
    """
    def __init__(self, close_arr, high_arr, low_arr, open_arr, feature_matrix):
        self.close_arr = close_arr
        self.high_arr = high_arr
        self.low_arr = low_arr
        self.open_arr = open_arr
        self.features = feature_matrix # Shape: (N, 12)
        
        self.n_samples = len(close_arr)
        self.current_idx = 0
        
    def reset(self):
        """Start at a random index"""
        if self.n_samples <= HORIZON + 50:
            self.current_idx = 0
        else:
            # Random start point within valid range
            self.current_idx = np.random.randint(0, self.n_samples - HORIZON - 50)
            
        return self.get_state(self.current_idx)

    def get_state(self, idx):
        """
        Retrieves state vector using numpy indexing (FAST).
        """
        if idx >= self.n_samples:
            return np.zeros(12, dtype=np.float32)
            
        return self.features[idx]

    def step(self, action):
        """
        Executes a trade using vectorized lookahead (SUPER FAST).
        """
        idx = self.current_idx
        
        # Lookahead vectorization
        # Future windows: [idx+1 : idx+HORIZON+1]
        future_close = self.close_arr[idx+1 : idx+HORIZON+1]
        future_high = self.high_arr[idx+1 : idx+HORIZON+1]
        future_low = self.low_arr[idx+1 : idx+HORIZON+1]
        
        # Logic
        entry = self.close_arr[idx]
        sl = entry * (1 + SL_PCT)
        tp = entry * (1 - TP_PCT)
        
        reward = 0.0
        done = False
        
        if action == 1: # FIRE SHORT
            # Check SL
            if np.any(future_high >= sl):
                reward = -SL_PCT * 100 - (WEAKNESS_PENALTY * 10) # Penalty
                done = True
                
            # Check TP
            elif np.any(future_low <= tp):
                reward = TP_PCT * 100
                done = True
                
            else:
                # Time Out
                exit_price = future_close[-1]
                pnl = (entry - exit_price) / entry
                reward = (pnl * 100) - (WEAKNESS_PENALTY * 2) # Penalty for holding
        else:
            # PASS
            # Regret check: How much did we miss?
            min_low = np.min(future_low)
            max_drop = (entry - min_low) / entry
            
            if max_drop > TP_PCT:
                reward = -2.0 # Missed opportunity
            else:
                reward = 0.1
        
        # Move pointer
        self.current_idx += 1
        if self.current_idx >= self.n_samples - HORIZON:
            done = True
            
        next_state = self.get_state(self.current_idx)
        
        return next_state, reward, done

def prepare_vectorized_data(df):
    """
    Converts DataFrame to Fast Numpy Arrays for the Environment.
    """
    print(" PREPARING VECTORIZED DATA...")
    
    # 1. Select Features
    feature_cols = [
        'velocity', 'acceleration', 'cvd_slope', 'bear_trap', 
        'vol_z', 'volume_ratio', 'dist_ema_20', 'dist_ema_200', 
        'staleness', 'weakness_score', 'is_fakeout'
    ]
    
    # Extract Matrix
    # Convert to float32 explicitly
    feature_matrix = df[feature_cols].values.astype(np.float32)
    
    # Add Padding Column for 12th dim
    padding = np.zeros((feature_matrix.shape[0], 1), dtype=np.float32)
    feature_matrix = np.hstack([feature_matrix, padding])
    
    # Normalize (Standardization Logic from Original Script)
    eps = 1e-9
    close_vals = df['close'].values.astype(np.float32)
    
    # Col 0: Velocity
    feature_matrix[:, 0] = feature_matrix[:, 0] / (close_vals + eps) * 10000
    # Col 1: Accel
    feature_matrix[:, 1] = feature_matrix[:, 1] / (close_vals + eps) * 10000
    # Col 2: CVD
    feature_matrix[:, 2] = feature_matrix[:, 2] / 1e6
    # Col 6: Dist EMA 20
    feature_matrix[:, 6] = feature_matrix[:, 6] * 100
    # Col 7: Dist EMA 200
    feature_matrix[:, 7] = feature_matrix[:, 7]
    # Col 8: Staleness
    feature_matrix[:, 8] = feature_matrix[:, 8] / 50.0
    # Col 9: Weakness
    feature_matrix[:, 9] = feature_matrix[:, 9] / 0.05
    
    # Arrays
    close_arr = df['close'].values.astype(np.float32)
    high_arr = df['high'].values.astype(np.float32)
    low_arr = df['low'].values.astype(np.float32)
    open_arr = df['open'].values.astype(np.float32)
    
    print(f"   Data Arrays Shape: {feature_matrix.shape}")
    return feature_matrix, close_arr, high_arr, low_arr, open_arr

def main():
    print(f"🐼 PANDA V1: VECTORIZED TRAINING START")
    print(f"   Objective: Optimize training speed for 96k candles.")
    
    # 1. Load Data
    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found: {DATASET_PATH}")
        return

    print("   Loading Raw Data...")
    df = pd.read_csv(DATASET_PATH)
    if 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    needed = ['velocity', 'bear_trap', 'vol_z', 'weakness_score']
    if not all(c in df.columns for c in needed):
         print("🧬 Calculating DNA...")
         df = calculate_phantom_dna(df)
    df.fillna(0, inplace=True)
    
    # 2. Prepare Arrays
    features, close_arr, high_arr, low_arr, open_arr = prepare_vectorized_data(df)
    
    # 3. Init Env
    env = VectorizedEnv(close_arr, high_arr, low_arr, open_arr, features)
    
    # 4. Training Setup
    device = torch.device("cpu")
    model = PandaNet(12, 2).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"   Training Started (Vectorized) on {len(df)} samples...")
    
    best_rew = -float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_reward = 0.0
        done_count = 0
        
        # One "Episode" per epoch (scanning random chunk of history)
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            # Epsilon Greedy
            epsilon = max(0.01, 1.0 - (epoch / (EPOCHS * 0.8))) 
            
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 2) # 0 or 1
            else:
                with torch.no_grad():
                    # Unsqueeze to Batch 1
                    q_values = model(torch.FloatTensor(state).unsqueeze(0).to(device))
                    action = torch.argmax(q_values).item()
            
            # Step
            next_state, reward, done = env.step(action)
            
            # Q-Learning Update (Online - No Replay Buffer for Speed in this Demo)
            # For pure stability we usually want Replay Buffer, but user script asked for this structure.
            # We will do a simple Online Update.
            
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            reward_t = torch.FloatTensor([reward]).unsqueeze(0).to(device)
            
            q_val = model(state_t)[0][action]
            
            with torch.no_grad():
                q_next = model(next_state_t).max(1)[0]
                target = reward_t + (0.95 * q_next) # Gamma 0.95
                
            loss = criterion(q_val, target[0])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            episode_reward += reward
            done_count += 1
            
            if done_count > 5000: # Limit episode length to prevent getting stuck
                break
        
        if epoch % 10 == 0:
             print(f"Ep {epoch:03d} | Reward: {episode_reward:.2f} | Steps: {done_count} | Eps: {epsilon:.2f}")
        
        if episode_reward > best_rew:
            best_rew = episode_reward
            if epoch > 10:
                torch.save(model.state_dict(), MODEL_DIR / MODEL_NAME)
            
    print(f"✅ PANDA V1 (VECTORIZED) TRAINED.")
    print(f"   Model Saved: {MODEL_DIR / MODEL_NAME}")

if __name__ == "__main__":
    main()
