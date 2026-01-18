#!/usr/bin/env python3
"""
Project Phantom: ETH Specialized DQN Trainer
Target: Train an agent that ONLY takes big drops (3%+).
Key Innovation: Asymmetric reward (PnL squared).
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

# Config
TRAINING_DATA_PATH = "data/phantom_training.pkl"

# Training Config
EPISODES = 300  # More episodes for rich dataset
BATCH_SIZE = 128  # Larger batch for stability
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LR = 0.0003  # Lower LR for stability with complex data
MEMORY_SIZE = 20000
TARGET_UPDATE = 10

# AMD ROCm GPU Detection
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 Using AMD GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU for training")
    return device

device = get_device()

class PhantomNet(nn.Module):
    """
    Phantom Neural Network.
    12-feature input with CVD Proxy.
    Deeper network for institutional patterns.
    """
    def __init__(self, input_dim=12, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
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

class PhantomEnvironment:
    """
    Phantom Environment for ETH.
    Uses pre-computed training data with asymmetric rewards.
    """
    def __init__(self, training_data):
        self.data = training_data.sample(frac=1).reset_index(drop=True)  # Shuffle
        self.current_idx = 0
    
    def reset(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)  # Re-shuffle
        self.current_idx = 0
        return self._get_state()
    
    def _get_state(self):
        if self.current_idx >= len(self.data):
            return None
        
        row = self.data.iloc[self.current_idx]
        state = np.array(row['state'], dtype=np.float32)
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        return state
    
    def step(self, action):
        if self.current_idx >= len(self.data):
            return None, 0, True
        
        row = self.data.iloc[self.current_idx]
        correct_action = row['action']
        
        # Asymmetric Reward
        if action == correct_action:
            if action == 1:  # Correct SHORT
                # Big reward for big drops (PnL squared effect)
                reward = row['reward']  # Already squared in data generator
            else:  # Correct PASS
                reward = 0.5  # Small positive
        else:
            if action == 1:  # Wrong SHORT (false positive)
                reward = -2.0  # Heavy penalty for false shorts
            else:  # Missed SHORT (false negative)
                reward = -1.0  # Moderate penalty
        
        self.current_idx += 1
        next_state = self._get_state()
        done = next_state is None
        
        return next_state, reward, done

class PhantomDQNAgent:
    def __init__(self):
        self.policy_net = PhantomNet().to(device)
        self.target_net = PhantomNet().to(device)
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
        
        loss = nn.SmoothL1Loss()(q_values.squeeze(), expected_q)  # Huber loss for stability
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

def main():
    print("🦅 PROJECT PHANTOM: ETH DQN TRAINING 🦅")
    print("=" * 60)
    print("📋 Key Innovations:")
    print("   - 12-feature state with CVD Proxy")
    print("   - Asymmetric reward (PnL²)")
    print("   - 350x more training data than Spectre")
    print("=" * 60)
    
    # Load training data
    if not os.path.exists(TRAINING_DATA_PATH):
        print("❌ Training data not found. Run phantom_data_generator.py first.")
        return
    
    training_data = pd.read_pickle(TRAINING_DATA_PATH)
    print(f"\n📊 Training data: {len(training_data)} examples")
    print(f"   SHORT: {(training_data['action'] == 1).sum()}")
    print(f"   PASS: {(training_data['action'] == 0).sum()}")
    
    # Create model directory
    model_dir = "models/phantom_eth"
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize
    agent = PhantomDQNAgent()
    env = PhantomEnvironment(training_data)
    
    best_reward = -float('inf')
    rewards_history = []
    
    print(f"\n🧠 Training Phantom for {EPISODES} episodes...\n")
    
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while state is not None:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            
            # Multiple training steps per env step (experience replay)
            for _ in range(2):
                agent.train_step()
            
            total_reward += reward
            state = next_state
            steps += 1
        
        agent.decay_epsilon()
        
        if episode % TARGET_UPDATE == 0:
            agent.update_target()
        
        rewards_history.append(total_reward)
        
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.policy_net.state_dict(), f"{model_dir}/phantom_net_best.pth")
        
        if episode % 10 == 0:
            avg = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else total_reward
            print(f"  Episode {episode}: Reward={total_reward:.2f} | Avg={avg:.2f} | Best={best_reward:.2f} | ε={agent.epsilon:.3f}")
    
    # Save final model
    torch.save(agent.policy_net.state_dict(), f"{model_dir}/phantom_net.pth")
    
    print(f"\n✅ Phantom training complete")
    print(f"📁 Model saved to: {model_dir}/")
    print(f"🏆 Best Reward: {best_reward:.2f}")

if __name__ == "__main__":
    main()
