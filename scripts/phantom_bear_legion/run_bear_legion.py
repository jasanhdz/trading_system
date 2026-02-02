#!/usr/bin/env python3
"""
🌊 PROJECT OCEAN: THE BEAR LEGION (FINAL)
The fully assembled Cyborg Trinity System.

ARCHITECTURE:
1.  **Regime Router:** Activates ONLY in BEAR_TREND.
2.  **Macro Shield:** Activates ONLY if Price < EMA 200.
3.  **The Judge (Forensic):** Vetoes Unsafe setups (Score < 0.50).
4.  **The Specialists (Switch):**
    *   High Volatility (>0.5%): 🐻 **GRIZZLY** (Sniper Mode).
    *   Low Volatility (<0.5%):  🐼 **PANDA** (Grinder Mode).

This script runs the final simulation of the combined army.
"""
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Fix path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from scripts.phantom_v9.detect_phantom_tops import calculate_phantom_dna

# --- CONFIGURACIÓN MAESTRA ---
DATA_PATH = ROOT_DIR / "scripts/phantom_bear_legion/data/regime_labeled_history.csv"
MODEL_GRIZZLY_PATH = ROOT_DIR / "scripts/phantom_bear_legion/models/grizzly_v1.pth"
MODEL_PANDA_PATH = ROOT_DIR / "scripts/phantom_bear_legion/models/panda_v1_vectorized.pth"
MODEL_JUDGE_PATH = ROOT_DIR / "scripts/phantom_bear_legion/models/judge_v1.pth"

# PARAMETROS TÁCTICOS
LEVERAGE = 20.0
FIXED_MARGIN = 20.0
EMA_200_SPAN = 200 * 24 * 12 # 57,600 velas

# 🐻 GRIZZLY CONFIG (Sniper) - High Vol (OPTIMIZED)
GRIZZLY_SL = 0.025 # 2.5%
GRIZZLY_TP = 0.070 # 7.0%
GRIZZLY_HORIZON = 48
VOL_THRESHOLD_HIGH = 0.005 # 0.5% cutoff

# 🐼 PANDA CONFIG (Grinder) - Low Vol (OPTIMIZED)
PANDA_SL = 0.015 # 1.5%
PANDA_TP = 0.010 # 1.0%
PANDA_HORIZON = 96

# ⚖️ JUDGE CONFIG
JUDGE_THRESHOLD = 0.50

# --- REDES NEURONALES ---

class BearLegionNet(nn.Module): # Used for Grizzly (13 inputs)
    def __init__(self):
        super(BearLegionNet, self).__init__()
        self.fc1 = nn.Linear(12, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

class PandaNet(nn.Module): # Used for Panda (12 inputs)
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

class JudgeNet(nn.Module):
    def __init__(self, input_dim):
        super(JudgeNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.sigmoid(self.fc3(x)) 

def get_features_vectorized(row, close_arr):
    """
    Returns features for Panda/Judge (12 dims) or Grizzly (12 dims).
    Note: Standardized on 12 dims for compatibility. 
    In original Grizzly script, it used 12 inputs.
    """
    state_list = [
        row['velocity'] / row['close'] * 10000,
        row['acceleration'] / row['close'] * 10000,
        row['cvd_slope'] / 1e6,
        0.0 if pd.isna(row.get('bear_trap')) else row['bear_trap'],
        row['vol_z'],
        row['volume_ratio'],
        row['dist_ema_20'] * 100,
        row['dist_ema_200'] * 100,
        row['staleness'] / 50.0,
        row['weakness_score'] / 0.05, # Normalized
        row['is_fakeout'],
        0.0 # Padding
    ]
    return np.array(state_list, dtype=np.float32)

def main():
    print(f"⚔️ THE BEAR LEGION: FINAL ASSEMBLY VERIFICATION")
    print(f"   Simulating Combined Army (Grizzly + Panda) on 3.5 Years...")
    
    # 1. Load Data
    if not DATA_PATH.exists(): return
    df = pd.read_csv(DATA_PATH)
    if 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    needed = ['velocity', 'weakness_score', 'vol_z']
    if not all(c in df.columns for c in needed):
         df = calculate_phantom_dna(df)
    df.fillna(0, inplace=True)
    
    df['ema_200_macro'] = df['close'].ewm(span=EMA_200_SPAN).mean()
    
    # 2. Load Models
    device = torch.device("cpu")
    
    # GRIZZLY
    grizzly = BearLegionNet().to(device)
    try:
        grizzly.load_state_dict(torch.load(MODEL_GRIZZLY_PATH, map_location=device))
        grizzly.eval()
        print("✅ GRIZZLY (Sniper) Ready.")
    except:
        print("❌ GRIZZLY NOT FOUND.")
        return

    # PANDA
    panda = PandaNet(12, 2).to(device)
    try:
        panda.load_state_dict(torch.load(MODEL_PANDA_PATH, map_location=device))
        panda.eval()
        print("✅ PANDA (Grinder) Ready.")
    except:
        print("❌ PANDA NOT FOUND.")
        return

    # JUDGE
    judge = JudgeNet(12).to(device)
    try:
        judge.load_state_dict(torch.load(MODEL_JUDGE_PATH, map_location=device))
        judge.eval()
        print("✅ JUDGE (Forense) Ready.")
    except:
        print("⚠️ JUDGE Disabled.")
        judge = None

    # 3. Simulation
    balance = 100.0
    trades = []
    active_trade = None # (entry_idx, entry_price, sl, tp, horizon, type)
    
    stats = {'grizzly_trades': 0, 'panda_trades': 0, 'vetoed': 0}
    
    print(f"   🚀 LEGION MARCHING ({len(df)} candles)...")
    
    for i in range(len(df) - PANDA_HORIZON):
        row = df.iloc[i]
        
        # --- ROUTER CHECKS ---
        # 1. Bear Trend Only
        if row.get('regime_type', 'NEUTRAL') != 'BEAR_TREND':
            if not active_trade: continue
            
        # 2. Macro Shield
        if row['close'] >= row['ema_200_macro']:
            if not active_trade: continue
            
        # 3. Manage Active Trade
        if active_trade:
            entry_idx, entry_price, sl, tp, limit, agent_type = active_trade
            steps_held = i - entry_idx
            
            outcome = None
            exit_price = row['close']
            
            if steps_held >= limit:
                outcome = "TIMEOUT"
            elif row['high'] >= sl:
                outcome = "SL"
                exit_price = sl
            elif row['low'] <= tp:
                outcome = "TP"
                exit_price = tp
            
            if outcome:
                raw_pnl = (entry_price - exit_price) / entry_price
                lev_pnl = raw_pnl * LEVERAGE
                profit = FIXED_MARGIN * lev_pnl
                balance += profit
                
                trades.append({
                    'time': row['timestamp'],
                    'type': agent_type,
                    'outcome': outcome,
                    'pnl_pct': raw_pnl * 100,
                    'balance': balance
                })
                active_trade = None
            continue
            
        # --- ENTRY LOGIC ---
        
        # 1. Judge Veto (Common for both)
        vec = get_features_vectorized(row, None)
        t_vec = torch.FloatTensor(vec).unsqueeze(0).to(device)
        
        is_safe = True
        if judge:
            with torch.no_grad():
                score = judge(t_vec).item()
                if score < JUDGE_THRESHOLD: is_safe = False
        
        if not is_safe:
            stats['vetoed'] += 1
            continue
            
        # 2. Specialist Selection (Volatility Switch)
        vol = row['vol_z'] if not pd.isna(row['vol_z']) else 0.0
        
        # Determine Agent based on Original Vol Threshold
        # Note: 'vol_z' is z-score, 'vol' in logic meant pct change (high-low)/open
        # Let's check raw vol
        raw_vol = (row['high'] - row['low']) / row['open']
        
        if raw_vol >= VOL_THRESHOLD_HIGH:
            # 🐻 GRIZZLY PATH
            with torch.no_grad():
                q = grizzly(t_vec)
                action = torch.argmax(q).item()
            
            if action == 1:
                sl = row['close'] * (1 + GRIZZLY_SL)
                tp = row['close'] * (1 - GRIZZLY_TP)
                active_trade = (i, row['close'], sl, tp, GRIZZLY_HORIZON, 'GRIZZLY')
                stats['grizzly_trades'] += 1
                
        else:
            # 🐼 PANDA PATH
            with torch.no_grad():
                q = panda(t_vec)
                action = torch.argmax(q).item()
            
            if action == 1:
                sl = row['close'] * (1 + PANDA_SL)
                tp = row['close'] * (1 - PANDA_TP)
                active_trade = (i, row['close'], sl, tp, PANDA_HORIZON, 'PANDA')
                stats['panda_trades'] += 1
                
    # --- REPORT ---
    print("\n" + "="*60)
    print("🏁 THE LEGION REPORT (COMBINED)")
    print("="*60)
    
    if not trades:
        print("❌ No trades taken.")
    else:
        rdf = pd.DataFrame(trades)
        wins = rdf[rdf['pnl_pct'] > 0]
        
        print(f"Initial Balance: $100.00")
        print(f"Final Balance:   ${balance:.2f}")
        print(f"Net Return:      {((balance - 100)/100)*100:.2f}%")
        print(f"Total Trades:    {len(rdf)}")
        print(f"Win Rate:       {len(wins)/len(rdf)*100:.2f}%")
        
        print("\n📊 UNIT PERFORMANCE:")
        g_trades = rdf[rdf['type'] == 'GRIZZLY']
        p_trades = rdf[rdf['type'] == 'PANDA']
        
        if not g_trades.empty:
            g_ret = g_trades['pnl_pct'].sum() * LEVERAGE * (FIXED_MARGIN/100) # Approx
            print(f"   🐻 Grizzly: {len(g_trades)} trades (Win Rate: {len(g_trades[g_trades['pnl_pct']>0])/len(g_trades)*100:.1f}%)")
        else:
            print("   🐻 Grizzly: 0 trades")
            
        if not p_trades.empty:
            print(f"   🐼 Panda:   {len(p_trades)} trades (Win Rate: {len(p_trades[p_trades['pnl_pct']>0])/len(p_trades)*100:.1f}%)")
        else:
            print("   🐼 Panda:   0 trades")
            
        out_path = ROOT_DIR / "scripts/phantom_bear_legion/reports/legion_final_results.csv"
        rdf.to_csv(out_path, index=False)
        print(f"\n📄 Saved to {out_path}")

if __name__ == "__main__":
    main()
