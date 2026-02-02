#!/usr/bin/env python3
"""
THE EXAM: Final Validation (The Ocean)
Proves the 'Bear Legion' hypothesis on the full 3.5 year history.
Logic:
1. If Regime == BULL/NEUTRAL -> IDLE (Safety).
2. If Regime == BEAR + Low Vol -> IDLE (Panda Paused).
3. If Regime == BEAR + High Vol -> ACTIVATE GRIZZLY.
"""
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Fix path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from scripts.phantom_v9.detect_phantom_tops import calculate_phantom_dna

# --- CONFIGURACIÓN FINAL ---
DATA_PATH = ROOT_DIR / "scripts/phantom_bear_legion/data/regime_labeled_history.csv"
MODEL_PATH = ROOT_DIR / "scripts/phantom_bear_legion/models/grizzly_v1.pth"

# Físicas de Trading (Alineadas con producción)
LEVERAGE = 20.0
SL_PCT = 0.030 # 3% (Igual que el entrenamiento)
TP_PCT = 0.060 # 6%
HORIZON = 48
FIXED_MARGIN = 20.0 # Arriesgar $20 USD fijos por trade

# Umbrales del Router
# Umbrales del Router
VOL_THRESHOLD_CRASH = 0.005 # 0.5% (Reverted to Original Sensitivity)

def main():
    print("🌊 THE EXAM: FINAL VALIDATION (OCEAN)")
    print("   Validating Grizzly V1 on 3.5 years of history...")
    
    if not DATA_PATH.exists():
        print(f"❌ History not found: {DATA_PATH}")
        return
        
    # 1. Cargar el Océano (Historial Completo)
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Asegurar que tenemos las columnas de ADN (si el CSV de clasificador las tiró)
    needed = ['velocity', 'acceleration', 'cvd_slope', 'bear_trap', 'vol_z', 'volume_ratio', 'dist_ema_20', 'dist_ema_200', 'staleness', 'weakness_score', 'is_fakeout']
    if not all(c in df.columns for c in needed):
        print("🧬 Calculating Phantom DNA on History...")
        df = calculate_phantom_dna(df)
        df.fillna(0, inplace=True)
    else:
        print("✅ DNA already present in History CSV.")

    # --- MACRO FILTER (EMA 200 DAYS) ---
    # 200 Days * 24 Hours * 12 (5m candles) = 57,600
    print("🌊 Calculating Macro Trend (EMA 200 Days)...")
    df['ema_macro'] = df['close'].ewm(span=57600).mean()
    # -----------------------------------

    # 2. Cargar el Grizzly
    device = torch.device("cpu")
    # Define la red de forma inline o importa si es un archivo separado.
    # Para este script, usaremos la definición inline para asegurar compatibilidad.
    
    class BearLegionNet(torch.nn.Module):
        def __init__(self):
            super(BearLegionNet, self).__init__()
            self.fc1 = torch.nn.Linear(12, 256)
            self.fc2 = torch.nn.Linear(256, 128)
            self.fc3 = torch.nn.Linear(128, 64)
            self.fc4 = torch.nn.Linear(64, 2)
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(0.3)

        def forward(self, x):
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.relu(self.fc3(x))
            return self.fc4(x)

    model = BearLegionNet().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("✅ Grizzly V1 Loaded.")
    except Exception as e:
        print(f"❌ Failed to load Grizzly: {e}")
        return

    # 3. Bucle de Simulación (El Router en Acción)
    balance = 100.0
    trades = []
    
    # Estadísticas de seguridad
    bull_market_trades = 0
    active_trade = None
    
    print(f"🚀 Simulating {len(df)} candles with Router Logic...")
    
    for i in range(len(df) - HORIZON):
        row = df.iloc[i]
        
        # --- ROUTER LÓGICA ---
        current_regime = row['regime_type'] # BEAR_TREND, BULL_TREND, NEUTRAL
        
        # FILTRO 1: Si no es Oso, no hacer nada (Seguridad en Toros)
        if current_regime != 'BEAR_TREND':
            if active_trade: 
                pass 
            else:
                continue

        # FILTRO 1.5: MACRO TREND CHECK (The Holy Grail)
        # Si el precio está SOBRE la EMA 200 Días, es un "Dip en Bull Market". NO SHORT.
        if row['close'] > row['ema_macro']:
             if not active_trade and i % 10000 == 0:
                 # print(f"   🛡️ MACRO SHIELD: Blocked Grizzly (Price > EMA 200D) at {row['timestamp']}")
                 pass
             continue
        
        # FILTRO 2: Calcular Volatilidad Intra-Candle para seleccionar Grizzly
        # Si es Oso, ¿Es un Crash o un Grinder?
        # Volatilidad = (High - Low) / Open
        current_vol = (row['high'] - row['low']) / row['open']
        
        if current_vol < VOL_THRESHOLD_CRASH:
            # Es un Grinder. PAUSAR (No tenemos modelo Panda listo)
            # Si hay trade abierto, lo dejamos correr, pero no abrimos nuevos
            if not active_trade and i % 10000 == 0:
                 pass
            # Nota: Si tuvieras el Panda, aquí cargarías el modelo Panda.
            continue
        else:
            # Es un CRASH. ACTIVAR GRIZZLY.
            if not active_trade and i % 10000 == 0:
                 pass

        # --- GESTIÓN DE POSICIÓN (Si estamos dentro de un trade) ---
        if active_trade:
            entry_idx, entry_price, sl, tp = active_trade
            
            # Simular tiempo transcurrido
            steps_held = i - entry_idx
            if steps_held >= HORIZON:
                # Time Exit
                exit_price = row['close']
                reason = "TIME"
                active_trade = None # Cerrar trade
            else:
                # Check SL / TP
                if row['high'] >= sl:
                    exit_price = sl
                    reason = "SL"
                    active_trade = None
                elif row['low'] <= tp:
                    exit_price = tp
                    reason = "TP"
                    active_trade = None
                else:
                    continue # Seguir dentro
            
            if active_trade is None: # Trade cerró justo arriba
                raw_pnl = (entry_price - exit_price) / entry_price
                leveraged_pnl = raw_pnl * LEVERAGE
                realized_pnl = FIXED_MARGIN * leveraged_pnl
                
                balance += realized_pnl
                
                trades.append({
                    'entry_time': df.iloc[entry_idx]['timestamp'],
                    'exit_time': row['timestamp'],
                    'regime': current_regime,
                    'reason': reason,
                    'pnl_pct': raw_pnl * 100,
                    'balance': balance
                })
                
                # Contador de seguridad
                if '2024' in str(df.iloc[entry_idx]['timestamp']):
                    bull_market_trades += 1
                    print(f"   ⚠️ 2024 TRADE: {df.iloc[entry_idx]['timestamp']} | Volatility: {current_vol:.5f} ({current_vol*100:.2f}%)")

        # --- ENTRADA (Si no hay trade y estamos en régimen de Crash) ---
        if active_trade is None:
            # Reconstruir Estado para el Modelo (Igual que en entrenamiento)
            state = np.array([
                row['velocity'] / row['close'] * 10000,
                row['acceleration'] / row['close'] * 10000,
                row['cvd_slope'] / 1e6,
                row['bear_trap'],
                row['vol_z'],
                row['volume_ratio'],
                row['dist_ema_20'] * 100,
                row['dist_ema_200'] * 100,
                row['staleness'] / 50.0,
                row['weakness_score'],
                row['is_fakeout'],
                0.0
            ], dtype=np.float32)
            
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q = model(state_t)
                action = torch.argmax(q).item()
            
            if action == 1: # FIRE SHORT
                entry_price = row['close']
                sl = entry_price * (1 + SL_PCT)
                tp = entry_price * (1 - TP_PCT)
                active_trade = (i, entry_price, sl, tp) # Abrir trade
                
                # Contador de seguridad
                if '2024' in str(row['timestamp']):
                    bull_market_trades += 1
    
    # --- REPORTE FINAL ---
    print("\n" + "="*50)
    print("🏁 FINAL RESULTS (THE OCEAN VALIDATION)")
    print("="*50)
    
    if not trades:
        print("No trades taken.")
    else:
        rdf = pd.DataFrame(trades)
        wins = rdf[rdf['pnl_pct'] > 0]
        win_rate = len(wins) / len(trades) * 100
        
        print(f"Initial Balance: $100.00")
        print(f"Final Balance:   ${balance:.2f}")
        print(f"Net Return:      {((balance - 100)/100)*100:.2f}%")
        print(f"Total Trades:    {len(trades)}")
        print(f"Win Rate:       {win_rate:.2f}%")
        
        print(f"\n🛡️ SAFETY CHECK (The Holy Grail Test):")
        print(f"   Trades in 2024 (Bull Year): {bull_market_trades}")
        
        if bull_market_trades == 0:
            print("   ✅ PERFECT SCORE: Bot stayed IDLE in Bull Market.")
            print("   System is IMMUNE to Bull Traps.")
        elif bull_market_trades < 5:
            print(f"   ✅ SAFE: Very low activity in Bull Market ({bull_market_trades} trades).")
            print("   System successfully avoided buying the top.")
        else:
            print(f"   ⚠️ WARNING: High activity in Bull Market ({bull_market_trades} trades).")
            print("   Filter needs adjustment.")
            
        out_path = ROOT_DIR / "scripts/phantom_bear_legion/reports/ocean_validation_results.csv"
        # Ensure dir exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rdf.to_csv(out_path, index=False)
        print(f"\n📄 Report saved to {out_path}")

if __name__ == "__main__":
    main()
