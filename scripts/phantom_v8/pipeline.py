import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import shap

# ==========================================
# 1. Forensic Cleaning
# ==========================================
def get_clean_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Handle timestamp column
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'open_time'}, inplace=True)
    
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace=True)
    df.sort_index(inplace=True)

    # 1. Winsorization (Eliminar mechas irreales > 3 sigmas)
    print("Applying Winsorization...")
    for col in ['high', 'low']:
        rolling_std = df[col].rolling(window=100).std()
        # Use close or open as reference? User used open.
        # df[col] > df['open'] + q_limit
        q_limit = rolling_std * 3
        
        upper_limit = df['open'] + q_limit
        lower_limit = df['open'] - q_limit
        
        df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
        df[col] = np.where(df[col] < lower_limit, lower_limit, df[col])

    # 2. Gap Filling (Rellenar minutos perdidos)
    print("Filling Gaps...")
    # Create full range index
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1min')
    df = df.reindex(full_idx).ffill()
    
    return df

# ==========================================
# 2.5. Feature Engineering Completo (ADN V8)
# ==========================================
def add_phantom_features(df):
    print("Engineering FULL Phantom Features (12 Dimensions)...")
    df = df.copy()
    
    # --- Features Base (Cinética) ---
    # Velocidad y Aceleración (Derivadas del precio)
    df['velocity'] = df['close'].diff(periods=5)
    df['acceleration'] = df['velocity'].diff(periods=5)

    # --- Features de Estructura de Mercado (CVD Proxy) ---
    # Si cvd no existe, usamos el Tick Rule
    if 'cvd' not in df.columns:
        price_diff = df['close'].diff()
        direction = np.sign(price_diff).replace(0, np.nan).ffill().fillna(0)
        df['cvd'] = (direction * df['volume']).cumsum()

    df['cvd_slope'] = df['cvd'].diff(periods=10)
    df['price_slope'] = df['close'].diff(periods=10)
    df['bear_trap'] = ((df['price_slope'] > 0) & (df['cvd_slope'] < 0)).astype(float)

    # --- Features de Volatilidad y Entropía ---
    rolling_std_20 = df['close'].rolling(20).std()
    rolling_std_200 = df['close'].rolling(200).std()
    df['vol_z'] = (rolling_std_20 - rolling_std_200) / (rolling_std_200 + 1e-8)
    
    # Volume Ratio (RVOL): ¿Es el volumen actual mayor al promedio?
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['vol_ma'] + 1e-8)

    # --- Features de Tendencia y Agotamiento (EMAs) ---
    ema_20 = df['close'].ewm(span=20).mean()
    ema_200 = df['close'].ewm(span=200).mean()
    
    # Distancia porcentual a las EMAs (Normalizado)
    df['dist_ema_20'] = (df['close'] - ema_20) / df['close']
    df['dist_ema_200'] = (df['close'] - ema_200) / df['close']

    # --- Features de "Frescura" (Staleness) ---
    # ¿Cuántas velas lleva sin hacer un nuevo máximo? (Agotamiento de compra)
    # Si staleness es alto, el mercado está "cansado" y es vulnerable a un Short
    df['is_high'] = df['close'] == df['close'].rolling(20).max()
    # Contamos eventos cumulativos y reseteamos cuando hay nuevo high
    # Corrected logic: Count consecutive False values in is_high
    s = ~df['is_high']
    df['staleness'] = s.groupby((s != s.shift()).cumsum()).cumsum()

    # --- Weakness Score (Proxy vs BTC) ---
    # NOTA: Para esto necesitaría datos de BTC en el mismo DF. 
    # Si no los tiene, usemos una proxy interna: Retorno relativo vs Volatilidad
    df['returns'] = df['close'].pct_change()
    df['weakness_score'] = (df['returns'].rolling(20).sum() / (df['vol_z'] + 1e-8)).clip(-5, 5)

    # --- Fakeout Detection (Wick Rejection) ---
    # Si el alto es muy alto pero el cierre es cerca del bajo = Rechazo
    df['range'] = df['high'] - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['is_fakeout'] = ((df['high'] > df['open'] * 1.005) & (df['range'] > df['body'] * 2)).astype(float)

    # Padding/Reserved para mantener dimensionalidad
    df['reserved'] = 0.0

    # Seleccionamos las 12 columnas finales
    feature_cols = [
        'velocity', 'acceleration', 
        'cvd_slope', 'bear_trap', 
        'vol_z', 'volume_ratio', 
        'dist_ema_20', 'dist_ema_200', 
        'staleness', 'weakness_score', 
        'is_fakeout', 'reserved'
    ]
    
    return df[feature_cols].dropna(), df # Retornamos features y df completo

# ==========================================
# 3. Asymmetric Loss & Model
# ==========================================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma=5.0): # Gamma 5 penaliza duramente los Falsos Positivos
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        loss = self.bce(pred, target)
        # Si el target es 0 (No entrar) pero predijo 1 (Short) -> Penalización extra
        # Target shape needs to match pred
        weights = torch.where((target == 0) & (pred > 0.5), self.gamma, 1.0)
        return (loss * weights).mean()

class PhantomShortNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# ==========================================
# 4. Backtest V8 Engine
# ==========================================
def run_v8_backtest(df, model_predictions):
    print("Running Backtest V8...")
    balance = 20.0
    active_trade = None
    equity_curve = [balance]
    trades = []
    
    # Ensure predictions align with df
    # We assume model_predictions is a numpy array or tensor matching df length
    
    for i in range(len(df)):
        row = df.iloc[i]
        timestamp = df.index[i]
        current_price = row['close']
        
        # 1. Filtro Time Sentinel (No Martes, No horas de baja liquidez)
        # Tuesday is day 1
        if timestamp.day_of_week == 1 or timestamp.hour in [0, 23]:
            if active_trade: # Cerrar por seguridad si entra el filtro
                pnl_pct = (active_trade['entry'] - current_price) / active_trade['entry']
                pnl = active_trade['capital'] * pnl_pct
                balance += pnl
                trades.append({'entry_time': active_trade['entry_time'], 'exit_time': timestamp, 'pnl': pnl, 'reason': 'TimeSentinel'})
                active_trade = None
            equity_curve.append(balance)
            continue

        # 2. Lógica de Salida/Nueva Señal (Exit-on-Signal)
        # Use pre-calculated prediction for speed in loop
        prediction = model_predictions[i]
        
        # Sensitivity Experiment: Lower threshold to 0.60
        if prediction > 0.60: # Señal de Short fuerte (Reduced from 0.75)
            if active_trade:
                # Close existing
                pnl_pct = (active_trade['entry'] - current_price) / active_trade['entry']
                pnl = active_trade['capital'] * pnl_pct
                balance += pnl
                trades.append({'entry_time': active_trade['entry_time'], 'exit_time': timestamp, 'pnl': pnl, 'reason': 'SignalFlip'})
                active_trade = None
            
            # Abrir nueva con 100% de capital acumulado
            active_trade = {
                'entry': current_price,
                'sl': current_price * 1.02, # SL inicial 2%
                'peak_price': current_price, # For short, peak is minimum price seen? No, peak profit means lowest price.
                # Wait, for trailing stop logic below:
                # "Si el precio baja (a favor del Short), bajamos el SL"
                # So 'peak_price' tracks the lowest price seen since entry.
                'lowest_price': current_price,
                'capital': balance,
                'entry_time': timestamp
            }

        # 3. Trailing Stop (Pirámide de seguridad)
        if active_trade:
            # Si el precio baja (a favor del Short), bajamos el SL
            if current_price < active_trade['lowest_price']:
                active_trade['lowest_price'] = current_price
                # El SL persigue al precio a una distancia del 1.5%
                new_sl = current_price * 1.015 
                active_trade['sl'] = min(active_trade['sl'], new_sl)

            # Verificación de Stop Loss
            if row['high'] >= active_trade['sl']:
                # Stopped out
                pnl_pct = (active_trade['entry'] - active_trade['sl']) / active_trade['entry']
                pnl = active_trade['capital'] * pnl_pct
                balance += pnl
                trades.append({'entry_time': active_trade['entry_time'], 'exit_time': timestamp, 'pnl': pnl, 'reason': 'StopLoss'})
                active_trade = None
        
        equity_curve.append(balance)

    return balance, equity_curve, trades

# ==========================================
# Main Execution
# ==========================================
def main():
    csv_path = 'ETH_CLEAN_FOR_THESIS.csv'
    
    # 1. Clean Data
    df_clean = get_clean_data(csv_path)
    print(f"Data cleaned. Shape: {df_clean.shape}")
    
    # 2. Features (Using new 12D function)
    df_features, df_full = add_phantom_features(df_clean)
    print(f"Features added. Shape: {df_features.shape}")
    
    # Align df_full with df_features (dropna happened)
    df_full = df_full.loc[df_features.index]
    
    # ==========================================
    # TARGETING V2: Cazador de Colapsos Reales
    # ==========================================
    print("Generating Advanced Targets (Real Crash Definition)...")
    
    lookforward = 45 # Buscamos 45 minutos en el futuro
    crash_threshold = 0.015 # Buscamos una caída del 1.5%
    stop_threshold = 0.005 # Si sube 0.5% antes de caer, es invalido (Stop Loss hit)
    
    # Calculamos el mínimo y el máximo en el futuro
    future_min = df_full['low'].rolling(window=lookforward).min().shift(-lookforward)
    future_max = df_full['high'].rolling(window=lookforward).max().shift(-lookforward)
    
    current_price = df_full['close']
    
    # La señal es válida SÍ:
    # 1. El precio baja > 1.5% (Target Hit)
    # 2. El precio NO subió > 0.5% antes de bajar (Stop Loss Avoided)
    df_full['target'] = (
        (future_min < current_price * (1 - crash_threshold)) & 
        (future_max < current_price * (1 + stop_threshold))
    ).astype(float)

    # Drop NaNs from target generation
    valid_indices = df_full.dropna().index
    df_full = df_full.loc[valid_indices]
    df_features = df_features.loc[valid_indices]
    
    # Features list
    feature_cols = [
        'velocity', 'acceleration', 
        'cvd_slope', 'bear_trap', 
        'vol_z', 'volume_ratio', 
        'dist_ema_20', 'dist_ema_200', 
        'staleness', 'weakness_score', 
        'is_fakeout', 'reserved'
    ]
    X = df_features[feature_cols].values
    y = df_full['target'].values
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    
    # ==========================================
    # RE-ENTRENAMIENTO (Valquiria Protocol)
    # ==========================================
    print("Retraining PhantomShortNet with Relaxed Constraints...")
    model = PhantomShortNet(input_size=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # CAMBIO CRÍTICO: Gamma de 5.0 a 1.5
    # Le damos permiso para equivocarse 1 de cada 3 veces si la recompensa es alta
    criterion = AsymmetricLoss(gamma=1.5) 
    
    epochs = 100
    
    # Scheduler para ajustar el aprendizaje si se estanca
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")
            
    # 4. Backtest
    print("Generating Predictions for Backtest...")
    model.eval()
    with torch.no_grad():
        # Predict on whole dataset for backtest
        X_all_t = torch.FloatTensor(X)
        all_preds = model(X_all_t).numpy().flatten()
    
    final_balance, equity, trades = run_v8_backtest(df_full, all_preds)
    
    print(f"\nInitial Balance: $20.00")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Total Trades: {len(trades)}")
    
    # Visualization (Equity Curve)
    plt.figure(figsize=(12, 6))
    plt.plot(equity)
    plt.title("Phantom V8 Equity Curve (Log Scale)")
    plt.yscale('log')
    plt.ylabel('Balance (USD)')
    plt.xlabel('Time (Minutes)')
    plt.savefig('phantom_v8_equity.png')
    print("Equity curve saved to phantom_v8_equity.png")

    # 5. SHAP Explainability
    print("Generating SHAP Explainability Plot for Thesis...")
    # Use a background sample for DeepExplainer
    background = X_train_t[:100]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(X_test_t[:50])
    
    # Graficar
    plt.figure()
    shap.summary_plot(shap_values, X_test_t[:50].numpy(), feature_names=feature_cols, show=False)
    plt.title("Phantom V8: Feature Importance (Why do we Short?)")
    plt.savefig('phantom_v8_shap_explainability.png')
    print("SHAP plot saved.")

if __name__ == "__main__":
    main()
