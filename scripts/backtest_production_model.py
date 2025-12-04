import os
import sys
import json
import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Añadir directorio raíz al path
sys.path.append(os.getcwd())

from ml.advanced_models.improved_architecture import DeepTemporalNet
from ml.nn_pattern.features import build_feature_frame

def load_model_and_artifacts(symbol, timeframe, fold=5):
    # ... (sin cambios)
    """Carga el modelo y artefactos entrenados."""
    base_path = Path(f"models/advanced/{symbol}/{timeframe}")
    
    # Cargar configuración
    with open(base_path / "production_training_results.json", "r") as f:
        results = json.load(f)
        
    config = results["config"]
    model_config = results["model_config"]
    selected_features = results["selected_features"]
    
    # Cargar scaler y selector
    scaler = joblib.load(base_path / "scaler.pkl")
    selector = joblib.load(base_path / "feature_selector.pkl")
    
    # Reconstruir modelo
    model = DeepTemporalNet(
        input_dim=len(selected_features),
        hidden_dim=model_config["hidden_dim"],
        lstm_layers=model_config["lstm_layers"],
        dense_dims=model_config["dense_dims"],
        dropout=model_config["dropout"],
        use_attention=model_config["use_attention"],
        bidirectional=model_config["bidirectional"],
        num_classes=model_config["num_classes"],
        use_regression=model_config["use_regression"],
        num_attention_heads=model_config["num_attention_heads"]
    )
    
    # Cargar pesos
    model_path = base_path / f"best_model_fold{fold}.pt"
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, scaler, selector, config, selected_features

def run_backtest(symbol="BTCUSDT", timeframe="15m", initial_capital=10000, commission=0.0004):
    """Ejecuta backtest vectorial."""
    print(f"🚀 Iniciando Backtest para {symbol} {timeframe}...")
    
    # 1. Cargar modelo
    try:
        model, scaler, selector, config, selected_features = load_model_and_artifacts(symbol, timeframe)
        print("✅ Modelo cargado correctamente.")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return

    # 2. Cargar datos (usamos los mismos datos que para entrenamiento por ahora, pero filtraremos test)
    # En producción real, cargaríamos datos nuevos desde la DB o API
    from data.storage.database_manager import db_manager
    
    # Cargar suficientes datos para indicadores y backtest
    # Probar variantes de símbolo (BTCUSDT vs BTC/USDT)
    variants = [symbol]
    if "/" not in symbol:
        variants.append(symbol.replace("USDT", "/USDT"))
    else:
        variants.append(symbol.replace("/", ""))
        
    df = pd.DataFrame()
    for s in variants:
        print(f"   Probando símbolo: {s}...")
        df = db_manager.get_ohlcv_data(s, timeframe, limit=10000)
        if not df.empty:
            print(f"✅ Datos encontrados para {s}")
            break
            
    if df.empty:
        print(f"❌ No hay datos para backtest (probado: {variants})")
        return

    print(f"📊 Datos cargados: {len(df)} velas.")

    # 3. Preprocesamiento (Feature Engineering)
    print("⚙️  Calculando features...")
    features_df, _ = build_feature_frame(df)
    
    # Alinear índices (build_feature_frame elimina filas con NaNs iniciales)
    df = df.loc[features_df.index]
    
    # Seleccionar features usadas por el modelo
    try:
        X = features_df[selected_features].values
    except KeyError as e:
        print(f"❌ Faltan features en los datos: {e}")
        # Imprimir features disponibles vs esperadas
        missing = set(selected_features) - set(features_df.columns)
        print(f"   Features faltantes: {missing}")
        return
        
    # Escalar
    X_scaled = scaler.transform(X)
    
    # Preparar secuencias
    sequence_length = config["sequence_length"]
    X_seq = []
    indices = []
    
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
        indices.append(df.index[i+sequence_length])
        
    X_seq = np.array(X_seq)
    X_tensor = torch.FloatTensor(X_seq)
    
    # Forzar CPU para inferencia (más estable que ROCm para este script)
    device = torch.device('cpu')
    model = model.to(device)
    X_tensor = X_tensor.to(device)
    print("ℹ️  Usando CPU para inferencia (estabilidad)")
        
    # 4. Inferencia
    print("🔮 Generando predicciones...")
    batch_size = 1024
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            outputs = model(batch)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            
    # Debug: Imprimir distribución de predicciones
    print("\n🔍 DEBUG PREDICCIONES:")
    print("Distribución de clases:", pd.Series(predictions).value_counts().to_dict())
    probs_long = [p[1] for p in probabilities]
    probs_short = [p[2] for p in probabilities]
    print(f"Prob. Máx Long: {max(probs_long):.4f}, Media: {np.mean(probs_long):.4f}")
    print(f"Prob. Máx Short: {max(probs_short):.4f}, Media: {np.mean(probs_short):.4f}")
            
    # 5. Simulación de Trading
    print("💰 Simulando trading...")
    
    # Crear DataFrame de resultados
    results_df = pd.DataFrame(index=indices)
    results_df['Close'] = df.loc[indices, 'close']
    results_df['Prediction'] = predictions
    results_df['Prob_Neutral'] = [p[0] for p in probabilities]
    results_df['Prob_Long'] = [p[1] for p in probabilities]
    results_df['Prob_Short'] = [p[2] for p in probabilities]
    
    # Calcular EMA 200 para filtro de tendencia
    results_df['ema_200'] = df.loc[indices, 'close'].ewm(span=200, adjust=False).mean()
    
    # Lógica de Trading Vectorial
    # 0: Neutral, 1: Long, 2: Short
    
    position = 0 # 0: Flat, 1: Long, -1: Short
    equity = initial_capital
    equity_curve = [equity]
    trades = []
    
    entry_price = 0
    entry_time = None
    
    # Umbral de confianza
    CONFIDENCE_THRESHOLD = 0.34 # Mínimo posible
    
    for i in range(1, len(results_df)):
        current_bar = results_df.iloc[i]
        prev_bar = results_df.iloc[i-1]
        timestamp = results_df.index[i]
        price = current_bar['Close']
        ema_200 = current_bar['ema_200']
        
        pred = current_bar['Prediction']
        prob_long = current_bar['Prob_Long']
        prob_short = current_bar['Prob_Short']
        
        # Señal de entrada (SIN FILTRO DE TENDENCIA - DEBUG)
        signal = 0
        if pred == 1 and prob_long > CONFIDENCE_THRESHOLD:
            # if price > ema_200: 
            signal = 1
        elif pred == 2 and prob_short > CONFIDENCE_THRESHOLD:
            # if price < ema_200:
            signal = -1
            
        # Ejecución
        if position == 0:
            if signal == 1:
                # Abrir Long
                position = 1
                entry_price = price
                entry_time = timestamp
                equity -= equity * commission # Comisión entrada
            elif signal == -1:
                # Abrir Short
                position = -1
                entry_price = price
                entry_time = timestamp
                equity -= equity * commission
                
        elif position == 1:
            if signal == -1 or pred == 0: # Reverse o Cierre
                # Cerrar Long
                pnl = (price - entry_price) / entry_price
                equity *= (1 + pnl)
                equity -= equity * commission # Comisión salida
                trades.append({'type': 'Long', 'entry': entry_time, 'exit': timestamp, 'pnl': pnl, 'pnl_abs': equity - equity_curve[-1]})
                
                position = 0
                if signal == -1: # Reverse a Short
                    position = -1
                    entry_price = price
                    entry_time = timestamp
                    equity -= equity * commission
                    
        elif position == -1:
            if signal == 1 or pred == 0: # Reverse o Cierre
                # Cerrar Short
                pnl = (entry_price - price) / entry_price
                equity *= (1 + pnl)
                equity -= equity * commission
                trades.append({'type': 'Short', 'entry': entry_time, 'exit': timestamp, 'pnl': pnl, 'pnl_abs': equity - equity_curve[-1]})
                
                position = 0
                if signal == 1: # Reverse a Long
                    position = 1
                    entry_price = price
                    entry_time = timestamp
                    equity -= equity * commission
                    
        equity_curve.append(equity)
        
    # Métricas Finales
    total_return = (equity - initial_capital) / initial_capital * 100
    num_trades = len(trades)
    win_rate = len([t for t in trades if t['pnl'] > 0]) / num_trades if num_trades > 0 else 0
    
    print("\n" + "="*50)
    print(f"📊 RESULTADOS BACKTEST: {symbol} {timeframe}")
    print("="*50)
    print(f"Capital Inicial: ${initial_capital}")
    print(f"Capital Final:   ${equity:.2f}")
    print(f"Retorno Total:   {total_return:.2f}%")
    print(f"Total Trades:    {num_trades}")
    print(f"Win Rate:        {win_rate*100:.2f}%")
    
    if num_trades > 0:
        avg_pnl = np.mean([t['pnl'] for t in trades]) * 100
        print(f"Avg PnL:         {avg_pnl:.2f}%")
        
        # Guardar trades
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(f"backtest_trades_{symbol}_{timeframe}.csv")
        print(f"📝 Trades guardados en backtest_trades_{symbol}_{timeframe}.csv")
        
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title(f"Equity Curve - {symbol} {timeframe}")
    plt.ylabel("Capital ($)")
    plt.xlabel("Trades (Time)")
    plt.grid(True)
    plt.savefig(f"backtest_equity_{symbol}_{timeframe}.png")
    print(f"📈 Gráfico guardado en backtest_equity_{symbol}_{timeframe}.png")

if __name__ == "__main__":
    run_backtest(symbol="ETHUSDT", timeframe="15m")
