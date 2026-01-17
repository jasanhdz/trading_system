#!/usr/bin/env python3
"""
Script de Explicabilidad "Berzerker AI"
Objetivo: Comparar estadísticamente las "Olas Ganadoras" vs "Olas Perdedoras".
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from data.storage.database_manager import db_manager
from utils.logger import setup_logger

logger = setup_logger("berzerker_explainer")

import argparse

def explain_winners():
    parser = argparse.ArgumentParser(description='Explain Berzerker Winners')
    parser.add_argument('--symbol', type=str, default='XRP/USDT', help='Symbol to analyze')
    args = parser.parse_args()
    
    symbol = args.symbol
    timeframe = '5m'
    
    logger.info(f"Cargando datos para {symbol} {timeframe}...")
    df = db_manager.get_ohlcv_data(symbol, timeframe)
    
    if df.empty and timeframe == '5m':
        logger.warning("No hay datos de 5m. Intentando resamplear desde 1m...")
        df_1m = db_manager.get_ohlcv_data(symbol, '1m')
        if not df_1m.empty:
            logger.info(f"Datos 1m cargados: {len(df_1m)} velas. Resampleando a 5m...")
            df = df_1m.resample('5min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
    
    if df.empty:
        logger.error("No hay datos.")
        return

    # --- Feature Engineering (Igual que en training) ---
    df['returns'] = df['close'].pct_change()
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Volumen relativo
    df['vol_ma_20'] = df['volume'].rolling(20).mean()
    df['vol_factor'] = df['volume'] / (df['vol_ma_20'] + 1e-8)
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # --- Identificación de Candidatos (2 Velas) ---
    df['is_green'] = df['close'] > df['open']
    df['prev_green'] = df['is_green'].shift(1)
    df['is_candidate'] = (df['is_green']) & (df['prev_green']) & (df['vol_factor'] > 1.5)
    
    candidates = df[df['is_candidate']].copy()
    logger.info(f"Total Candidatos (2 Velas + Vol > 1.5): {len(candidates)}")
    
    # --- Etiquetado (Labelling) ---
    TP_PCT = 0.03
    SL_PCT = 0.015
    HORIZON = 48 
    
    labels = []
    
    for idx, row in candidates.iterrows():
        try:
            loc = df.index.get_loc(idx)
        except:
            labels.append(0)
            continue
            
        if loc + HORIZON >= len(df):
            labels.append(0)
            continue
            
        future = df.iloc[loc+1 : loc+HORIZON+1]
        entry_price = row['close']
        tp_price = entry_price * (1 + TP_PCT)
        sl_price = entry_price * (1 - SL_PCT)
        
        outcome = 0 
        for _, f_row in future.iterrows():
            if f_row['low'] <= sl_price:
                outcome = 0 
                break
            if f_row['high'] >= tp_price:
                outcome = 1 
                break
        labels.append(outcome)
        
    candidates['is_winner'] = labels
    
    # --- Comparativa Estadística ---
    winners = candidates[candidates['is_winner'] == 1]
    losers = candidates[candidates['is_winner'] == 0]
    
    print("\n" + "="*60)
    print(f"ANÁLISIS DE ADN BERZERKER ({len(winners)} Ganadores vs {len(losers)} Perdedores)")
    print("="*60)
    
    features = {
        'Volumen Factor (x veces promedio)': 'vol_factor',
        'RSI (Fuerza Relativa)': 'rsi',
        'Cuerpo Vela (% precio)': 'body', # Necesitamos normalizar esto
        'Mecha Superior (% precio)': 'upper_wick'
    }
    
    # Normalizar para display
    candidates['body_pct'] = (candidates['body'] / candidates['open']) * 100
    candidates['wick_pct'] = (candidates['upper_wick'] / candidates['open']) * 100
    
    winners = candidates[candidates['is_winner'] == 1]
    losers = candidates[candidates['is_winner'] == 0]
    
    print(f"{'FEATURE':<30} | {'GANADORES (Media)':<18} | {'PERDEDORES (Media)':<18} | {'DIFERENCIA':<10}")
    print("-" * 85)
    
    stats = [
        ('Volumen Factor', 'vol_factor', '{:.2f}x'),
        ('RSI', 'rsi', '{:.1f}'),
        ('Tamaño Cuerpo', 'body_pct', '{:.3f}%'),
        ('Mecha Superior', 'wick_pct', '{:.3f}%')
    ]
    
    for name, col, fmt in stats:
        w_mean = winners[col].mean()
        l_mean = losers[col].mean()
        diff = ((w_mean - l_mean) / l_mean) * 100
        print(f"{name:<30} | {fmt.format(w_mean):<18} | {fmt.format(l_mean):<18} | {diff:+.1f}%")
        
    print("-" * 85)
    print("\nCONCLUSIONES PRELIMINARES:")
    if winners['vol_factor'].mean() > losers['vol_factor'].mean():
        print("1. El VOLUMEN es clave: Las ganadoras tienen más volumen explosivo.")
    else:
        print("1. El VOLUMEN no es todo: A veces demasiado volumen es clímax (agotamiento).")
        
    if winners['wick_pct'].mean() < losers['wick_pct'].mean():
        print("2. MECHAS CORTAS: Las ganadoras cierran cerca de máximos (menos rechazo).")
    else:
        print("2. MECHAS LARGAS: Las ganadoras sobrevivieron a la volatilidad inicial.")

if __name__ == "__main__":
    explain_winners()
