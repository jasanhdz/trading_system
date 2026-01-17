#!/usr/bin/env python3
"""
Script de Análisis "Berzerker"
Objetivo: Validar estadísticamente el patrón de "Ola" (2 vs 3 velas con volumen).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from data.storage.database_manager import db_manager
from utils.logger import setup_logger

logger = setup_logger("berzerker_analysis")

def calculate_features(df):
    """Calcular features necesarios para el análisis"""
    df = df.copy()
    
    # Velas verdes/rojas
    df['is_green'] = df['close'] > df['open']
    
    # Tamaño del cuerpo y mechas
    df['body_size'] = abs(df['close'] - df['open'])
    df['body_pct'] = df['body_size'] / df['open']
    
    # Volumen relativo (vs media de 20 periodos)
    df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
    df['vol_factor'] = df['volume'] / df['vol_ma_20']
    
    return df

def simulate_guardian_trade(df, entry_idx, direction='long', leverage=50, hard_stop_roe=-0.15, max_candles=60):
    """
    Simular trade con lógica 'Universal Profit Guardian' (Berzerker Mode).
    """
    entry_price = df.iloc[entry_idx]['close']
    entry_time = df.index[entry_idx]
    
    peak_roe = 0.0
    
    # Mirar el futuro
    future_df = df.iloc[entry_idx+1 : entry_idx+1+max_candles]
    
    for i, row in future_df.iterrows():
        # Calcular ROE High y Low de la vela actual
        if direction == 'long':
            roe_high = (row['high'] - entry_price) / entry_price * leverage
            roe_low = (row['low'] - entry_price) / entry_price * leverage
            roe_close = (row['close'] - entry_price) / entry_price * leverage
        else:
            roe_high = (entry_price - row['low']) / entry_price * leverage # Short gana si baja
            roe_low = (entry_price - row['high']) / entry_price * leverage # Short pierde si sube
            roe_close = (entry_price - row['close']) / entry_price * leverage

        # 1. Chequear Hard Stop (Safety Net)
        # Asumimos que si el Low toca el stop, nos saca.
        if roe_low <= hard_stop_roe:
            return hard_stop_roe, i - entry_time # Loss por Hard Stop

        # 2. Actualizar Peak ROE
        if roe_high > peak_roe:
            peak_roe = roe_high

        # 3. Universal Profit Guardian Logic
        # Solo se activa si Peak ROE > 5%
        if peak_roe > 0.05:
            # Berzerker Mode: Linear 30% Drawdown
            # Allowed Drawdown = 30% of Peak
            # Exit Threshold = Peak - (Peak * 0.30) = Peak * 0.70
            exit_threshold = peak_roe * 0.70
            
            # Verificamos si en esta vela bajamos del umbral
            # Usamos roe_low para ser pesimistas (o roe_close para ser realistas)
            # Si roe_low < threshold, asumimos que nos sacó en el threshold (si el open estaba arriba)
            # Simplificación: Si roe_low cruza el threshold, salimos al threshold.
            if roe_low < exit_threshold:
                return exit_threshold, i - entry_time # Win (Trailing Stop Hit)
        
        # Si no salimos, seguimos a la siguiente vela
                
    # Si se acaba el tiempo, cerrar al cierre
    final_roe = roe_close
    return final_roe, future_df.index[-1] - entry_time

def analyze_pattern(df, consecutive_candles=2, min_vol_factor=1.5, min_body_pct=0.001):
    """
    Analizar patrón de N velas consecutivas verdes con volumen.
    """
    results = []
    
    # Iterar (optimizable con vectorización, pero bucle es más claro para lógica compleja)
    # Empezamos desde el índice necesario para tener historia
    for i in range(20, len(df) - 60):
        
        # Verificar patrón hacia atrás
        pattern_match = True
        
        # Verificar las N velas anteriores (incluyendo la actual i)
        for j in range(consecutive_candles):
            idx = i - j
            row = df.iloc[idx]
            
            # Condición: Vela verde
            if not row['is_green']:
                pattern_match = False
                break
            
            # Condición: Volumen fuerte
            if row['vol_factor'] < min_vol_factor:
                pattern_match = False
                break
                
            # Condición: Cuerpo mínimo (evitar dojis)
            if row['body_pct'] < min_body_pct:
                pattern_match = False
                break
        
        if pattern_match:
            # ENTRADA CON GUARDIAN
            pnl, duration = simulate_guardian_trade(df, i, direction='long', leverage=50)
            
            results.append({
                'entry_time': df.index[i],
                'entry_price': df.iloc[i]['close'],
                'pnl': pnl, # Esto ahora es ROE (con apalancamiento)
                'duration': duration,
                'vol_factor_entry': df.iloc[i]['vol_factor']
            })
            
    return pd.DataFrame(results)

import argparse

def run_analysis():
    parser = argparse.ArgumentParser(description='Analyze Berzerker Pattern')
    parser.add_argument('--symbol', type=str, default='XRP/USDT', help='Symbol to analyze')
    parser.add_argument('--days', type=int, default=0, help='Analyze only last N days')
    args = parser.parse_args()
    
    symbol = args.symbol
    timeframe = '5m' # Empezamos con 5m que es más fiable
    
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
        logger.error("No hay datos. Ejecuta primero el script de descarga.")
        return

    # Filter by days if requested
    if args.days > 0:
        cutoff_date = df.index.max() - timedelta(days=args.days)
        df = df[df.index >= cutoff_date]
        logger.info(f"Filtrando últimos {args.days} días (desde {cutoff_date})")

    logger.info(f"Datos cargados: {len(df)} velas.")
    logger.info(f"Symbol: {symbol} | First Close: {df.iloc[0]['close']} | Last Close: {df.iloc[-1]['close']}")
    
    # Calcular features
    df = calculate_features(df)
    
    # 1. Análisis de 2 Velas
    logger.info("\n--- ANÁLISIS PATRÓN 2 VELAS ---")
    results_2 = analyze_pattern(df, consecutive_candles=2, min_vol_factor=1.8)
    if not results_2.empty:
        wins = results_2[results_2['pnl'] > 0]
        losses = results_2[results_2['pnl'] <= 0]
        win_rate = len(wins) / len(results_2)
        avg_pnl = results_2['pnl'].mean()
        
        logger.info(f"Trades encontrados: {len(results_2)}")
        logger.info(f"✅ Wins: {len(wins)} | ❌ Losses: {len(losses)}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Avg PnL: {avg_pnl:.2%}")
        logger.info(f"Total Return (Simple): {results_2['pnl'].sum():.2%}")
        print(results_2[['entry_time', 'entry_price', 'pnl']])
    else:
        logger.info("No se encontraron patrones de 2 velas.")

    # 2. Análisis de 3 Velas
    logger.info("\n--- ANÁLISIS PATRÓN 3 VELAS ---")
    results_3 = analyze_pattern(df, consecutive_candles=3, min_vol_factor=1.8)
    if not results_3.empty:
        wins = results_3[results_3['pnl'] > 0]
        losses = results_3[results_3['pnl'] <= 0]
        win_rate = len(wins) / len(results_3)
        avg_pnl = results_3['pnl'].mean()
        
        logger.info(f"Trades encontrados: {len(results_3)}")
        logger.info(f"✅ Wins: {len(wins)} | ❌ Losses: {len(losses)}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Avg PnL: {avg_pnl:.2%}")
        logger.info(f"Total Return (Simple): {results_3['pnl'].sum():.2%}")
    else:
        logger.info("No se encontraron patrones de 3 velas.")

if __name__ == "__main__":
    run_analysis()
