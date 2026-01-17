#!/usr/bin/env python3
"""
Backtest Rápido: Escuadrón Berzerker (Últimos 7 días)
Misión: Contar cuántos Home Runs (+3%) se habrían activado.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Agregar raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from services.ml_service_v2 import manager, candles_db

def run_squad_backtest():
    symbols = ['BTCUSDT', 'DOGEUSDT', 'SOLUSDT', 'XRPUSDT', 'AVAXUSDT', 'NEARUSDT', 'FETUSDT']
    results = []

    print(f"\n{'PAR':<10} | {'GATILLOS':<10} | {'WIN RATE':<10} | {'PNL EST.'}")
    print("-" * 50)

    for symbol in symbols:
        # Normalizar símbolo para DB
        db_symbol = symbol.replace("USDT", "/USDT")
        
        # Cargar últimos 7 días
        df = candles_db.get_ohlcv_data(db_symbol, "5m", limit=2016) # 7 días en 5m
        
        if df.empty: continue

        triggers = 0
        wins = 0
        total_pnl = 0

        # Simular tick por tick (desde la vela 20 para tener historial)
        for i in range(20, len(df) - 48):
            window = df.iloc[i-19 : i+1]
            score = manager.calculate_score(window, symbol)

            if score > 0.85:
                triggers += 1
                entry_price = df.iloc[i]['close']
                tp_price = entry_price * 1.03
                sl_price = entry_price * 0.985 # -1.5%

                # Mirar futuro (horizonte 4h / 48 velas)
                future = df.iloc[i+1 : i+49]
                for _, f_row in future.iterrows():
                    if f_row['low'] <= sl_price:
                        total_pnl -= 0.015
                        break
                    if f_row['high'] >= tp_price:
                        wins += 1
                        total_pnl += 0.03
                        break
        
        wr = (wins / triggers * 100) if triggers > 0 else 0
        print(f"{symbol:<10} | {triggers:<10} | {wr:>8.2f}% | {total_pnl:>+8.2%}")
        
        results.append({'symbol': symbol, 'triggers': triggers, 'pnl': total_pnl})

    print("-" * 50)
    print(f"ESTADO: Escuadrón listo para cazar. 🦅📈")

if __name__ == "__main__":
    run_squad_backtest()
