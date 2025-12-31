"""
Parallel Grid Search Optimizer for the Ninja Trading System

Uses multiprocessing to run multiple configurations simultaneously.
Each worker tests a different (threshold, stop) combination independently.

Usage:
    python scripts/grid_search_parallel.py --symbol BTCUSDT --days 3 --workers 8
"""
import os
import sys
import argparse
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import warnings

# Suppress warnings in worker processes
warnings.filterwarnings('ignore')

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Suppress verbose logging
import logging
logging.getLogger("ml_service_v2").setLevel(logging.ERROR)
logging.getLogger("EnsembleManager").setLevel(logging.ERROR)
logging.getLogger("BacktesterV2").setLevel(logging.ERROR)

# Database path (same as ml_service_v2.py)
DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"

def load_shared_data(symbol: str, days: int, hours: int = 0) -> pd.DataFrame:
    """Load data once to be shared across all workers."""
    start_ts = int((datetime.now() - timedelta(days=days, hours=hours)).timestamp() * 1000)
    db_symbol = symbol
    
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT 
        o.timestamp,
        o.mid_price as price,
        o.micro_price,
        o.bid_depth_20 as bid_depth, 
        o.ask_depth_20 as ask_depth, 
        o.spread_pct as bid_ask_spread, 
        o.obi_5,
        o.obi_10,
        o.obi_20 as obi,
        d.funding_rate, 
        d.open_interest,
        d.taker_buy_vol,
        d.taker_sell_vol
    FROM orderbook_metrics o
    JOIN derivatives_data d ON o.timestamp = d.timestamp AND o.symbol = d.symbol
    WHERE (o.symbol = '{db_symbol}' OR o.symbol = '{db_symbol.replace("USDT", "/USDT:USDT")}')
    AND o.timestamp > {start_ts}
    ORDER BY o.timestamp ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def run_single_config(args):
    """
    Run a single backtest configuration. 
    This function will be called by each worker process.
    """
    symbol, base_thr, stop_pct, shared_data_path = args
    
    # Import inside worker to avoid pickling issues
    from backtest_system_v2 import NinjaBotSimulator
    import pandas as pd
    
    # Load shared data from pickle (faster than re-querying DB)
    shared_data = pd.read_pickle(shared_data_path)
    
    # Create simulator and override config
    sim = NinjaBotSimulator(symbol=symbol, initial_capital=1000.0, leverage=10)
    sim.base_threshold = base_thr
    sim.hard_stop_pct = stop_pct
    
    # Run simulation
    sim.run(shared_data)
    
    # Calculate metrics
    num_trades = len(sim.trades)
    if num_trades > 0:
        wins = [t for t in sim.trades if t['pnl'] > 0]
        losses = [t for t in sim.trades if t['pnl'] <= 0]
        win_rate = len(wins) / num_trades
        profit_factor = sum(t['pnl'] for t in wins) / abs(sum(t['pnl'] for t in losses)) if losses else float('inf')
    else:
        win_rate = 0
        profit_factor = 0
    
    return_pct = ((sim.balance - 1000)/1000)*100
    
    return {
        'base_thr': base_thr,
        'stop_pct': stop_pct,
        'config': f"Thr:{base_thr:.2f} Stop:{stop_pct:.0%}",
        'final_balance': sim.balance,
        'return_pct': return_pct,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': num_trades
    }

def run_parallel_grid_search(symbol: str, days: int = 3, hours: int = 0, batch_size: int = 3):
    """
    Run grid search in batches to conserve memory.
    Each batch runs 'batch_size' configs simultaneously, then moves to the next batch.
    """
    # Define grid
    base_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    hard_stop_options = [-0.05, -0.10, -0.15]
    
    print(f"\n{'='*70}")
    print(f"🚀 BATCHED GRID SEARCH OPTIMIZER: {symbol}")
    print(f"   Período: {days} días, {hours} horas")
    print(f"   Configuraciones: {len(base_thresholds) * len(hard_stop_options)}")
    print(f"   Batch size: {batch_size} (configs simultáneas)")
    print(f"{'='*70}\n")
    
    # Load data ONCE
    print("📥 Cargando datos (una sola vez)...")
    shared_data = load_shared_data(symbol, days, hours)
    print(f"✅ Cargados {len(shared_data)} registros.\n")
    
    # Save to temp pickle for workers to read
    temp_pickle_path = "/tmp/grid_search_shared_data.pkl"
    shared_data.to_pickle(temp_pickle_path)
    
    # Build list of configs to test
    configs = []
    for base_thr in base_thresholds:
        for stop_pct in hard_stop_options:
            configs.append((symbol, base_thr, stop_pct, temp_pickle_path))
    
    # Split configs into batches
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    batches = list(chunks(configs, batch_size))
    total_batches = len(batches)
    
    print(f"⚡ Ejecutando {len(configs)} configuraciones en {total_batches} lotes de {batch_size}...")
    start_time = datetime.now()
    
    results = []
    for batch_num, batch in enumerate(batches, 1):
        print(f"\n--- Lote {batch_num}/{total_batches} ({len(batch)} configs) ---")
        
        # Run batch in parallel
        with Pool(processes=len(batch)) as pool:
            batch_results = pool.map(run_single_config, batch)
        
        results.extend(batch_results)
        
        # Print batch progress
        for r in batch_results:
            print(f"  ✅ {r['config']} -> Return: {r['return_pct']:>+6.2f}% | Trades: {r['total_trades']} | WR: {r['win_rate']:.0%}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ Completado en {elapsed:.1f} segundos ({elapsed/60:.1f} minutos).\n")
    
    # Clean up temp file
    os.remove(temp_pickle_path)
    
    # Ranking
    print("="*70)
    print("📊 RANKING DE CONFIGURACIONES (ORDENADO POR GANANCIA)")
    print("="*70)
    
    ranked_results = sorted(results, key=lambda x: x['return_pct'], reverse=True)
    
    print(f"{'Rank':<5} {'Config':<25} {'Return':<10} {'Trades':<8} {'WinRate':<10} {'PF':<8}")
    print("-" * 70)
    
    for i, r in enumerate(ranked_results, 1):
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "Inf"
        print(f"{i:<5} {r['config']:<25} {r['return_pct']:>+7.2f}% {r['total_trades']:<8} {r['win_rate']:.0%}{'':<6} {pf_str:<8}")
    
    # Best config
    best = ranked_results[0]
    print("\n" + "="*70)
    print("🏆 MEJOR CONFIGURACIÓN ENCONTRADA")
    print("="*70)
    print(f"   Threshold Base: {best['base_thr']:.2f}")
    print(f"   Hard Stop:      {best['stop_pct']:.0%} ROE")
    print(f"   Retorno:        {best['return_pct']:+.2f}%")
    print(f"   Win Rate:       {best['win_rate']:.0%}")
    print(f"   Profit Factor:  {best['profit_factor']:.2f}")
    print(f"   Total Trades:   {best['total_trades']}")
    print("="*70 + "\n")
    
    return ranked_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batched Grid Search Optimizer")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--hours", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=3, help="Configs to run simultaneously per batch (default: 3)")
    args = parser.parse_args()
    
    run_parallel_grid_search(args.symbol, days=args.days, hours=args.hours, batch_size=args.batch_size)
