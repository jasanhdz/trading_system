#!/usr/bin/env python3
"""
Grid Search for a SINGLE SYMBOL with multiple configurations.

Usage:
    python scripts/grid_search_single_symbol.py --symbol ETHUSDT --days 3
    
This script tests 15 configurations (5 thresholds × 3 stops) for ONE symbol
and saves results to a CSV file.
"""
import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Suppress verbose logging
import logging
logging.getLogger("ml_service_v2").setLevel(logging.WARNING)
logging.getLogger("EnsembleManager").setLevel(logging.WARNING)
logging.getLogger("BacktesterV2").setLevel(logging.WARNING)

from backtest_system_v2 import NinjaBotSimulator

# Grid parameters - 15 total configurations
BASE_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]
HARD_STOP_OPTIONS = [-0.05, -0.10, -0.15]

# Output directory
OUTPUT_DIR = REPO_ROOT / "scripts" / "grid_search_results"


def run_grid_search(symbol: str, days: int = 3):
    """Run 15 config grid search for a single symbol."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{symbol}_grid_results.csv"
    
    print(f"\n{'='*70}")
    print(f"🎯 GRID SEARCH: {symbol}")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Configurations: {len(BASE_THRESHOLDS)} × {len(HARD_STOP_OPTIONS)} = {len(BASE_THRESHOLDS) * len(HARD_STOP_OPTIONS)}")
    print(f"   Days: {days}")
    print(f"{'='*70}\n")
    
    results = []
    config_num = 0
    total_configs = len(BASE_THRESHOLDS) * len(HARD_STOP_OPTIONS)
    
    for base_thr in BASE_THRESHOLDS:
        for stop_pct in HARD_STOP_OPTIONS:
            config_num += 1
            config_name = f"Thr:{base_thr:.2f} Stop:{stop_pct:.0%}"
            
            print(f"[{config_num}/{total_configs}] {config_name}...", end=" ", flush=True)
            
            try:
                # Create simulator with this config
                sim = NinjaBotSimulator(symbol=symbol, initial_capital=1000.0, leverage=10)
                
                # Override config
                sim.base_threshold = base_thr
                sim.hard_stop_pct = stop_pct
                
                # Load data
                data = sim.load_data(days=days)
                
                if len(data) < 100:
                    print(f"⚠️ Not enough data ({len(data)} records)")
                    continue
                
                # Run simulation
                sim.run(data)
                
                # Calculate metrics
                num_trades = len(sim.trades)
                if num_trades > 0:
                    wins = [t for t in sim.trades if t['pnl'] > 0]
                    losses = [t for t in sim.trades if t['pnl'] <= 0]
                    win_rate = len(wins) / num_trades
                    total_wins = sum(t['pnl'] for t in wins)
                    total_losses = abs(sum(t['pnl'] for t in losses))
                    profit_factor = total_wins / total_losses if total_losses > 0 else 999.0
                else:
                    win_rate = 0
                    profit_factor = 0
                
                return_pct = ((sim.balance - 1000) / 1000) * 100
                
                results.append({
                    'symbol': symbol,
                    'base_threshold': base_thr,
                    'hard_stop_pct': stop_pct,
                    'return_pct': return_pct,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'total_trades': num_trades
                })
                
                print(f"Return: {return_pct:>+7.2f}% | Trades: {num_trades:>3} | WR: {win_rate:.0%} | PF: {profit_factor:.2f}")
                
                # Save incrementally
                pd.DataFrame(results).to_csv(output_file, index=False)
                
            except Exception as e:
                print(f"ERROR: {e}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"📊 RESULTADOS ORDENADOS POR RETORNO: {symbol}")
    print(f"{'='*70}")
    
    if results:
        df = pd.DataFrame(results)
        df_sorted = df.sort_values('return_pct', ascending=False)
        
        print(f"\n{'Rank':<5} {'Config':<20} {'Return':<10} {'Trades':<8} {'WR':<8} {'PF':<8}")
        print("-" * 60)
        
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            config = f"Thr:{row['base_threshold']:.2f} Stop:{row['hard_stop_pct']:.0%}"
            print(f"{i:<5} {config:<20} {row['return_pct']:>+7.2f}% {int(row['total_trades']):<8} {row['win_rate']:.0%}{'':<4} {row['profit_factor']:.2f}")
        
        # Best config
        best = df_sorted.iloc[0]
        print(f"\n🏆 MEJOR CONFIG PARA {symbol}:")
        print(f"   Threshold: {best['base_threshold']:.2f}")
        print(f"   Hard Stop: {best['hard_stop_pct']:.0%}")
        print(f"   Retorno:   {best['return_pct']:+.2f}%")
        print(f"   Win Rate:  {best['win_rate']:.0%}")
        print(f"   Trades:    {int(best['total_trades'])}")
    
    print(f"\n✅ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search for Single Symbol")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to test (e.g., ETHUSDT)")
    parser.add_argument("--days", type=int, default=3, help="Days of historical data")
    args = parser.parse_args()
    
    run_grid_search(args.symbol, days=args.days)
