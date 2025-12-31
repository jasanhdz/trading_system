"""
Multi-Symbol Sequential Grid Search (FIXED VERSION)

Uses the WORKING NinjaBotSimulator from backtest_system_v2.py
Runs grid search for ALL symbols sequentially (one at a time).
Saves results to a CSV file for later analysis.

Designed to run as a PM2 background process overnight.

Usage:
    pm2 start scripts/grid_search_all_symbols.py --name "grid-search" --interpreter python3
"""
import os
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
import csv

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

# Import the WORKING backtester
from backtest_system_v2 import NinjaBotSimulator

# All symbols to test
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT", 
    "SOLUSDT",
    "AVAXUSDT",
    "DOGEUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "LINKUSDT",
]

# Grid parameters
BASE_THRESHOLDS = [0.30, 0.35, 0.40]
HARD_STOP_OPTIONS = [-0.05, -0.10, -0.15]

# Output file
OUTPUT_FILE = REPO_ROOT / "scripts" / "grid_search_results_all_symbols.csv"


def run_grid_search_for_symbol(symbol: str, days: int = 3) -> list:
    """Run all grid configs for a single symbol using the REAL backtester."""
    results = []
    
    print(f"\n{'='*60}")
    print(f"🎯 SYMBOL: {symbol}")
    print(f"{'='*60}")
    
    config_num = 0
    total_configs = len(BASE_THRESHOLDS) * len(HARD_STOP_OPTIONS)
    
    for base_thr in BASE_THRESHOLDS:
        for stop_pct in HARD_STOP_OPTIONS:
            config_num += 1
            config_name = f"Thr:{base_thr:.2f} Stop:{stop_pct:.0%}"
            
            print(f"   [{config_num}/{total_configs}] {config_name}...", end=" ", flush=True)
            
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
                
                print(f"Return: {return_pct:>+6.2f}% | Trades: {num_trades} | WR: {win_rate:.0%}")
                
            except Exception as e:
                print(f"ERROR: {e}")
    
    return results


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("🚀 MULTI-SYMBOL GRID SEARCH OPTIMIZER (FIXED)")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Symbols: {len(SYMBOLS)}")
    print(f"   Configs per symbol: {len(BASE_THRESHOLDS) * len(HARD_STOP_OPTIONS)}")
    print("="*70)
    
    all_results = []
    
    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"\n📊 Progress: {i}/{len(SYMBOLS)} symbols")
        
        results = run_grid_search_for_symbol(symbol, days=3)
        all_results.extend(results)
        
        # Save incrementally
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"   💾 Saved to {OUTPUT_FILE}")
    
    # Final summary
    print("\n" + "="*70)
    print("🏆 FINAL SUMMARY")
    print("="*70)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Best config per symbol
        for symbol in df['symbol'].unique():
            sym_df = df[df['symbol'] == symbol]
            if len(sym_df) > 0 and sym_df['total_trades'].sum() > 0:
                best = sym_df.loc[sym_df['return_pct'].idxmax()]
                print(f"\n{symbol}:")
                print(f"   Best: Thr:{best['base_threshold']:.2f} Stop:{best['hard_stop_pct']:.0%}")
                print(f"   Return: {best['return_pct']:+.2f}% | Trades: {best['total_trades']:.0f} | WR: {best['win_rate']:.0%}")
            else:
                print(f"\n{symbol}: No trades generated")
    
    print(f"\n✅ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
