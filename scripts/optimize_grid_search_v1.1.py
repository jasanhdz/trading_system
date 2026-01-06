#!/usr/bin/env python3
"""
NINJA v6.0: Grid Search Optimizer v1.1 (Full Logic)
Uses the EXACT same logic as backtest_v6_real_ml.py

Usage:
    python scripts/optimize_grid_search_v1.1.py --symbol AVAXUSDT --days 12
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')

import argparse
import sys
from pathlib import Path
from itertools import product
import tempfile
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# Import the REAL backtest engine
from scripts.backtest_v6_real_ml import ProductionBacktester

# ═══════════════════════════════════════════════════════════════════════════
# GRID CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
GRID = {
    'leverage': [3, 5, 10],
    'entry_threshold': [0.40, 0.45, 0.50, 0.55],
    'hard_stop_roe': [-0.02, -0.03, -0.04, -0.05],
    'tp_roe': [0.015, 0.02, 0.025, 0.03]
}

class GridSearchV1_1:
    def __init__(self, base_config_path: str):
        self.base_config_path = Path(base_config_path)
        with open(self.base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def create_temp_config(self, symbol: str, lev: int, entry_th: float, sl: float, tp: float) -> str:
        """Create a temporary config file with the test parameters."""
        config = self.base_config.copy()
        
        # Override for all regimes for this symbol
        if 'SYMBOL_OVERRIDES' not in config:
            config['SYMBOL_OVERRIDES'] = {}
        
        config['SYMBOL_OVERRIDES'][symbol] = {
            'MONK': {'leverage': lev, 'entry_threshold': entry_th, 'hard_stop_roe': sl, 'tp_roe': tp},
            'WHALE': {'leverage': lev, 'entry_threshold': entry_th, 'hard_stop_roe': sl},
            'BLOODBATH': {'leverage': lev, 'entry_threshold': entry_th, 'hard_stop_roe': sl, 'tp_roe': tp}
        }
        
        # Write to temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config, temp_file)
        temp_file.close()
        return temp_file.name
    
    def run_backtest(self, symbol: str, days: int, config_path: str) -> dict:
        """Run backtest with specific config and return metrics."""
        try:
            backtester = ProductionBacktester(config_path)
            result = backtester.run(symbol, days)
            
            if not result or 'trades' not in result or len(result['trades']) == 0:
                return None
            
            trades = result['trades']
            net_return = result['net_return']
            
            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] <= 0]
            
            gross_profit = sum(t['pnl'] for t in wins) if wins else 0
            gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.001
            
            return {
                'net_return': net_return,
                'num_trades': len(trades),
                'win_rate': len(wins) / len(trades) * 100 if trades else 0,
                'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss
            }
        except Exception as e:
            print(f"  Error: {e}")
            return None
        finally:
            # Cleanup temp file
            if config_path != str(self.base_config_path):
                try:
                    os.unlink(config_path)
                except:
                    pass
    
    def optimize(self, symbol: str, days: int):
        print(f"\n{'='*60}")
        print(f"🔍 GRID SEARCH v1.1: {symbol} ({days} days)")
        print(f"   Using FULL backtest logic (Guardian + Regime)")
        print(f"{'='*60}")
        
        combos = list(product(
            GRID['leverage'],
            GRID['entry_threshold'],
            GRID['hard_stop_roe'],
            GRID['tp_roe']
        ))
        print(f"[Grid] Testing {len(combos)} combinations...")
        
        results = []
        for i, (lev, entry_th, sl, tp) in enumerate(combos):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(combos)}")
            
            config_path = self.create_temp_config(symbol, lev, entry_th, sl, tp)
            result = self.run_backtest(symbol, days, config_path)
            
            if result and result['profit_factor'] > 0:
                result['params'] = {
                    'leverage': lev,
                    'entry_threshold': entry_th,
                    'hard_stop_roe': sl,
                    'tp_roe': tp
                }
                results.append(result)
        
        if not results:
            print("❌ No profitable configurations found")
            return None
        
        # Sort by profit factor
        results.sort(key=lambda x: x['profit_factor'], reverse=True)
        
        # Filter only PF > 1.0 (actually profitable)
        profitable = [r for r in results if r['profit_factor'] >= 1.0]
        
        print(f"\n🏆 TOP 5 CONFIGURATIONS:")
        print("-" * 60)
        for i, r in enumerate(results[:5]):
            p = r['params']
            pf_icon = "✅" if r['profit_factor'] >= 1.0 else "⚠️"
            print(f"{i+1}. Lev={p['leverage']} | Entry={p['entry_threshold']} | Stop={p['hard_stop_roe']} | TP={p['tp_roe']}")
            print(f"   Return: {r['net_return']:+.1f}% | WR: {r['win_rate']:.1f}% | PF: {r['profit_factor']:.2f} {pf_icon} | Trades: {r['num_trades']}")
        
        best = results[0]
        bp = best['params']
        print(f"\n{'='*60}")
        print(f"✅ BEST CONFIG FOR {symbol}:")
        print(f"{'='*60}")
        print(f"  leverage: {bp['leverage']}")
        print(f"  entry_threshold: {bp['entry_threshold']}")
        print(f"  hard_stop_roe: {bp['hard_stop_roe']}")
        print(f"  tp_roe: {bp['tp_roe']}")
        print(f"\n  Expected: Return={best['net_return']:+.1f}% | WR={best['win_rate']:.1f}% | PF={best['profit_factor']:.2f}")
        
        if profitable:
            print(f"\n🎯 {len(profitable)} configurations with PF >= 1.0 (profitable)")
        else:
            print(f"\n⚠️ No configurations with PF >= 1.0 found")
        
        print(f"\n💡 YAML Override:")
        print(f"  {symbol}:")
        print(f"    MONK: {{ leverage: {bp['leverage']}, entry_threshold: {bp['entry_threshold']}, hard_stop_roe: {bp['hard_stop_roe']}, tp_roe: {bp['tp_roe']} }}")
        
        return best

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--days", type=int, default=12)
    args = parser.parse_args()
    
    config_path = REPO_ROOT / "binance-futures-bot-ts" / "regime_config.live.yaml"
    optimizer = GridSearchV1_1(str(config_path))
    optimizer.optimize(args.symbol, args.days)
