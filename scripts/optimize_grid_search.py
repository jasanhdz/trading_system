#!/usr/bin/env python3
"""
NINJA v6.0: Grid Search Optimizer
Finds optimal trading parameters by testing parameter combinations.

Usage:
    python scripts/optimize_grid_search.py --symbol AVAXUSDT --days 12
"""

import os
# ROCm Setup (must be before torch import)
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product
import pandas as pd
import numpy as np
import sqlite3
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import torch
from ml.advanced_models.ensemble_manager import EnsembleManager

# ═══════════════════════════════════════════════════════════════════════════
# GRID CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
GRID = {
    'leverage': [3, 5, 10],
    'entry_threshold': [0.40, 0.45, 0.50, 0.55],
    'hard_stop_roe': [-0.02, -0.03, -0.04, -0.05],
    'tp_roe': [0.015, 0.02, 0.025, 0.03]
}

# ═══════════════════════════════════════════════════════════════════════════
# ML PREDICTOR (Reused from backtest)
# ═══════════════════════════════════════════════════════════════════════════
import json
import joblib

MODELS_DIR = REPO_ROOT / "models" / "v2_ensemble"

class RealMLPredictor:
    def __init__(self):
        self.ensembles = {}
        self.scalers = {}
        self.feature_cols = {}
        self.device = "cpu"  # CPU for stability
        print(f"[ML] Device: {self.device}")

    def _clean_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").replace(":", "").replace("-", "").replace("USDT", "") + "USDT"

    def load_model(self, symbol: str) -> bool:
        clean_sym = self._clean_symbol(symbol)
        if clean_sym in self.ensembles:
            return True
        
        symbol_dir = MODELS_DIR / clean_sym
        if not symbol_dir.exists():
            print(f"[ML] No models for {clean_sym}")
            return False
        
        try:
            self.scalers[clean_sym] = joblib.load(symbol_dir / "scaler.pkl")
            with open(symbol_dir / "features.json", 'r') as f:
                self.feature_cols[clean_sym] = json.load(f)
            
            is_v2_1 = len(self.feature_cols[clean_sym]) >= 19
            version = "v2.1" if is_v2_1 else "default"
            
            ensemble = EnsembleManager(device=self.device)
            ensemble.load_weights_from_config(version)
            ensemble.load_model("tcn_v2", "tcn", str(symbol_dir / "tcn.pt"), str(symbol_dir / "tcn_config.json"))
            ensemble.load_model("xgb_v2", "xgboost", str(symbol_dir / "xgboost.joblib"), str(symbol_dir / "xgboost_config.json"))
            
            transformer_path = symbol_dir / "transformer.pt"
            if transformer_path.exists():
                ensemble.load_model("transformer_v2", "transformer", str(transformer_path), str(symbol_dir / "transformer_config.json"))
            
            self.ensembles[clean_sym] = ensemble
            print(f"[ML] ✅ Loaded {clean_sym} ({len(self.feature_cols[clean_sym])} features)")
            return True
        except Exception as e:
            print(f"[ML] ❌ Failed to load {clean_sym}: {e}")
            return False

    def predict(self, symbol: str, df: pd.DataFrame) -> dict:
        clean_sym = self._clean_symbol(symbol)
        if clean_sym not in self.ensembles:
            if not self.load_model(symbol):
                return {'longProb': 0.33, 'shortProb': 0.33, 'neutralProb': 0.34}
        
        df = df.copy()
        df['buy_sell_ratio'] = df['taker_buy_vol'] / (df['taker_sell_vol'] + 1e-8)
        df['net_taker_flow'] = df['taker_buy_vol'] - df['taker_sell_vol']
        df['depth_imbalance'] = (df['bid_depth_20'] - df['ask_depth_20']) / (df['bid_depth_20'] + df['ask_depth_20'] + 1e-8)
        df['micro_price_delta'] = df['micro_price'] - df['price']
        
        feature_cols = self.feature_cols[clean_sym]
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[feature_cols].values
        if len(X) < 12:
            return {'longProb': 0.33, 'shortProb': 0.33, 'neutralProb': 0.34}
        
        X_scaled = self.scalers[clean_sym].transform(X)
        X_seq = np.expand_dims(X_scaled[-12:], axis=0)
        
        # Convert to tensor for pytorch models
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        
        result = self.ensembles[clean_sym].predict(X_tensor)
        probs = result['ensemble_probs'][0].cpu().numpy()  # First batch item
        return {'longProb': float(probs[0]), 'shortProb': float(probs[1]), 'neutralProb': float(probs[2])}

# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════
class NinjaOptimizer:
    def __init__(self):
        self.ml_predictor = RealMLPredictor()
        self.df_cache = {}
        self.pred_cache = {}

    def load_data(self, symbol: str, days: int) -> pd.DataFrame:
        if symbol in self.df_cache:
            return self.df_cache[symbol]
            
        DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
        start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        conn = sqlite3.connect(DB_PATH)
        query = f"""
        SELECT o.timestamp, o.mid_price as price, o.spread_pct, o.obi_20 as obi,
               o.micro_price, o.obi_5, o.obi_10, o.spread_pct as bid_ask_spread,
               o.bid_depth_20, o.ask_depth_20, d.funding_rate, d.open_interest,
               d.taker_buy_vol, d.taker_sell_vol
        FROM orderbook_metrics o
        JOIN derivatives_data d ON o.timestamp = d.timestamp AND o.symbol = d.symbol
        WHERE (o.symbol = '{symbol}' OR o.symbol = '{symbol.replace("USDT", "/USDT:USDT")}')
        AND o.timestamp > {start_ts}
        ORDER BY o.timestamp ASC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        df = df.fillna(0)
        self.df_cache[symbol] = df
        return df

    def get_predictions(self, symbol: str, df: pd.DataFrame) -> dict:
        if symbol in self.pred_cache:
            return self.pred_cache[symbol]
            
        if not self.ml_predictor.load_model(symbol):
            return None
            
        SEQ_LEN = 12
        step = 20  # Faster for grid search
        predictions = []
        
        for i in range(SEQ_LEN, len(df), step):
            window_df = df.iloc[max(0, i-60):i+1].copy()
            pred = self.ml_predictor.predict(symbol, window_df)
            predictions.append((i, pred))
        
        # Build lookup
        pred_lookup = {}
        last_pred = {'longProb': 0.33, 'shortProb': 0.33, 'neutralProb': 0.34}
        pred_idx = 0
        for i in range(len(df)):
            if pred_idx < len(predictions) and i >= predictions[pred_idx][0]:
                last_pred = predictions[pred_idx][1]
                pred_idx += 1
            pred_lookup[i] = last_pred
        
        self.pred_cache[symbol] = pred_lookup
        return pred_lookup

    def simulate(self, symbol: str, df: pd.DataFrame, pred_lookup: dict,
                 lev: int, entry_th: float, sl: float, tp: float,
                 initial_capital: float = 1000.0) -> dict:
        """Fast simulation with specific parameters."""
        balance = initial_capital
        position = None
        entry_price = 0
        entry_time = 0
        qty = 0
        trades = []
        last_exit_time = 0
        SEQ_LEN = 12

        for i in range(SEQ_LEN, len(df)):
            row = df.iloc[i]
            timestamp = row['timestamp']
            price = row['price']
            preds = pred_lookup[i]
            
            # EXIT LOGIC
            if position:
                roi = (price - entry_price) / entry_price * (1 if position == 'LONG' else -1)
                
                # Hard Stop
                if roi < sl:
                    pnl = (price - entry_price) * qty * (1 if position == 'LONG' else -1)
                    balance += pnl - abs(price * qty) * 0.0008
                    trades.append({'roi': roi, 'pnl': pnl, 'reason': 'STOP'})
                    position = None
                    last_exit_time = timestamp
                    continue
                
                # Take Profit
                if roi >= tp:
                    pnl = (price - entry_price) * qty * (1 if position == 'LONG' else -1)
                    balance += pnl - abs(price * qty) * 0.0008
                    trades.append({'roi': roi, 'pnl': pnl, 'reason': 'TP'})
                    position = None
                    last_exit_time = timestamp
                    continue
            
            # ENTRY LOGIC
            if not position and (timestamp - last_exit_time) > 30 * 60 * 1000:
                long_p = preds['longProb']
                short_p = preds['shortProb']
                
                if long_p > entry_th:
                    position = 'LONG'
                    entry_price = price
                    entry_time = timestamp
                    qty = (balance * lev) / price
                    balance -= qty * price * 0.0004
                elif short_p > entry_th:
                    position = 'SHORT'
                    entry_price = price
                    entry_time = timestamp
                    qty = (balance * lev) / price
                    balance -= qty * price * 0.0004

        if not trades:
            return None
        
        df_t = pd.DataFrame(trades)
        total_pnl = balance - initial_capital
        net_return = (total_pnl / initial_capital) * 100
        num_trades = len(df_t)
        
        wins = df_t[df_t['pnl'] > 0]
        losses = df_t[df_t['pnl'] <= 0]
        win_rate = len(wins) / num_trades if num_trades > 0 else 0
        
        gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0.001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'net_return': net_return,
            'num_trades': num_trades,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'final_balance': balance
        }

    def optimize(self, symbol: str, days: int):
        print(f"\n{'='*60}")
        print(f"🔍 GRID SEARCH: {symbol} ({days} days)")
        print(f"{'='*60}")
        
        df = self.load_data(symbol, days)
        if df.empty:
            print(f"❌ No data for {symbol}")
            return None
        print(f"[Data] Loaded {len(df)} rows")
        
        pred_lookup = self.get_predictions(symbol, df)
        if pred_lookup is None:
            print(f"❌ No ML model for {symbol}")
            return None
        print(f"[ML] Predictions ready")
        
        # Generate all combinations
        combos = list(product(
            GRID['leverage'],
            GRID['entry_threshold'],
            GRID['hard_stop_roe'],
            GRID['tp_roe']
        ))
        print(f"[Grid] Testing {len(combos)} combinations...")
        
        results = []
        for i, (lev, entry_th, sl, tp) in enumerate(combos):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(combos)}")
            
            result = self.simulate(symbol, df, pred_lookup, lev, entry_th, sl, tp)
            if result and result['profit_factor'] > 0:
                result['params'] = {'leverage': lev, 'entry_threshold': entry_th, 
                                   'hard_stop_roe': sl, 'tp_roe': tp}
                results.append(result)
        
        if not results:
            print("❌ No profitable configurations found")
            return None
        
        # Sort by profit factor
        results.sort(key=lambda x: x['profit_factor'], reverse=True)
        
        # Show top 5
        print(f"\n🏆 TOP 5 CONFIGURATIONS:")
        print("-" * 60)
        for i, r in enumerate(results[:5]):
            p = r['params']
            print(f"{i+1}. Lev={p['leverage']} | Entry={p['entry_threshold']} | Stop={p['hard_stop_roe']} | TP={p['tp_roe']}")
            print(f"   Return: {r['net_return']:+.1f}% | WR: {r['win_rate']:.1f}% | PF: {r['profit_factor']:.2f} | Trades: {r['num_trades']}")
        
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
        
        print(f"\n💡 YAML Override:")
        print(f"  {symbol}:")
        print(f"    MONK: {{ leverage: {bp['leverage']}, entry_threshold: {bp['entry_threshold']}, hard_stop_roe: {bp['hard_stop_roe']}, tp_roe: {bp['tp_roe']} }}")
        
        return best

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--days", type=int, default=12)
    args = parser.parse_args()
    
    optimizer = NinjaOptimizer()
    optimizer.optimize(args.symbol, args.days)
