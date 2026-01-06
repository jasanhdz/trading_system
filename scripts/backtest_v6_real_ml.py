#!/usr/bin/env python3
"""
NINJA v6.0 REAL ML: Production Backtest Engine with Real Model Predictions
Replica EXACTAMENTE la lógica del bot v5.1.1 CON modelos ML entrenados.

Uso:
    source .venv_rocm62/bin/activate
    python scripts/backtest_v6_real_ml.py --symbol DOGEUSDT --days 7
    python scripts/backtest_v6_real_ml.py --symbol ALL --days 7
"""

import argparse
import sqlite3
import sys
import json
import os
import time as time_module
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# ═══════════════════════════════════════════════════════════════════════════
# ROCm Environment Setup (MUST be before torch import)
# ═══════════════════════════════════════════════════════════════════════════
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'  # Para RX 6600
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')  # Use first GPU

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import joblib

# Agregar rutas del proyecto
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Import real ML models
from ml.advanced_models.ensemble_manager import EnsembleManager

# ═══════════════════════════════════════════════════════════════════════════
# 1. UNIVERSAL PROFIT GUARDIAN (Port of TS)
# ═════════════════════════════════════════════════════════════════════════════════════
class UniversalProfitGuardian:
    def __init__(self, config: Dict):
        self.config = config

    def evaluate(self, ctx: Dict) -> bool:
        if ctx['peakRoe'] < self.config['peakThreshold']:
            return False
        drawdown = ctx['peakRoe'] - ctx['currentRoe']
        volMultiplier = 1.0
        if ctx['volatilityFactor'] >= 1.5:
            volMultiplier = 1.0 - (self.config['volatilitySensitivity'] * 0.5)
        elif ctx['volatilityFactor'] <= 0.8:
            volMultiplier = 1.0 + (self.config['volatilitySensitivity'] * 0.5)
        volMultiplier = max(0.5, min(1.5, volMultiplier))
        allowedDrawdown = self.config['baseDrawdown'] * volMultiplier
        if drawdown >= allowedDrawdown:
            if self.config['enableTrendProtection']:
                biasFavorsMe = (
                    (ctx['positionSide'] == 'LONG' and ctx['marketBias'] == 'BULL') or
                    (ctx['positionSide'] == 'SHORT' and ctx['marketBias'] == 'BEAR')
                )
                if biasFavorsMe:
                    return False
            return True
        return False

    @staticmethod
    def WHALE_CONFIG():
        return {'peakThreshold': 0.015, 'baseDrawdown': 0.40, 'volatilitySensitivity': 0.20, 'enableTrendProtection': True}
    @staticmethod
    def MONK_CONFIG():
        return {'peakThreshold': 0.01, 'baseDrawdown': 0.25, 'volatilitySensitivity': 0.30, 'enableTrendProtection': True}
    @staticmethod
    def BLOODBATH_CONFIG():
        return {'peakThreshold': 0.005, 'baseDrawdown': 0.15, 'volatilitySensitivity': 0.50, 'enableTrendProtection': False}

# ═══════════════════════════════════════════════════════════════════════════
# 2. REAL ML MODEL MANAGER (From ml_service_v2.py)
# ═══════════════════════════════════════════════════════════════════════════
MODELS_DIR = REPO_ROOT / "models" / "v2_ensemble"

class RealMLPredictor:
    def __init__(self):
        self.ensembles: Dict[str, EnsembleManager] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_cols: Dict[str, List[str]] = {}
        # Force CPU for backtest stability (GPU used by live service)
        self.device = "cpu"
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

    def predict(self, symbol: str, df: pd.DataFrame) -> Dict:
        clean_sym = self._clean_symbol(symbol)
        if clean_sym not in self.ensembles:
            if not self.load_model(symbol):
                return {'longProb': 0.33, 'shortProb': 0.33, 'neutralProb': 0.34}
        
        df = df.copy()
        df['buy_sell_ratio'] = df['taker_buy_vol'] / (df['taker_sell_vol'] + 1e-8)
        df['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'] + 1e-8)
        
        cols = self.feature_cols.get(clean_sym, [])
        is_v2_1 = len(cols) >= 19 or 'mean_obi_12' in cols
        
        if is_v2_1:
            window = 12
            df['mean_obi_12'] = df['obi'].rolling(window, min_periods=1).mean()
            df['max_obi_12'] = df['obi'].rolling(window, min_periods=1).max()
            df['std_obi_12'] = df['obi'].rolling(window, min_periods=2).std().fillna(0)
            df['total_volume'] = df['taker_buy_vol'] + df['taker_sell_vol']
            df['mean_volume_12'] = df['total_volume'].rolling(window, min_periods=1).mean()
            df['volume_trend'] = df['total_volume'] / (df['mean_volume_12'] + 1e-8)
            df['slope_price_12'] = (df['price'] - df['price'].shift(window).bfill()) / window
            df['cvd_12'] = (df['taker_buy_vol'] - df['taker_sell_vol']).rolling(window, min_periods=1).sum()
            df['cvd_norm_12'] = df['cvd_12'] / (df['mean_volume_12'] * window + 1e-8)
            df['std_price_12'] = df['price'].rolling(window, min_periods=2).std().fillna(0)
            df['volatility_ratio'] = df['std_price_12'] / (df['price'] + 1e-8)
        
        try:
            X = df[cols].values
            X = np.nan_to_num(X, nan=0.0)
        except KeyError as e:
            print(f"[ML] Missing columns for {clean_sym}: {e}")
            return {'longProb': 0.33, 'shortProb': 0.33, 'neutralProb': 0.34}
        
        scaler = self.scalers[clean_sym]
        X_scaled = scaler.transform(X)
        
        SEQ_LEN = 12
        if len(X_scaled) < SEQ_LEN:
            pad_len = SEQ_LEN - len(X_scaled)
            X_scaled = np.pad(X_scaled, ((pad_len, 0), (0, 0)), mode='edge')
        
        X_seq = X_scaled[-SEQ_LEN:]
        X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            result = self.ensembles[clean_sym].predict(X_tensor)
        
        probs = result['ensemble_probs'][0].tolist()
        return {
            'shortProb': float(probs[0]),
            'neutralProb': float(probs[1]),
            'longProb': float(probs[2])
        }

# ═══════════════════════════════════════════════════════════════════════════
# 3. CONFIG & REGIME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════
class NinjaConfigLoader:
    def __init__(self, config_path: str = None):
        self.config_path = Path(config_path) if config_path else REPO_ROOT / "binance-futures-bot-ts" / "regime_config.live.yaml"
        with open(self.config_path, 'r') as f:
            self.data = yaml.safe_load(f)

    def get_regime_config(self, regime: str, symbol: str = None) -> Dict:
        regime_key = regime.upper()
        base = self.data['REGIMES'][regime_key]
        merged = base.copy()
        if symbol and self.data.get('SYMBOL_OVERRIDES'):
            sym_overrides = self.data['SYMBOL_OVERRIDES'].get(symbol, {})
            reg_overrides = sym_overrides.get(regime_key, {})
            if reg_overrides:
                merged.update(reg_overrides)
        return merged

class RegimeDetector:
    def __init__(self, config_loader):
        self.last_regime = 'BUNKER'
        self.regime_sticky_counter = 0
        self.config = config_loader.data['REGIME_DETECTOR']

    def analyze(self, snapshot: Dict) -> Dict:
        volatility = 'LOW'
        if snapshot['spreadPct'] > self.config['volatility_spread_high']:
            volatility = 'HIGH'
        elif snapshot['spreadPct'] > self.config['volatility_spread_low']:
            volatility = 'MED'
        
        bias = 'NEUTRAL'
        diff = snapshot['longProb'] - snapshot['shortProb']
        if diff > self.config['bias_strength_threshold']:
            bias = 'BULL'
        elif diff < -self.config['bias_strength_threshold']:
            bias = 'BEAR'
        
        raw_regime = 'BUNKER'
        if volatility == 'HIGH' and snapshot['neutralProb'] > 0.50:
            raw_regime = 'BLOODBATH'
        elif volatility == 'MED' and bias != 'NEUTRAL':
            raw_regime = 'WHALE'
        elif volatility == 'LOW' and bias != 'NEUTRAL':
            raw_regime = 'WHALE'
        elif volatility == 'LOW' and bias == 'NEUTRAL':
            raw_regime = 'MONK'
        
        thresholds = {'BLOODBATH': 3, 'WHALE': 12, 'MONK': 6, 'BUNKER': 2}
        sticky_threshold = thresholds.get(self.last_regime, 6)
        if raw_regime == self.last_regime:
            self.regime_sticky_counter = sticky_threshold
        else:
            self.regime_sticky_counter -= 1
        
        if self.regime_sticky_counter <= 0:
            self.last_regime = raw_regime
            self.regime_sticky_counter = thresholds.get(raw_regime, 6)
        else:
            raw_regime = self.last_regime
        
        return {'type': raw_regime, 'bias': bias, 'volatility': volatility}

# ═══════════════════════════════════════════════════════════════════════════
# 4. STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════
class WhaleStrategy:
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.guardian = UniversalProfitGuardian(UniversalProfitGuardian.WHALE_CONFIG())
    def get_config(self, symbol): return self.config_loader.get_regime_config('WHALE', symbol)
    def evaluate_exit(self, ctx, symbol=None):
        config = self.get_config(symbol)
        if ctx['currentRoe'] < config['hard_stop_roe']: return 'WHALE_HARD_STOP'
        if ctx['opposingProb'] > 0.80: return 'WHALE_PANIC_EXTREME'
        if self.guardian.evaluate(ctx): return 'WHALE_DYNAMIC_LOCK'
        return None

class MonkStrategy:
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.guardian = UniversalProfitGuardian(UniversalProfitGuardian.MONK_CONFIG())
    def get_config(self, symbol): return self.config_loader.get_regime_config('MONK', symbol)
    def evaluate_exit(self, ctx, symbol=None):
        config = self.get_config(symbol)
        if ctx['currentRoe'] < config['hard_stop_roe']: return 'MONK_HARD_STOP'
        if ctx['currentRoe'] >= config['tp_roe']: return 'MONK_RANGE_TP'
        if self.guardian.evaluate(ctx): return 'MONK_DYNAMIC_LOCK'
        return None

class BloodbathStrategy:
    def __init__(self, config_loader):
        self.config_loader = config_loader
    def get_config(self, symbol): return self.config_loader.get_regime_config('BLOODBATH', symbol)
    def evaluate_exit(self, ctx, symbol=None):
        config = self.get_config(symbol)
        if ctx['currentRoe'] < config['hard_stop_roe']: return 'BLOODBATH_HARD_STOP'
        if ctx['currentRoe'] >= config['tp_roe']: return 'BLOODBATH_MICRO_TP'
        if ctx['opposingProb'] > 0.55: return 'BLOODBATH_PANIC_FAST'
        return None

class BunkerStrategy:
    def __init__(self, config_loader):
        self.config_loader = config_loader
    def get_config(self, symbol): return self.config_loader.get_regime_config('BUNKER', symbol)
    def evaluate_exit(self, ctx, symbol=None):
        if ctx['currentRoe'] < -0.05: return 'BUNKER_STOP_LOSS'
        if ctx['opposingProb'] > 0.60: return 'BUNKER_PANIC_EXIT'
        return None

# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
class NinjaBacktester:
    def __init__(self, config_path=None):
        self.config_loader = NinjaConfigLoader(config_path)
        self.detector = RegimeDetector(self.config_loader)
        self.ml_predictor = RealMLPredictor()
        self.strategies = {
            'WHALE': WhaleStrategy(self.config_loader),
            'MONK': MonkStrategy(self.config_loader),
            'BLOODBATH': BloodbathStrategy(self.config_loader),
            'BUNKER': BunkerStrategy(self.config_loader)
        }
        # Post-Exit Gate state (matches live bot behavior)
        self.post_exit_data = {}

    def evaluate_post_exit_gate(self, symbol: str, current_price: float, timestamp: int) -> bool:
        """
        Post-Exit Gate Logic - Aggressive re-entry on pullback/breakout.
        Matches TypeScript bot's evaluatePostExitGate behavior.
        """
        if symbol not in self.post_exit_data:
            return True  # No gate, allow entry
        
        gate = self.post_exit_data[symbol]
        if gate.get('ready', False):
            return True  # Gate already cleared
        
        # Gate parameters (match live bot)
        pullback_pct = 0.006    # 0.6% drop required
        rebound_pct = 0.35      # 35% rebound of the drop
        breakout_pct = 0.0015   # 0.15% above exit price
        timeout_ms = 300_000    # 5 minutes timeout
        
        exit_price = gate.get('exit_price', current_price)
        exit_time = gate.get('exit_time', 0)
        
        # Timeout check
        if timestamp - exit_time > timeout_ms:
            gate['ready'] = True
            gate['reason'] = 'timeout'
            return True
        
        # Track min/max since exit
        gate['min_price'] = min(gate.get('min_price', exit_price), current_price)
        gate['max_price'] = max(gate.get('max_price', exit_price), current_price)
        
        # PULLBACK: Price dropped, then rebounded
        drop = exit_price - gate['min_price']
        if drop > 0:
            drop_threshold = exit_price * pullback_pct
            if drop >= drop_threshold:
                rebound_target = gate['min_price'] + (drop * rebound_pct)
                if current_price >= rebound_target:
                    gate['ready'] = True
                    gate['reason'] = 'pullback'
                    return True
        
        # BREAKOUT: Price broke above exit price
        breakout_target = exit_price * (1 + breakout_pct)
        if current_price >= breakout_target:
            gate['ready'] = True
            gate['reason'] = 'breakout'
            return True
        
        return False  # Still waiting

    def load_data(self, symbol: str, days: int) -> pd.DataFrame:
        DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
        start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        conn = sqlite3.connect(DB_PATH)
        # Full query with ALL columns needed for ML features
        query = f"""
        SELECT o.timestamp, o.mid_price as price, o.micro_price,
               o.spread_pct, o.spread_pct as bid_ask_spread,
               o.obi_5, o.obi_10, o.obi_20 as obi,
               o.bid_depth_20 as bid_depth, o.ask_depth_20 as ask_depth,
               d.funding_rate, d.open_interest,
               d.taker_buy_vol, d.taker_sell_vol
        FROM orderbook_metrics o
        JOIN derivatives_data d ON o.timestamp = d.timestamp AND o.symbol = d.symbol
        WHERE (o.symbol = '{symbol}' OR o.symbol = '{symbol.replace("USDT", "/USDT:USDT")}')
        AND o.timestamp > {start_ts}
        ORDER BY o.timestamp ASC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        # Fill NaN values
        df = df.fillna(0)
        return df

    def run(self, symbol: str, days: int, initial_capital: float = 1000.0):
        print(f"\n{'='*60}")
        print(f"BACKTEST V6.0 REAL ML: {symbol} ({days} Days)")
        print(f"{'='*60}")

        df = self.load_data(symbol, days)
        if df.empty:
            print(f"❌ No data for {symbol}")
            return None
        
        print(f"[Data] Loaded {len(df)} rows")
        
        if not self.ml_predictor.load_model(symbol):
            print(f"❌ No ML model for {symbol}")
            return None

        # Pre-compute predictions in batches for speed
        print(f"[ML] Computing predictions...")
        predictions = []
        SEQ_LEN = 12
        step = 10  # Predict every 10 ticks to speed up
        
        for i in range(SEQ_LEN, len(df), step):
            window_df = df.iloc[max(0, i-60):i+1].copy()
            pred = self.ml_predictor.predict(symbol, window_df)
            predictions.append((i, pred))
        
        print(f"[ML] Generated {len(predictions)} predictions")
        
        # Create prediction lookup
        pred_lookup = {}
        last_pred = {'longProb': 0.33, 'shortProb': 0.33, 'neutralProb': 0.34}
        pred_idx = 0
        
        for i in range(len(df)):
            if pred_idx < len(predictions) and i >= predictions[pred_idx][0]:
                last_pred = predictions[pred_idx][1]
                pred_idx += 1
            pred_lookup[i] = last_pred

        # Simulation
        balance = initial_capital
        position = None
        entry_price = 0
        entry_time = None
        qty = 0
        peak_roe = 0
        trades = []
        equity_curve = []
        last_exit_time = 0
        commission_rate = 0.0004

        for i in range(SEQ_LEN, len(df)):
            row = df.iloc[i]
            timestamp = row['timestamp']
            price = row['price']
            preds = pred_lookup[i]
            
            snapshot = {
                'longProb': preds['longProb'],
                'shortProb': preds['shortProb'],
                'neutralProb': preds['neutralProb'],
                'spreadPct': row['spread_pct'] if pd.notna(row['spread_pct']) else 0.0004,
                'fundingRate': row['funding_rate'] if pd.notna(row['funding_rate']) else 0,
                'obi': row['obi'] if pd.notna(row['obi']) else 0
            }

            ctx = {
                'currentRoe': 0.0, 'peakRoe': 0.0, 'holdTimeMs': 0,
                'opposingProb': 0.0, 'neutralProb': snapshot['neutralProb'],
                'volatilityFactor': snapshot['spreadPct'] / 0.0004,
                'marketBias': 'NEUTRAL', 'positionSide': position
            }

            if position:
                roi = (price - entry_price) / entry_price * (1 if position == 'LONG' else -1)
                if roi > peak_roe:
                    peak_roe = roi
                ctx['currentRoe'] = roi
                ctx['peakRoe'] = peak_roe
                ctx['holdTimeMs'] = timestamp - entry_time
                ctx['marketBias'] = 'BULL' if snapshot['longProb'] > snapshot['shortProb'] else ('BEAR' if snapshot['shortProb'] > snapshot['longProb'] else 'NEUTRAL')
                ctx['opposingProb'] = snapshot['shortProb'] if position == 'LONG' else snapshot['longProb']

            exit_reason = None
            if position:
                regime_data = self.detector.analyze(snapshot)
                strategy = self.strategies[regime_data['type']]
                reason = strategy.evaluate_exit(ctx, symbol)
                if reason:
                    exit_reason = reason

            if position and exit_reason:
                pnl = (price - entry_price) * qty * (1 if position == 'LONG' else -1)
                fee = abs(price * qty) * commission_rate * 2
                balance += pnl - fee
                trades.append({
                    'entry_time': entry_time, 'exit_time': timestamp,
                    'side': position, 'entry_price': entry_price, 'exit_price': price,
                    'roi_pct': ctx['currentRoe'] * 100, 'pnl': pnl - fee, 'reason': exit_reason
                })
                position = None
                peak_roe = 0
                last_exit_time = timestamp

            # v6.0 ORIGINAL: Fixed 30-minute cooldown (proven best results)
            if not position and (timestamp - last_exit_time) > 30 * 60 * 1000:
                regime_data = self.detector.analyze(snapshot)
                strategy = self.strategies[regime_data['type']]
                config = strategy.get_config(symbol)
                
                if config['leverage'] == 0:
                    continue

                thr = config['entry_threshold']
                if snapshot['longProb'] > thr:
                    position = 'LONG'
                elif snapshot['shortProb'] > thr:
                    position = 'SHORT'
                
                if position:
                    entry_price = price
                    entry_time = timestamp
                    qty = (balance * config['leverage']) / price
                    fee = (price * qty) * commission_rate
                    balance -= fee

            equity_curve.append(balance)

        # Report
        if trades:
            df_trades = pd.DataFrame(trades)
            wins = df_trades[df_trades['pnl'] > 0]
            total_pnl = df_trades['pnl'].sum()
            
            print(f"\n📊 RESULTS:")
            print(f"Capital Final: ${balance:.2f}")
            print(f"Retorno: {((balance - initial_capital)/initial_capital)*100:+.2f}%")
            print(f"Trades: {len(trades)} | WR: {len(wins)/len(trades)*100:.1f}%")
            print(f"Total PnL: ${total_pnl:.2f}")
            print(f"\nExit Reasons:\n{df_trades['reason'].value_counts()}")
            
            safe_sym = symbol.replace('/', '_').replace(':', '_')
            df_trades.to_csv(REPO_ROOT / f"backtest_v6_ml_{safe_sym}.csv", index=False)
            
            plt.figure(figsize=(12, 5))
            plt.plot(equity_curve)
            plt.title(f"Backtest V6.0 REAL ML - {symbol}")
            plt.xlabel("Ticks")
            plt.ylabel("Balance ($)")
            plt.grid(True, alpha=0.3)
            plt.savefig(REPO_ROOT / f"backtest_v6_ml_{safe_sym}.png")
            plt.close()
            print(f"💾 Saved: backtest_v6_ml_{safe_sym}.csv/.png")
            
            return {'symbol': symbol, 'return_pct': ((balance - initial_capital)/initial_capital)*100, 'trades': len(trades), 'win_rate': len(wins)/len(trades)*100}
        else:
            print("No trades executed.")
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    config_path = REPO_ROOT / "binance-futures-bot-ts" / "regime_config.live.yaml"
    backtester = NinjaBacktester(config_path=str(config_path))
    
    PROD_SYMBOLS = ["DOGEUSDT", "LINKUSDT", "AVAXUSDT", "POLUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT", "BTCUSDT"]
    
    if args.symbol == "ALL":
        print("🚀 Running REAL ML backtest for all production symbols...")
        results = []
        for sym in PROD_SYMBOLS:
            try:
                result = backtester.run(sym, args.days)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ {sym}: {e}")
        
        if results:
            print("\n" + "="*60)
            print("📊 FLEET SUMMARY")
            print("="*60)
            df_results = pd.DataFrame(results)
            print(df_results.to_string(index=False))
            print(f"\nAvg Return: {df_results['return_pct'].mean():+.2f}%")
            print(f"Avg Win Rate: {df_results['win_rate'].mean():.1f}%")
    else:
        backtester.run(args.symbol, args.days)
