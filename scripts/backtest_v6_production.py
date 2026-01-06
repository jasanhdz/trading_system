#!/usr/bin/env python3
"""
NINJA v6.0 FINAL: Production Backtest Engine (Fixes Applied)
Replica EXACTAMENTE la lógica del bot v5.1.1 (Guardian + Regimes).

FIXES v6.0:
1. ELIMINADO: MONK_BREAKEVEN_LOCK (Delegación total a UniversalProfitGuardian).
2. CORREGIDO: Typo neutralProb (Consistencia en todo el archivo).
3. CORREGIDO: Referencia a config_loader en Estrategias.

Uso:
    python scripts/backtest_v6_production.py --symbol DOGEUSDT --days 7 --mock
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

# Agregar rutas del proyecto
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# ═══════════════════════════════════════════════════════════════════════════
# 1. UNIVERSAL PROFIT GUARDIAN (Port of TS)
# ═════════════════════════════════════════════════════════════════════════════════════
class UniversalProfitGuardian:
    """Python port of UniversalProfitGuardian v5.1"""
    def __init__(self, config: Dict):
        self.config = config

    def evaluate(self, ctx: Dict) -> bool:
        """
        ctx: dict { peakRoe, currentRoe, volatilityFactor, marketBias, positionSide }
        Returns: TRUE if should lock profits (close).
        """
        # 1. ANTI-NOISE: Si peak below threshold, no actuar
        if ctx['peakRoe'] < self.config['peakThreshold']:
            return False

        # 2. CALCULAR DRAWDOWN (Fall from peak)
        drawdown = ctx['peakRoe'] - ctx['currentRoe']

        # 3. DYNAMIC LIMIT BASED ON VOLATILITY
        volMultiplier = 1.0
        if ctx['volatilityFactor'] >= 1.5:
            volMultiplier = 1.0 - (self.config['volatilitySensitivity'] * 0.5)
        elif ctx['volatilityFactor'] <= 0.8:
            volMultiplier = 1.0 + (self.config['volatilitySensitivity'] * 0.5)
        volMultiplier = max(0.5, min(1.5, volMultiplier))

        allowedDrawdown = self.config['baseDrawdown'] * volMultiplier

        # 4. CHECK IF EXCEEDS LIMIT
        if drawdown >= allowedDrawdown:
            # 5. ML TREND FILTER
            if self.config['enableTrendProtection']:
                biasFavorsMe = (
                    (ctx['positionSide'] == 'LONG' and ctx['marketBias'] == 'BULL') or
                    (ctx['positionSide'] == 'SHORT' and ctx['marketBias'] == 'BEAR')
                )
                if biasFavorsMe:
                    return False  # HOLD: Trend still alive
            return True  # CLOSE: Secure profit

        return False  # HOLD

    @staticmethod
    def WHALE_CONFIG():
        return {
            'peakThreshold': 0.015, 'baseDrawdown': 0.40,
            'volatilitySensitivity': 0.20, 'enableTrendProtection': True
        }

    @staticmethod
    def MONK_CONFIG():
        return {
            'peakThreshold': 0.01, 'baseDrawdown': 0.25,
            'volatilitySensitivity': 0.30, 'enableTrendProtection': True
        }

    @staticmethod
    def BLOODBATH_CONFIG():
        return {
            'peakThreshold': 0.005, 'baseDrawdown': 0.15,
            'volatilitySensitivity': 0.50, 'enableTrendProtection': False
        }

# ═════════════════════════════════════════════════════════════════════════════════════
# 2. CONFIG MANAGER (YAML Loader)
# ═════════════════════════════════════════════════════════════════════════════════════
class NinjaConfigLoader:
    def __init__(self, config_path: str = None):
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = REPO_ROOT / "binance-futures-bot-ts" / "regime_config.live.yaml"

        print(f"[Config] Loading from {self.config_path}")
        with open(self.config_path, 'r') as f:
            self.data = yaml.safe_load(f)

    def get_regime_config(self, regime: str, symbol: str = None) -> Dict:
        """Merge base config + symbol overrides."""
        regime_key = regime.upper()
        base = self.data['REGIMES'][regime_key]
        
        merged = base.copy()
        if symbol and self.data.get('SYMBOL_OVERRIDES'):
            sym_overrides = self.data['SYMBOL_OVERRIDES'].get(symbol, {})
            reg_overrides = sym_overrides.get(regime_key, {})
            if reg_overrides:
                merged.update(reg_overrides)
        
        return merged

# ═════════════════════════════════════════════════════════════════════════════════════
# 3. REGIME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════════════
class RegimeDetector:
    def __init__(self, config_loader):
        self.last_regime = 'BUNKER'
        self.regime_sticky_counter = 0
        self.config = config_loader.data['REGIME_DETECTOR']

    def get_hysteresis_threshold(self, regime: str) -> int:
        thresholds = {'BLOODBATH': 3, 'WHALE': 12, 'MONK': 6, 'BUNKER': 2}
        return thresholds.get(regime, 6)

    def analyze(self, snapshot: Dict) -> Dict:
        # 1. Volatility
        volatility = 'LOW'
        if snapshot['spreadPct'] > self.config['volatility_spread_high']:
            volatility = 'HIGH'
        elif snapshot['spreadPct'] > self.config['volatility_spread_low']:
            volatility = 'MED'

        # 2. Bias
        bias = 'NEUTRAL'
        diff = snapshot['longProb'] - snapshot['shortProb']
        if diff > self.config['bias_strength_threshold']:
            bias = 'BULL'
        elif diff < -self.config['bias_strength_threshold']:
            bias = 'BEAR'

        # 3. Raw Regime
        raw_regime = 'BUNKER'
        if volatility == 'HIGH' and snapshot['neutralProb'] > 0.50:
            raw_regime = 'BLOODBATH'
        elif volatility == 'MED' and bias != 'NEUTRAL':
            funding_aligned = (bias == 'BULL' and snapshot['fundingRate'] > 0) or \
                             (bias == 'BEAR' and snapshot['fundingRate'] < 0)
            obi_aligned = (bias == 'BULL' and snapshot['obi'] > 0.1) or \
                         (bias == 'BEAR' and snapshot['obi'] < -0.1)
            if funding_aligned or obi_aligned:
                raw_regime = 'WHALE'
        elif volatility == 'LOW' and bias != 'NEUTRAL':
            raw_regime = 'WHALE'
        elif volatility == 'LOW' and bias == 'NEUTRAL':
            raw_regime = 'MONK'

        # 4. Hysteresis
        sticky_threshold = self.get_hysteresis_threshold(self.last_regime)
        if raw_regime == self.last_regime:
            self.regime_sticky_counter = sticky_threshold
        else:
            self.regime_sticky_counter -= 1
        
        if self.regime_sticky_counter <= 0:
            self.last_regime = raw_regime
            self.regime_sticky_counter = self.get_hysteresis_threshold(raw_regime)
        else:
            raw_regime = self.last_regime

        return {
            'type': raw_regime,
            'bias': bias,
            'volatility': volatility,
            'confidence': 'HIGH' if max(snapshot['longProb'], snapshot['shortProb'], snapshot['neutralProb']) > 0.6 else 'LOW'
        }

# ═════════════════════════════════════════════════════════════════════════════════════
# 4. STRATEGIES
# ═════════════════════════════════════════════════════════════════════════════════════
class WhaleStrategy:
    def __init__(self, config_loader):
        self.name = 'WHALE'
        self.config_loader = config_loader
        self.guardian = UniversalProfitGuardian(UniversalProfitGuardian.WHALE_CONFIG())

    def get_config(self, symbol):
        return self.config_loader.get_regime_config('WHALE', symbol)

    def evaluate_exit(self, ctx: Dict, symbol=None):
        config = self.get_config(symbol)
        if ctx['currentRoe'] < config['hard_stop_roe']:
            return 'WHALE_HARD_STOP'
        if ctx['opposingProb'] > 0.80:
            return 'WHALE_PANIC_EXTREME'
        if self.guardian.evaluate(ctx):
            return 'WHALE_DYNAMIC_LOCK'
        return None

class MonkStrategy:
    def __init__(self, config_loader):
        self.name = 'MONK'
        self.config_loader = config_loader
        self.guardian = UniversalProfitGuardian(UniversalProfitGuardian.MONK_CONFIG())

    def get_config(self, symbol):
        return self.config_loader.get_regime_config('MONK', symbol)

    def evaluate_exit(self, ctx: Dict, symbol=None):
        config = self.get_config(symbol)
        if ctx['currentRoe'] < config['hard_stop_roe']:
            return 'MONK_HARD_STOP'
        if ctx['currentRoe'] >= config['tp_roe']:
            return 'MONK_RANGE_TP'
        if self.guardian.evaluate(ctx):
            return 'MONK_DYNAMIC_LOCK'
        # ❌ ELIMINADO v6.0: MONK_BREAKEVEN_LOCK
        return None

class BloodbathStrategy:
    def __init__(self, config_loader):
        self.name = 'BLOODBATH'
        self.config_loader = config_loader
        self.guardian = UniversalProfitGuardian(UniversalProfitGuardian.BLOODBATH_CONFIG())

    def get_config(self, symbol):
        return self.config_loader.get_regime_config('BLOODBATH', symbol)

    def evaluate_exit(self, ctx: Dict, symbol=None):
        config = self.get_config(symbol)
        if ctx['currentRoe'] < config['hard_stop_roe']:
            return 'BLOODBATH_HARD_STOP'
        if ctx['currentRoe'] >= config['tp_roe']:
            return 'BLOODBATH_MICRO_TP'
        if ctx['opposingProb'] > 0.55:
            return 'BLOODBATH_PANIC_FAST'
        if ctx['currentRoe'] > config['tp_roe'] * 0.5 and ctx['neutralProb'] > 0.50:
            return 'BLOODBATH_NEUTRAL_EXIT'
        return None

class BunkerStrategy:
    def __init__(self, config_loader):
        self.name = 'BUNKER'
        self.config_loader = config_loader

    def get_config(self, symbol):
        return self.config_loader.get_regime_config('BUNKER', symbol)

    def evaluate_exit(self, ctx: Dict, symbol=None):
        if ctx['currentRoe'] < -0.05:
            return 'BUNKER_STOP_LOSS'
        if ctx['opposingProb'] > 0.60:
            return 'BUNKER_PANIC_EXIT'
        if ctx['peakRoe'] > 0.01 and ctx['currentRoe'] < ctx['peakRoe'] * 0.5:
            return 'BUNKER_TRAILING_EXIT'
        if ctx['currentRoe'] > 0.005 and ctx['neutralProb'] > 0.55:
            return 'BUNKER_NEUTRAL_SECURE'
        return None

# ═════════════════════════════════════════════════════════════════════════════════════
# 5. MAIN BACKTEST CLASS
# ═════════════════════════════════════════════════════════════════════════════════════
class NinjaBacktester:
    def __init__(self, config_path=None, use_mock_signals=False):
        self.config_loader = NinjaConfigLoader(config_path)
        self.detector = RegimeDetector(self.config_loader)
        self.use_mock_signals = use_mock_signals
        
        self.strategies = {
            'WHALE': WhaleStrategy(self.config_loader),
            'MONK': MonkStrategy(self.config_loader),
            'BLOODBATH': BloodbathStrategy(self.config_loader),
            'BUNKER': BunkerStrategy(self.config_loader)
        }

    def load_data(self, symbol: str, days: int) -> pd.DataFrame:
        DB_PATH = REPO_ROOT / "data" / "market_data_v2.db"
        start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        conn = sqlite3.connect(DB_PATH)
        query = f"""
        SELECT 
            o.timestamp, o.mid_price as price, o.spread_pct, o.obi_20 as obi,
            d.funding_rate as fundingRate, d.taker_buy_vol, d.taker_sell_vol
        FROM orderbook_metrics o
        JOIN derivatives_data d ON o.timestamp = d.timestamp AND o.symbol = d.symbol
        WHERE (o.symbol = '{symbol}' OR o.symbol = '{symbol.replace("USDT", "/USDT:USDT")}')
        AND o.timestamp > {start_ts}
        ORDER BY o.timestamp ASC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def generate_predictions(self, df: pd.DataFrame):
        """Genera predicciones Mock (RSI)."""
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        df['longProb'] = rsi / 100.0
        df['shortProb'] = (100 - rsi) / 100.0
        df['neutralProb'] = 0.33
        df['fundingRate'] = df['fundingRate'].fillna(0)
        df['obi'] = df['obi'].fillna(0)
        df['spreadPct'] = df['spread_pct'].fillna(0.0004)
        return df.dropna()

    def run(self, symbol: str, days: int, initial_capital: float = 1000.0):
        print(f"\n{'='*60}")
        print(f"BACKTEST V6.0: {symbol} ({days} Days)")
        print(f"{'='*60}\n")

        df = self.load_data(symbol, days)
        if df.empty:
            print(f"❌ No data for {symbol}")
            return
        
        df = self.generate_predictions(df)
        df.set_index('timestamp', inplace=True)

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

        for i in range(1, len(df)):
            row = df.iloc[i]
            timestamp = row.name
            price = row['price']

            # Context
            ctx = {
                'currentRoe': 0.0, 'peakRoe': 0.0, 'holdTimeMs': 0,
                'opposingProb': 0.0, 'neutralProb': row['neutralProb'],
                'volatilityFactor': row['spreadPct'] / 0.0004,
                'marketBias': 'NEUTRAL', 'positionSide': position
            }

            if position:
                roi = (price - entry_price) / entry_price * (1 if position == 'LONG' else -1)
                if roi > peak_roe:
                    peak_roe = roi
                ctx['currentRoe'] = roi
                ctx['peakRoe'] = peak_roe
                ctx['holdTimeMs'] = timestamp - entry_time
                ctx['marketBias'] = 'BULL' if row['longProb'] > row['shortProb'] else ('BEAR' if row['shortProb'] > row['longProb'] else 'NEUTRAL')
                ctx['opposingProb'] = row['shortProb'] if position == 'LONG' else row['longProb']

            # Exit Logic
            exit_reason = None
            if position:
                regime_data = self.detector.analyze(row)
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

            # Entry Logic
            if not position and (timestamp - last_exit_time) > 30 * 60 * 1000:
                regime_data = self.detector.analyze(row)
                strategy = self.strategies[regime_data['type']]
                config = strategy.get_config(symbol)
                
                if config['leverage'] == 0:
                    continue

                thr = config['entry_threshold']
                if row['longProb'] > thr:
                    position = 'LONG'
                elif row['shortProb'] > thr:
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
            
            print(f"Capital Final: ${balance:.2f}")
            print(f"Retorno: {((balance - initial_capital)/initial_capital)*100:+.2f}%")
            print(f"Trades: {len(trades)} | WR: {len(wins)/len(trades)*100:.1f}%")
            print(f"\nExit Reasons:\n{df_trades['reason'].value_counts()}")
            
            safe_sym = symbol.replace('/', '_').replace(':', '_')
            df_trades.to_csv(REPO_ROOT / f"backtest_v6_{safe_sym}.csv", index=False)
            
            plt.figure(figsize=(12, 5))
            plt.plot(equity_curve)
            plt.title(f"Backtest V6.0 - {symbol}")
            plt.xlabel("Ticks")
            plt.ylabel("Balance ($)")
            plt.savefig(REPO_ROOT / f"backtest_v6_{safe_sym}.png")
            plt.close()
            print(f"💾 Saved: backtest_v6_{safe_sym}.csv/.png")
        else:
            print("No trades executed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    config_path = REPO_ROOT / "binance-futures-bot-ts" / "regime_config.live.yaml"
    backtester = NinjaBacktester(config_path=str(config_path), use_mock_signals=args.mock)
    
    PROD_SYMBOLS = ["DOGEUSDT", "LINKUSDT", "AVAXUSDT", "POLUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT", "BTCUSDT"]
    
    if args.symbol == "ALL":
        print("🚀 Running backtest for all production symbols...")
        for sym in PROD_SYMBOLS:
            try:
                backtester.run(sym, args.days)
            except Exception as e:
                print(f"❌ {sym}: {e}")
    else:
        backtester.run(args.symbol, args.days)
