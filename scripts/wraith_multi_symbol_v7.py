#!/usr/bin/env python3
"""
Project Wraith V7: Hybrid Multi-Patrol
PRODUCTION SHADOW SERVICE - DUAL BRAIN ARCHITECTURE
- BTC/SOL: Wraith V6 (Break of Structure)
- ETH: Phantom V8 (CVD Liquidity Sweep)
"""
import time
import pandas as pd
import numpy as np
import torch
import requests
import os
import sys
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from threading import Lock

sys.path.append(str(Path(__file__).parent.parent))

from scripts.detect_distribution_tops import calculate_physics_features
from scripts.train_wraith_dqn import WraithNet
from scripts.train_phantom_dqn import PhantomNet

load_dotenv(Path(__file__).parent.parent / "binance-futures-bot-ts" / ".env")

# =====================================================
# HYBRID MULTI-PATROL CONFIG
# =====================================================
# Symbol Configuration
WRAITH_SYMBOLS = ['BTC/USDT', 'SOL/USDT']  # Use Wraith V6 brain
PHANTOM_SYMBOLS = ['ETH/USDT']              # Use Phantom V8 brain
ALL_SYMBOLS = WRAITH_SYMBOLS + PHANTOM_SYMBOLS

TIMEFRAME = '5m'
POLL_INTERVAL = 30

# Model Paths
WRAITH_MODEL_PATH = "models/wraith_dqn/wraith_net.pth"
PHANTOM_MODEL_PATH = "models/phantom_eth/phantom_net_best.pth"

# Trading Params (V7 - Adaptive per symbol)
INITIAL_BALANCE = 20.0
FEE_RATE = 0.0005

# Wraith Params (BTC/SOL)
WRAITH_LEVERAGE = 5
WRAITH_SL_PCT = 0.015
WRAITH_TP_PCT = 0.06
WRAITH_BE_ROE = 0.05
WRAITH_TRAILING = 0.010
WRAITH_CONFIDENCE = 0.85
WRAITH_RVOL = 2.0

# Phantom Params (ETH) - More conservative due to higher DD
PHANTOM_LEVERAGE = 3  # Reduced from 5 to manage DD
PHANTOM_SL_PCT = 0.015
PHANTOM_TP_PCT = 0.06
PHANTOM_BE_ROE = 0.10  # 10% ROE for BE
PHANTOM_TRAILING = 0.015
PHANTOM_CONFIDENCE = 0.55
PHANTOM_CVD_THRESHOLD = -5000  # Strong selling pressure required

# Equity Protection (Unified)
HOUSE_MONEY_MULTIPLIER = 2.0
HOUSE_MONEY_REDUCTION = 0.5
CIRCUIT_BREAKER_DD = 0.15
CIRCUIT_BREAKER_HOURS = 24
MAX_CONCURRENT_TRADES = 3  # One per symbol max

# Time Sentinel
FORBIDDEN_HOURS = [1, 4, 5, 10, 13, 18, 19, 23]
FORBIDDEN_DAYS = ['Tuesday']

# Paths
WALLET_FILE = "wraith_wallet_v7.json"
TRADES_FILE = "wraith_trades_v7.json"
STATE_FILE = "wraith_state_v7.json"

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

device = torch.device("cpu")
wallet_lock = Lock()

# =====================================================
# TIME SENTINEL
# =====================================================
class TimeSentinel:
    @staticmethod
    def is_forbidden(timestamp):
        hour = timestamp.hour
        day = timestamp.strftime('%A')
        return day in FORBIDDEN_DAYS or hour in FORBIDDEN_HOURS

# =====================================================
# SHARED WALLET
# =====================================================
class SharedWallet:
    def __init__(self):
        self.balance = INITIAL_BALANCE
        self.peak_balance = INITIAL_BALANCE
        self.load()
        
    def load(self):
        if os.path.exists(WALLET_FILE):
            with open(WALLET_FILE, 'r') as f:
                data = json.load(f)
                self.balance = data.get('balance', INITIAL_BALANCE)
                self.peak_balance = data.get('peak_balance', self.balance)
                
    def save(self):
        with open(WALLET_FILE, 'w') as f:
            json.dump({
                'balance': self.balance,
                'peak_balance': self.peak_balance,
                'updated_at': datetime.now().isoformat()
            }, f, indent=2)
            
    def update(self, pnl):
        with wallet_lock:
            self.balance += pnl
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            self.save()
        
    def get_drawdown(self):
        if self.peak_balance == 0:
            return 0
        return (self.peak_balance - self.balance) / self.peak_balance
    
    def get_available_margin(self, num_open_trades):
        if num_open_trades >= MAX_CONCURRENT_TRADES:
            return 0
        return self.balance / (MAX_CONCURRENT_TRADES - num_open_trades + 1)

# =====================================================
# STATE MANAGER
# =====================================================
class StateManager:
    def __init__(self):
        self.circuit_breaker_until = None
        self.house_money_active = False
        self.load()
        
    def load(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
                cb = data.get('circuit_breaker_until')
                if cb:
                    self.circuit_breaker_until = datetime.fromisoformat(cb)
                    
    def save(self):
        with open(STATE_FILE, 'w') as f:
            json.dump({
                'circuit_breaker_until': self.circuit_breaker_until.isoformat() if self.circuit_breaker_until else None,
                'house_money_active': self.house_money_active
            }, f, indent=2)
            
    def activate_circuit_breaker(self):
        self.circuit_breaker_until = datetime.utcnow() + timedelta(hours=CIRCUIT_BREAKER_HOURS)
        self.save()
        
    def is_circuit_breaker_active(self):
        if self.circuit_breaker_until is None:
            return False
        if datetime.utcnow() > self.circuit_breaker_until:
            self.circuit_breaker_until = None
            self.save()
            return False
        return True

# =====================================================
# HYBRID TRADE MANAGER
# =====================================================
class HybridTradeManager:
    def __init__(self, wallet, state):
        self.wallet = wallet
        self.state = state
        self.active_trades = {}
        self.load_trades()
        
    def load_trades(self):
        if os.path.exists(TRADES_FILE):
            with open(TRADES_FILE, 'r') as f:
                trades = json.load(f)
                for t in trades:
                    if t['status'] == 'OPEN':
                        self.active_trades[t['symbol']] = t
                        
    def save_trade(self, trade):
        trades = []
        if os.path.exists(TRADES_FILE):
            with open(TRADES_FILE, 'r') as f:
                trades = json.load(f)
        found = False
        for i, t in enumerate(trades):
            if t['id'] == trade['id']:
                trades[i] = trade
                found = True
                break
        if not found:
            trades.append(trade)
        with open(TRADES_FILE, 'w') as f:
            json.dump(trades, f, indent=2)
            
    def get_open_count(self):
        return len(self.active_trades)
    
    def has_open_trade(self, symbol):
        return symbol in self.active_trades

    def get_leverage(self, symbol):
        # Base leverage per symbol type
        if symbol in PHANTOM_SYMBOLS:
            base = PHANTOM_LEVERAGE
        else:
            base = WRAITH_LEVERAGE
            
        # House Money reduction
        if self.wallet.balance >= INITIAL_BALANCE * HOUSE_MONEY_MULTIPLIER:
            if not self.state.house_money_active:
                self.state.house_money_active = True
                self.state.save()
            return base * HOUSE_MONEY_REDUCTION
        return base
    
    def get_params(self, symbol):
        """Get trading params based on symbol type."""
        if symbol in PHANTOM_SYMBOLS:
            return {
                'sl_pct': PHANTOM_SL_PCT,
                'tp_pct': PHANTOM_TP_PCT,
                'be_roe': PHANTOM_BE_ROE,
                'trailing': PHANTOM_TRAILING
            }
        else:
            return {
                'sl_pct': WRAITH_SL_PCT,
                'tp_pct': WRAITH_TP_PCT,
                'be_roe': WRAITH_BE_ROE,
                'trailing': WRAITH_TRAILING
            }

    def open_short(self, symbol, price, confidence, timestamp, brain_type):
        if self.has_open_trade(symbol):
            return None
        if self.get_open_count() >= MAX_CONCURRENT_TRADES:
            return None
            
        leverage = self.get_leverage(symbol)
        margin = self.wallet.get_available_margin(self.get_open_count())
        if margin <= 0:
            return None
            
        params = self.get_params(symbol)
        size = margin * leverage
        qty = size / price
        
        trade = {
            'id': int(datetime.now().timestamp() * 1000),
            'symbol': symbol,
            'brain': brain_type,  # 'WRAITH' or 'PHANTOM'
            'type': 'SHORT',
            'status': 'OPEN',
            'entry_price': price,
            'quantity': qty,
            'margin': margin,
            'leverage': leverage,
            'sl_price': price * (1 + params['sl_pct']),
            'tp_price': price * (1 - params['tp_pct']),
            'be_price': price * (1 - 0.002),
            'be_roe': params['be_roe'],
            'trailing_dev': params['trailing'],
            'entry_time': timestamp.isoformat(),
            'confidence': confidence,
            'is_breakeven': False,
            'peak_price': price,
            'pnl': 0
        }
        self.active_trades[symbol] = trade
        self.save_trade(trade)
        return trade
        
    def update_trade(self, symbol, current_low, current_high, current_close, timestamp):
        if symbol not in self.active_trades:
            return None
            
        trade = self.active_trades[symbol]
        entry_price = trade['entry_price']
        leverage = trade['leverage']
        
        if current_low < trade['peak_price']:
            trade['peak_price'] = current_low
            
        current_roe = (entry_price - current_low) / entry_price * leverage
        
        # Break-even (symbol-specific threshold)
        if current_roe >= trade['be_roe'] and not trade['is_breakeven']:
            trade['sl_price'] = trade['be_price']
            trade['is_breakeven'] = True
            self.save_trade(trade)
            return {'event': 'BREAKEVEN', 'symbol': symbol, 'roe': current_roe}
            
        # Trailing Stop
        if trade['is_breakeven']:
            trailing_sl = trade['peak_price'] * (1 + trade['trailing_dev'])
            if current_high >= trailing_sl:
                return self.close_trade(symbol, trailing_sl, "TRAILING_GUARDIAN", timestamp)
                
        if current_high >= trade['sl_price']:
            return self.close_trade(symbol, trade['sl_price'], "STOP_LOSS", timestamp)
            
        if current_low <= trade['tp_price']:
            return self.close_trade(symbol, trade['tp_price'], "TAKE_PROFIT", timestamp)
            
        return None
        
    def close_trade(self, symbol, exit_price, reason, timestamp):
        if symbol not in self.active_trades:
            return None
            
        trade = self.active_trades[symbol]
        raw_pnl = (trade['entry_price'] - exit_price) * trade['quantity']
        entry_fee = (trade['entry_price'] * trade['quantity']) * FEE_RATE
        exit_fee = (exit_price * trade['quantity']) * FEE_RATE
        net_pnl = raw_pnl - entry_fee - exit_fee
        roi = (net_pnl / trade['margin']) * 100
        
        trade['status'] = 'CLOSED'
        trade['exit_price'] = exit_price
        trade['exit_time'] = timestamp.isoformat()
        trade['exit_reason'] = reason
        trade['pnl'] = net_pnl
        trade['roi'] = roi
        
        self.wallet.update(net_pnl)
        self.save_trade(trade)
        
        if self.wallet.get_drawdown() >= CIRCUIT_BREAKER_DD:
            self.state.activate_circuit_breaker()
            
        del self.active_trades[symbol]
        return trade

# =====================================================
# PHANTOM FEATURE CALCULATOR
# =====================================================
def calculate_phantom_features(df):
    """Calculate Phantom-specific features including CVD."""
    # CVD Proxy
    df['direction'] = np.where(df['close'] > df['open'], 1, -1)
    df['volume_delta'] = df['direction'] * df['volume']
    df['cvd_20'] = df['volume_delta'].rolling(20).sum()
    df['cvd_slope'] = df['cvd_20'].diff(5)
    df['cvd_z'] = (df['cvd_20'] - df['cvd_20'].rolling(50).mean()) / (df['cvd_20'].rolling(50).std() + 1e-8)
    
    # Volatility
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['volatility_z'] = (df['volatility'] - df['volatility'].expanding().mean()) / (df['volatility'].expanding().std() + 1e-8)
    
    # EMAs
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()
    df['dist_ema20'] = (df['close'] - df['ema_20']) / df['close']
    df['dist_ema200'] = (df['close'] - df['ema_200']) / df['close']
    
    # Momentum
    df['velocity'] = df['close'].diff()
    df['acceleration'] = df['velocity'].diff()
    df['velocity_sm'] = df['velocity'].ewm(span=5).mean()
    df['acceleration_sm'] = df['acceleration'].ewm(span=5).mean()
    
    # Fakeout
    df['body'] = abs(df['open'] - df['close'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['is_fakeout'] = df['upper_wick'] > (df['body'] * 1.5)
    
    # Volume
    df['vol_sma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_sma'] + 1e-8)
    
    # Staleness
    df['is_doji'] = df['body'] < (df['high'] - df['low']) * 0.1
    df['staleness'] = df['is_doji'].rolling(10).sum()
    
    return df

# =====================================================
# MAIN HYBRID SERVICE
# =====================================================
class HybridMultiPatrol:
    def __init__(self):
        self.wraith_model = None
        self.phantom_model = None
        self.wallet = SharedWallet()
        self.state = StateManager()
        self.trade_manager = HybridTradeManager(self.wallet, self.state)
        self.last_processed = {s: None for s in ALL_SYMBOLS}
        
        # Health Check Tracking
        self.candles_processed = {s: 0 for s in ALL_SYMBOLS}
        self.signals_detected = {'WRAITH': 0, 'PHANTOM': 0}
        self.last_health_check = datetime.utcnow()
        self.health_check_interval = timedelta(hours=12)
        self.service_start_time = datetime.utcnow()
        
        self.load_models()
        
    def load_models(self):
        print("🦅 Loading Hybrid V7 Brains...")
        
        # Wraith Brain (BTC/SOL)
        self.wraith_model = WraithNet(input_dim=6, output_dim=2).to(device)
        self.wraith_model.load_state_dict(torch.load(WRAITH_MODEL_PATH, map_location=device))
        self.wraith_model.eval()
        print("  ✅ Wraith V6 Brain loaded (BTC/SOL)")
        
        # Phantom Brain (ETH)
        self.phantom_model = PhantomNet(input_dim=12, output_dim=2).to(device)
        self.phantom_model.load_state_dict(torch.load(PHANTOM_MODEL_PATH, map_location=device))
        self.phantom_model.eval()
        print("  ✅ Phantom V8 Brain loaded (ETH)")

    def send_health_check(self):
        """Send 12-hour health check summary to Telegram."""
        now = datetime.utcnow()
        uptime = now - self.service_start_time
        uptime_hours = uptime.total_seconds() / 3600
        
        # Load trade history for stats
        total_trades = 0
        winning_trades = 0
        total_pnl = 0
        
        if os.path.exists(TRADES_FILE):
            with open(TRADES_FILE, 'r') as f:
                trades = json.load(f)
                for t in trades:
                    if t['status'] == 'CLOSED':
                        total_trades += 1
                        if t.get('pnl', 0) > 0:
                            winning_trades += 1
                        total_pnl += t.get('pnl', 0)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Build status emoji
        wraith_status = "✅" if self.wraith_model is not None else "❌"
        phantom_status = "✅" if self.phantom_model is not None else "❌"
        
        # Circuit breaker status
        cb_status = "🔒 ACTIVE" if self.state.is_circuit_breaker_active() else "✅ Ready"
        
        # House Money status
        hm_status = "🏠 Active" if self.state.house_money_active else "📊 Normal"
        
        message = (
            f"📊 **HYBRID V7 HEALTH CHECK** 📊\n\n"
            f"⏱️ Uptime: **{uptime_hours:.1f} hours**\n\n"
            f"**🧠 Brain Status:**\n"
            f"  {wraith_status} Wraith V6 (BTC/SOL)\n"
            f"  {phantom_status} Phantom V8 (ETH)\n\n"
            f"**📈 Processing Stats:**\n"
            f"  BTC: {self.candles_processed.get('BTC/USDT', 0)} candles\n"
            f"  SOL: {self.candles_processed.get('SOL/USDT', 0)} candles\n"
            f"  ETH: {self.candles_processed.get('ETH/USDT', 0)} candles\n\n"
            f"**🎯 Signals Detected:**\n"
            f"  Wraith: {self.signals_detected['WRAITH']}\n"
            f"  Phantom: {self.signals_detected['PHANTOM']}\n\n"
            f"**💰 Account Status:**\n"
            f"  Balance: **${self.wallet.balance:.2f}**\n"
            f"  Peak: ${self.wallet.peak_balance:.2f}\n"
            f"  DD: {self.wallet.get_drawdown()*100:.1f}%\n"
            f"  Total PnL: ${total_pnl:.2f}\n\n"
            f"**🛡️ Protections:**\n"
            f"  Circuit Breaker: {cb_status}\n"
            f"  House Money: {hm_status}\n\n"
            f"**📊 Performance:**\n"
            f"  Trades: {total_trades}\n"
            f"  Win Rate: {win_rate:.1f}%\n"
            f"  Open: {self.trade_manager.get_open_count()}"
        )
        
        self.send_telegram(message)
        self.last_health_check = now
        print(f"📊 Health check sent at {now.isoformat()}")

    def send_telegram(self, message):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, data=data, timeout=5)
        except:
            pass

    def fetch_candles(self, symbol, limit=250):
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {'symbol': symbol.replace('/', ''), 'interval': TIMEFRAME, 'limit': limit}
        try:
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            return df
        except Exception as e:
            print(f"Fetch error {symbol}: {e}")
            return pd.DataFrame()

    def check_wraith_bos(self, df, idx):
        """Wraith trigger: Break of Structure."""
        if idx < 20:
            return False
        current = df.iloc[idx]
        previous = df.iloc[idx-1]
        vol_avg = df['volume'].iloc[idx-20:idx].mean()
        return current['close'] < previous['low'] and current['volume'] > vol_avg * WRAITH_RVOL

    def check_phantom_cvd(self, df, idx):
        """Phantom trigger: CVD Liquidity Sweep."""
        if idx < 50:
            return False
        row = df.iloc[idx]
        
        # CVD slope must be strongly negative
        if pd.isna(row['cvd_slope']) or row['cvd_slope'] > PHANTOM_CVD_THRESHOLD:
            return False
        
        # Bearish candle
        if row['close'] >= row['open']:
            return False
        
        return True

    def get_wraith_state(self, row):
        """6-feature state for Wraith."""
        return np.array([
            row['dist_to_ema'] * 100,
            row['velocity_sm'] / row['close'] * 1000,
            row['acceleration_sm'] / row['close'] * 1000,
            row['volatility_z'],
            row['bb_dist'] * 100,
            (row['volume'] / (row['vol_sm'] + 1e-8)) - 1.0
        ], dtype=np.float32)

    def get_phantom_state(self, row):
        """12-feature state for Phantom."""
        return np.array([
            row['cvd_z'] if not pd.isna(row['cvd_z']) else 0,
            row['cvd_slope'] / 10000 if not pd.isna(row['cvd_slope']) else 0,
            0,  # Weakness score (would need BTC data)
            row['volatility_z'] if not pd.isna(row['volatility_z']) else 0,
            float(row['is_fakeout']) if not pd.isna(row['is_fakeout']) else 0,
            row['vol_ratio'] - 1.0 if not pd.isna(row['vol_ratio']) else 0,
            row['staleness'] / 10 if not pd.isna(row['staleness']) else 0,
            row['velocity_sm'] / row['close'] * 1000 if not pd.isna(row['velocity_sm']) else 0,
            row['acceleration_sm'] / row['close'] * 1000 if not pd.isna(row['acceleration_sm']) else 0,
            row['dist_ema20'] * 100 if not pd.isna(row['dist_ema20']) else 0,
            row['dist_ema200'] * 100 if not pd.isna(row['dist_ema200']) else 0,
            0  # Reserved
        ], dtype=np.float32)

    def process_wraith_symbol(self, symbol):
        """Process BTC or SOL with Wraith brain."""
        try:
            df = self.fetch_candles(symbol)
            if df.empty:
                return
            
            current_candle = df.iloc[-1]
            last_closed = df.iloc[-2]
            current_time = last_closed['timestamp']
            
            # Trade Management
            if self.trade_manager.has_open_trade(symbol):
                result = self.trade_manager.update_trade(
                    symbol,
                    current_candle['low'],
                    current_candle['high'],
                    current_candle['close'],
                    current_candle['timestamp']
                )
                self._handle_trade_result(symbol, result)
            
            # Signal Detection
            if not self.trade_manager.has_open_trade(symbol) and self.last_processed[symbol] != current_time:
                if TimeSentinel.is_forbidden(current_time):
                    self.last_processed[symbol] = current_time
                    return
                
                if self.trade_manager.get_open_count() >= MAX_CONCURRENT_TRADES:
                    self.last_processed[symbol] = current_time
                    return
                
                df = calculate_physics_features(df)
                row = df.iloc[-2]
                
                # Physics Check
                near_ceiling = -0.01 < row['dist_to_ema'] < 0.005
                is_decelerating = row['acceleration_sm'] < 0
                vol_drying = row['volume'] < row['vol_sm']
                is_compressed = row['volatility_z'] < -0.5
                is_red = row['close'] < row['open']
                
                is_candidate = near_ceiling and is_decelerating and (vol_drying or is_compressed) and is_red
                
                if is_candidate and self.check_wraith_bos(df, len(df)-2):
                    state = self.get_wraith_state(row)
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_values = self.wraith_model(state_t)
                        action = torch.argmax(q_values).item()
                        confidence = torch.softmax(q_values, dim=1)[0][1].item()
                        
                    if action == 1 and confidence > WRAITH_CONFIDENCE:
                        trade = self.trade_manager.open_short(symbol, row['close'], confidence, current_time, 'WRAITH')
                        if trade:
                            self._send_entry_alert(trade, 'WRAITH')
                
                self.last_processed[symbol] = current_time
                
        except Exception as e:
            print(f"Wraith error {symbol}: {e}")

    def process_phantom_symbol(self, symbol):
        """Process ETH with Phantom brain."""
        try:
            df = self.fetch_candles(symbol)
            if df.empty:
                return
            
            current_candle = df.iloc[-1]
            last_closed = df.iloc[-2]
            current_time = last_closed['timestamp']
            
            # Trade Management
            if self.trade_manager.has_open_trade(symbol):
                result = self.trade_manager.update_trade(
                    symbol,
                    current_candle['low'],
                    current_candle['high'],
                    current_candle['close'],
                    current_candle['timestamp']
                )
                self._handle_trade_result(symbol, result)
            
            # Signal Detection
            if not self.trade_manager.has_open_trade(symbol) and self.last_processed[symbol] != current_time:
                if TimeSentinel.is_forbidden(current_time):
                    self.last_processed[symbol] = current_time
                    return
                
                if self.trade_manager.get_open_count() >= MAX_CONCURRENT_TRADES:
                    self.last_processed[symbol] = current_time
                    return
                
                df = calculate_phantom_features(df)
                row = df.iloc[-2]
                
                # Phantom CVD Check
                if self.check_phantom_cvd(df, len(df)-2):
                    state = self.get_phantom_state(row)
                    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_values = self.phantom_model(state_t)
                        action = torch.argmax(q_values).item()
                        confidence = torch.softmax(q_values, dim=1)[0][1].item()
                        
                    if action == 1 and confidence > PHANTOM_CONFIDENCE:
                        trade = self.trade_manager.open_short(symbol, row['close'], confidence, current_time, 'PHANTOM')
                        if trade:
                            self._send_entry_alert(trade, 'PHANTOM')
                
                self.last_processed[symbol] = current_time
                
        except Exception as e:
            print(f"Phantom error {symbol}: {e}")

    def _handle_trade_result(self, symbol, result):
        if result:
            if isinstance(result, dict) and result.get('event') == 'BREAKEVEN':
                self.send_telegram(f"🛡️ **{symbol}** BE @ +{result['roe']*100:.1f}% ROE")
            elif isinstance(result, dict) and result.get('exit_reason'):
                emoji = "✅" if result['pnl'] > 0 else "❌"
                brain = result.get('brain', 'UNKNOWN')
                self.send_telegram(
                    f"{emoji} **{symbol} CLOSED** [{brain}]\n\n"
                    f"Reason: {result['exit_reason']}\n"
                    f"PnL: **${result['pnl']:.2f}** ({result['roi']:.1f}%)\n"
                    f"Balance: **${self.wallet.balance:.2f}**"
                )
                if self.state.is_circuit_breaker_active():
                    self.send_telegram(f"🔒 **CIRCUIT BREAKER** - 24h Pause")

    def _send_entry_alert(self, trade, brain_type):
        brain_emoji = "🦅" if brain_type == 'WRAITH' else "👻"
        hm = "🏠" if trade['leverage'] < WRAITH_LEVERAGE else ""
        self.send_telegram(
            f"{brain_emoji} **HYBRID V7 SIGNAL** {brain_emoji}\n\n"
            f"📉 **SHORT {trade['symbol']}** [{brain_type}] {hm}\n"
            f"💰 Entry: ${trade['entry_price']:,.2f}\n"
            f"💵 Margin: ${trade['margin']:.2f} ({trade['leverage']}x)\n"
            f"🧠 Confidence: {trade['confidence']*100:.1f}%\n"
            f"🛑 SL: ${trade['sl_price']:,.2f}\n"
            f"🎯 TP: ${trade['tp_price']:,.2f}\n"
            f"📊 Open Positions: {self.trade_manager.get_open_count()}"
        )

    def run(self):
        print(f"🦅👻 HYBRID MULTI-PATROL V7 🦅👻")
        print(f"📊 Wraith Symbols: {WRAITH_SYMBOLS}")
        print(f"📊 Phantom Symbols: {PHANTOM_SYMBOLS}")
        print(f"💰 Balance: ${self.wallet.balance:.2f}")
        
        self.send_telegram(
            f"🦅👻 **HYBRID V7 STARTED** 🦅👻\n\n"
            f"📅 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
            f"**🧠 BRAINS:**\n"
            f"  🦅 Wraith V6: {', '.join(WRAITH_SYMBOLS)}\n"
            f"  👻 Phantom V8: {', '.join(PHANTOM_SYMBOLS)}\n\n"
            f"**⚙️ WRAITH CONFIG (BTC/SOL):**\n"
            f"  Leverage: **{WRAITH_LEVERAGE}x**\n"
            f"  SL: {WRAITH_SL_PCT*100}% | TP: {WRAITH_TP_PCT*100}%\n"
            f"  BE ROE: {WRAITH_BE_ROE*100}% | Trail: {WRAITH_TRAILING*100}%\n"
            f"  Confidence: {WRAITH_CONFIDENCE*100}%\n\n"
            f"**⚙️ PHANTOM CONFIG (ETH):**\n"
            f"  Leverage: **{PHANTOM_LEVERAGE}x**\n"
            f"  SL: {PHANTOM_SL_PCT*100}% | TP: {PHANTOM_TP_PCT*100}%\n"
            f"  BE ROE: {PHANTOM_BE_ROE*100}% | Trail: {PHANTOM_TRAILING*100}%\n"
            f"  Confidence: {PHANTOM_CONFIDENCE*100}%\n"
            f"  CVD Threshold: {PHANTOM_CVD_THRESHOLD}\n\n"
            f"**🛡️ PROTECTIONS:**\n"
            f"  House Money: {HOUSE_MONEY_MULTIPLIER}x → {HOUSE_MONEY_REDUCTION*100}%\n"
            f"  Circuit Breaker: {CIRCUIT_BREAKER_DD*100}% DD\n"
            f"  Max Concurrent: {MAX_CONCURRENT_TRADES}\n"
            f"  Forbidden Hours: {FORBIDDEN_HOURS}\n"
            f"  Forbidden Days: {FORBIDDEN_DAYS}\n\n"
            f"**💰 ACCOUNT:**\n"
            f"  Balance: **${self.wallet.balance:.2f}**\n"
            f"  Peak: ${self.wallet.peak_balance:.2f}\n"
            f"  DD: {self.wallet.get_drawdown()*100:.1f}%\n\n"
            f"**📊 Mode: Paper Trading**\n"
            f"**🔔 Health Check: Every 12h**"
        )
        
        while True:
            try:
                if self.state.is_circuit_breaker_active():
                    time.sleep(POLL_INTERVAL)
                    continue
                
                # Process Wraith symbols
                for symbol in WRAITH_SYMBOLS:
                    self.process_wraith_symbol(symbol)
                    self.candles_processed[symbol] = self.candles_processed.get(symbol, 0) + 1
                    time.sleep(1)
                
                # Process Phantom symbols
                for symbol in PHANTOM_SYMBOLS:
                    self.process_phantom_symbol(symbol)
                    self.candles_processed[symbol] = self.candles_processed.get(symbol, 0) + 1
                    time.sleep(1)
                
                # Health Check (every 12 hours)
                if datetime.utcnow() - self.last_health_check >= self.health_check_interval:
                    self.send_health_check()
                
                time.sleep(POLL_INTERVAL)
                
            except Exception as e:
                print(f"Main loop error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    service = HybridMultiPatrol()
    service.run()
