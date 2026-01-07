#!/home/jasan/Develop/trading_system/binance-futures-bot-ts/.venv/bin/python3
"""
📊 TRADE REPORT - Comprehensive Trade Analysis with Peak ROI
=============================================================
Combines Binance trade history with PM2 log analysis to show:
- Entry/Exit times and prices
- PnL and ROI
- Peak ROI (positive and negative) during trade lifetime
- Salvability analysis (could trailing stop have saved the trade?)

Usage:
  trade_report --today                    # Today's trades
  trade_report --yesterday --status LOSS  # Yesterday's losses with peak ROI
  trade_report --week --status WIN        # Week's winning trades
  trade_report --peak                     # Include Peak ROI analysis (slower)
"""
import ccxt
import pandas as pd
import os
import re
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict
import argparse

# Config
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent / "binance-futures-bot-ts"
DOTENV_PATH = PROJECT_ROOT / ".env"
LOG_FILE = Path.home() / ".pm2/logs/01-Trading-Bot-out.log"

load_dotenv(DOTENV_PATH)
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# All 21 trading symbols (Priority + Secondary)
TARGET_SYMBOLS = [
    # Priority (Alpha Batch)
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
    'DOGE/USDT', 'LINK/USDT', 'AVAX/USDT', 'POL/USDT',
    # Secondary (Bravo Batch)
    'BNB/USDT', 'DOT/USDT', 'LTC/USDT', 'UNI/USDT', 'ATOM/USDT',
    'NEAR/USDT', 'PEPE/USDT', 'FET/USDT', 'SEI/USDT', 'WLD/USDT',
    'INJ/USDT', 'APT/USDT'
]

ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def parse_args():
    parser = argparse.ArgumentParser(
        description="📊 Trade Report with Peak ROI Analysis",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    time_group = parser.add_argument_group('⏱️  Time Filters')
    time_group.add_argument('--today', action='store_true', help='Trades from today')
    time_group.add_argument('--yesterday', action='store_true', help='Trades from yesterday')
    time_group.add_argument('--week', action='store_true', help='Last 7 days')
    time_group.add_argument('--month', action='store_true', help='Last 30 days')
    time_group.add_argument('--days', type=int, metavar='N', help='Last N days')
    
    filter_group = parser.add_argument_group('🔍 Filters')
    filter_group.add_argument('--status', type=str, choices=['WIN', 'LOSS'], help='WIN or LOSS trades only')
    filter_group.add_argument('--symbol', type=str, help='Filter by symbol (e.g., SOL, BTCUSDT)')
    filter_group.add_argument('--side', type=str, choices=['LONG', 'SHORT'], help='Filter by trade direction')
    
    analysis_group = parser.add_argument_group('📈 Analysis')
    analysis_group.add_argument('--peak', action='store_true', help='Include Peak ROI analysis (parses logs)')
    analysis_group.add_argument('--salvable', type=float, default=3.0, help='ROI threshold for salvability (default: 3.0%%)')
    
    return parser.parse_args()

def get_time_range(args):
    """Get start/end datetime based on args."""
    now = datetime.now()
    
    if args.today:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
        label = "HOY"
    elif args.yesterday:
        yesterday = now - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        label = "AYER"
    elif args.week:
        start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
        label = "ÚLTIMA SEMANA"
    elif args.month:
        start = (now - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
        label = "ÚLTIMO MES"
    elif args.days:
        start = (now - timedelta(days=args.days)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
        label = f"ÚLTIMOS {args.days} DÍAS"
    else:
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
        label = "HOY + AYER"
    
    return start, end, label

def fetch_trades(start_date, end_date):
    """Fetch trades from Binance within date range."""
    print("🔄 Conectando a Binance...", flush=True)
    
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'options': {'defaultType': 'future'},
        'enableRateLimit': True
    })
    
    # Convert to timestamp
    since = int(start_date.timestamp() * 1000)
    
    all_trades = []
    for symbol in TARGET_SYMBOLS:
        try:
            trades = exchange.fetch_my_trades(symbol, since=since)
            for t in trades:
                t['symbol'] = symbol
                info = t.get('info', {})
                t['realized_pnl'] = float(info.get('realizedPnl', 0))
                t['commission'] = float(info.get('commission', 0))
                t['position_side'] = info.get('positionSide', 'BOTH')
            all_trades.extend(trades)
        except Exception as e:
            pass
    
    if not all_trades:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_trades)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Make timezone naive for comparison
    if df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_localize(None)
    
    # Filter by date range
    df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
    
    return df

def group_into_operations(df):
    """Group individual trades into complete operations (entry + exit)."""
    if df.empty:
        return []
    
    operations = []
    
    # Group by symbol
    for symbol in df['symbol'].unique():
        sym_df = df[df['symbol'] == symbol].sort_values('datetime')
        
        # Track position state
        position = None
        
        for _, trade in sym_df.iterrows():
            pnl = trade['realized_pnl']
            side = trade['side']  # 'buy' or 'sell'
            
            if pnl != 0:  # This is an exit trade
                if position is not None:
                    # Complete the operation
                    position['exit_time'] = trade['datetime']
                    position['exit_price'] = trade['price']
                    position['pnl'] = pnl
                    position['exit_qty'] = trade['amount']
                    operations.append(position)
                    position = None
            else:
                # This is an entry
                if position is None:
                    position = {
                        'symbol': symbol.replace('/USDT', '').replace('/', ''),
                        'side': 'LONG' if side == 'buy' else 'SHORT',
                        'entry_time': trade['datetime'],
                        'entry_price': trade['price'],
                        'entry_qty': trade['amount'],
                        'exit_time': None,
                        'exit_price': None,
                        'pnl': None
                    }
    
    return operations

def parse_log_for_peak_roi(operations):
    """Parse PM2 logs to find Peak ROI for each operation's specific lifetime."""
    if not LOG_FILE.exists():
        print("⚠️  Log file not found, skipping Peak ROI analysis")
        return operations
    
    print("📊 Analizando logs para Peak ROI...", flush=True)
    
    # Read log file
    with open(LOG_FILE, 'rb') as f:
        content = f.read().decode('utf-8', errors='ignore')
    
    lines = content.split('\n')
    
    # Parse all ROI entries with timestamps and entry prices
    # Pattern: TIME | SYMBOL | SIDE | ENTRY_PRICE | MARK_PRICE | ROI%
    pattern = r'(\d+:\d+:\d+\s+[AP]M)\s+.*?(\w+USDT)\s+.*?(LONG|SHORT)\s+.*?([\d.]+)\s+.*?([\d.]+)\s+.*?([+-]?[\d.]+)%'
    
    # Build index of log entries by symbol, side, and entry_price
    log_entries = []
    
    for line in lines:
        clean_line = ANSI_ESCAPE.sub('', line)
        match = re.search(pattern, clean_line)
        if match:
            try:
                time_str = match.group(1)
                entry_price = float(match.group(4))
                mark_price = float(match.group(5))
                roi = float(match.group(6))
                
                log_entries.append({
                    'time_str': time_str,
                    'symbol': match.group(2),
                    'side': match.group(3),
                    'entry_price': entry_price,
                    'mark_price': mark_price,
                    'roi': roi
                })
            except:
                pass
    
    print(f"   Parsed {len(log_entries):,} log entries")
    
    # Match ROI entries to operations BY ENTRY PRICE (unique identifier)
    for op in operations:
        if op['entry_time'] is None or op['exit_time'] is None or op['entry_price'] is None:
            op['peak_pos'] = None
            op['peak_neg'] = None
            op['salvable'] = None
            continue
        
        op_symbol = op['symbol'] + 'USDT'
        op_side = op['side']
        op_entry_price = op['entry_price']
        
        # Find ALL log entries that match this specific trade by:
        # 1. Same symbol
        # 2. Same side
        # 3. Same entry price (within 0.01% tolerance for float comparison)
        price_tolerance = op_entry_price * 0.0001  # 0.01% tolerance
        
        relevant_rois = [
            e['roi'] for e in log_entries
            if e['symbol'] == op_symbol 
            and e['side'] == op_side
            and abs(e['entry_price'] - op_entry_price) < price_tolerance
        ]
        
        if relevant_rois:
            op['peak_pos'] = max(relevant_rois)
            op['peak_neg'] = min(relevant_rois)
            op['salvable'] = op['peak_pos'] >= 3.0
            op['roi_samples'] = len(relevant_rois)
        else:
            op['peak_pos'] = None
            op['peak_neg'] = None
            op['salvable'] = None
            op['roi_samples'] = 0
    
    return operations

def print_report_simple(operations, args, label):
    """Print formatted report. Filters already applied."""
    if not operations:
        print("❌ No se encontraron operaciones en el rango seleccionado.")
        return
    
    # Sort by entry time descending
    operations.sort(key=lambda x: x['entry_time'] if x['entry_time'] else datetime.min, reverse=True)
    
    # Calculate totals
    total_pnl = sum(op['pnl'] for op in operations if op['pnl'])
    wins = sum(1 for op in operations if op['pnl'] and op['pnl'] > 0)
    losses = sum(1 for op in operations if op['pnl'] and op['pnl'] <= 0)
    win_rate = (wins / len(operations) * 100) if operations else 0
    
    # Print header
    status_label = f" ({args.status})" if args.status else ""
    print(f"\n📊 REPORTE DE OPERACIONES - {label}{status_label}")
    print("="*100)
    
    # Determine if we're showing peak ROI
    show_peak = args.peak and any(op.get('peak_pos') is not None for op in operations)
    
    if show_peak:
        print(f"{'Par':<8} {'Lado':<6} {'Entrada':<14} {'Salida':<14} {'P.Entrada':<12} {'P.Salida':<12} {'PnL':<10} {'Peak+':<8} {'Peak-':<8} {'@3%?'}")
        print("-"*100)
    else:
        print(f"{'Par':<8} {'Lado':<6} {'Entrada':<14} {'Salida':<14} {'P.Entrada':<12} {'P.Salida':<12} {'PnL':<10}")
        print("-"*80)
    
    for op in operations:
        symbol = op['symbol']
        side = op['side'][0]  # L or S
        entry_time = op['entry_time'].strftime('%m-%d %H:%M') if op['entry_time'] else '-'
        exit_time = op['exit_time'].strftime('%m-%d %H:%M') if op['exit_time'] else '-'
        entry_price = f"${op['entry_price']:.4f}" if op['entry_price'] else '-'
        exit_price = f"${op['exit_price']:.4f}" if op['exit_price'] else '-'
        pnl = f"${op['pnl']:.2f}" if op['pnl'] else '-'
        
        # Color PnL
        if op['pnl'] and op['pnl'] < 0:
            pnl = f"\033[91m{pnl}\033[0m"  # Red
        elif op['pnl'] and op['pnl'] > 0:
            pnl = f"\033[92m{pnl}\033[0m"  # Green
        
        if show_peak:
            peak_pos = f"+{op['peak_pos']:.2f}%" if op.get('peak_pos') is not None else '-'
            peak_neg = f"{op['peak_neg']:.2f}%" if op.get('peak_neg') is not None else '-'
            salvable = "✅" if op.get('salvable') else "❌" if op.get('salvable') is False else "-"
            print(f"{symbol:<8} {side:<6} {entry_time:<14} {exit_time:<14} {entry_price:<12} {exit_price:<12} {pnl:<10} {peak_pos:<8} {peak_neg:<8} {salvable}")
        else:
            print(f"{symbol:<8} {side:<6} {entry_time:<14} {exit_time:<14} {entry_price:<12} {exit_price:<12} {pnl:<10}")
    
    # Summary
    print("="*100)
    print(f"\n📈 RESUMEN:")
    print(f"   Total Operaciones: {len(operations)}")
    print(f"   Ganadas: {wins} | Perdidas: {losses} | Win Rate: {win_rate:.1f}%")
    print(f"   PnL Total: ${total_pnl:.2f}")
    
    if show_peak:
        salvable_count = sum(1 for op in operations if op.get('salvable'))
        print(f"\n   🛡️ Operaciones salvables @{args.salvable}%: {salvable_count}/{len(operations)}")

def main():
    args = parse_args()
    
    # Get time range
    start, end, label = get_time_range(args)
    print(f"📅 Rango: {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')}")
    
    # Fetch trades
    df = fetch_trades(start, end)
    
    if df.empty:
        print("❌ No se encontraron trades en el rango seleccionado.")
        return
    
    # Group into operations
    operations = group_into_operations(df)
    
    # Apply filters BEFORE counting
    if args.status:
        if args.status == 'WIN':
            operations = [op for op in operations if op['pnl'] and op['pnl'] > 0]
        else:
            operations = [op for op in operations if op['pnl'] and op['pnl'] <= 0]
    
    if args.symbol:
        sym = args.symbol.upper().replace('USDT', '').replace('/', '')
        operations = [op for op in operations if op['symbol'] == sym]
    
    if args.side:
        operations = [op for op in operations if op['side'] == args.side]
    
    if not operations:
        print("❌ No se encontraron operaciones con los filtros aplicados.")
        return
    
    print(f"   Operaciones: {len(operations)}")
    
    # Parse logs for Peak ROI if requested
    if args.peak:
        operations = parse_log_for_peak_roi(operations)
    
    # Print report (filters already applied)
    print_report_simple(operations, args, label)

if __name__ == "__main__":
    main()
