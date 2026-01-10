
import ccxt
import os
from pathlib import Path

# Load env manually
env_path = Path("/home/jasan/Develop/trading_system/binance-futures-bot-ts/.env")
if not env_path.exists():
    print(f"❌ .env not found at {env_path}")
    exit(1)

api_key = None
secret = None

with open(env_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key == "BINANCE_API_KEY":
                api_key = value
            elif key == "BINANCE_API_SECRET":
                secret = value

if not api_key or not secret:
    print("❌ API keys not found in .env")
    exit(1)

try:
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': secret,
        'options': {'defaultType': 'future'}
    })

    # Fetch Balance
    balance = exchange.fetch_balance()
    total_wallet_balance = float(balance['info']['totalWalletBalance'])
    total_unrealized_pnl = float(balance['info']['totalUnrealizedProfit'])
    total_margin_balance = float(balance['info']['totalMarginBalance'])

    print(f"\n💰 **BALANCE CHECK**")
    print(f"   Wallet Balance:   ${total_wallet_balance:.2f}")
    print(f"   Unrealized PnL:   ${total_unrealized_pnl:.2f}")
    print(f"   Total Equity:     ${total_margin_balance:.2f}")
    
    # Fetch Positions
    positions = exchange.fetch_positions()
    active_positions = [p for p in positions if float(p['contracts']) > 0]

    print(f"\n📊 **OPEN POSITIONS ({len(active_positions)})**")
    print("="*80)
    print(f"{'Symbol':<10} {'Side':<6} {'Entry':<10} {'Mark':<10} {'PnL':<10} {'ROI':<8} {'Lev':<4}")
    print("-" * 80)

    for p in active_positions:
        symbol = p['symbol']
        side = p['side'].upper()
        entry_price = float(p['entryPrice'])
        mark_price = float(p['markPrice'])
        pnl = float(p['unrealizedPnl'])
        
        # Safe leverage handling
        lev_raw = p.get('leverage', 1)
        try:
            leverage = float(lev_raw) if lev_raw is not None else 1.0
        except:
            leverage = 1.0
        
        # Calculate ROI %
        if entry_price > 0:
            if side == 'LONG':
                roi = ((mark_price - entry_price) / entry_price) * leverage * 100
            else:
                roi = ((entry_price - mark_price) / entry_price) * leverage * 100
        else:
            roi = 0.0

        pnl_str = f"${pnl:.2f}"
        roi_str = f"{roi:.2f}%"
        
        print(f"{symbol:<10} {side:<6} {entry_price:<10.4f} {mark_price:<10.4f} {pnl_str:<10} {roi_str:<8} {int(leverage):<4}x")

    print("="*80)

except Exception as e:
    print(f"❌ Error: {e}")
