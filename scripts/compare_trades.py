import pandas as pd
import json
import sys

def compare_trades():
    # Load Python trades
    try:
        df_py = pd.read_csv('reports/python_trades.csv')
        df_py['entry_time'] = pd.to_datetime(df_py['entry_time'])
        print(f"Python Trades: {len(df_py)}")
    except Exception as e:
        print(f"Error loading Python trades: {e}")
        return

    # Load TS trades
    try:
        with open('binance-futures-bot-ts/reports/ts_backtest_results.json', 'r') as f:
            data = json.load(f)
            ts_trades = data['trades']
        
        df_ts = pd.DataFrame(ts_trades)
        # TS timestamp is Exit Time, entryTime is Entry Time
        df_ts['entry_time'] = pd.to_datetime(df_ts['entryTime'], unit='ms')
        print(f"TS Trades: {len(df_ts)}")
    except Exception as e:
        print(f"Error loading TS trades: {e}")
        return

    # Compare
    print("\n--- Comparison ---")
    
    # Find first divergence
    min_len = min(len(df_py), len(df_ts))
    for i in range(min_len):
        py_trade = df_py.iloc[i]
        ts_trade = df_ts.iloc[i]
        
        # Allow 1 minute tolerance for entry time
        time_diff = abs((py_trade['entry_time'] - ts_trade['entry_time']).total_seconds())
        
        if time_diff > 300: # 5 minutes
            print(f"Divergence at Trade #{i+1}")
            print(f"Python: {py_trade['entry_time']} | Entry: {py_trade.get('entry_price', 'N/A')} | Reason: {py_trade['reason']}")
            print(f"TS:     {ts_trade['entry_time']} | Entry: {ts_trade['entryPrice']} | Reason: {ts_trade['reason']}")
            break
            
        # Check Exit Reason
        if py_trade['reason'] != ts_trade['reason']:
            print(f"Exit Reason Mismatch at Trade #{i+1} ({py_trade['entry_time']})")
            print(f"Python: {py_trade['reason']} | PnL: {py_trade['pnl']:.2f}")
            print(f"TS:     {ts_trade['reason']} | PnL: {ts_trade['pnl']:.2f}")
            
            # If TS is STOP_LOSS and Python is TRAILING, this is our issue
            if ts_trade['reason'] == 'STOP_LOSS' and py_trade['reason'] == 'TRAILING':
                print("--> CRITICAL: TS hit SL, Python hit Trailing")
            break

if __name__ == "__main__":
    compare_trades()
