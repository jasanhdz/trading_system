import json
import os
import sys

HISTORY_FILE = '/home/jasan/Develop/trading_system/data/model_history.json'

def analyze_history():
    if not os.path.exists(HISTORY_FILE):
        print("History file not found. Run build_model_history.py first.")
        return

    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)

    # Get all unique symbols
    all_symbols = set()
    for date in history:
        all_symbols.update(history[date].keys())
    
    sorted_dates = sorted(history.keys())
    if not sorted_dates:
        print("No data in history.")
        return

    print(f"\n{'='*80}")
    print(f"{'SYMBOL':<10} | {'TREND (Last 3)':<20} | {'CURRENT ERROR':<15} | {'STATUS':<15}")
    print(f"{'-'*80}")

    recommendations = {
        'RECOMMENDED': [],
        'WATCH': [],
        'PAUSE': []
    }

    for sym in sorted(list(all_symbols)):
        # Extract time series for this symbol
        series = []
        for d in sorted_dates:
            if sym in history[d]:
                series.append(history[d][sym])
        
        if not series:
            continue
            
        current_error = series[-1]
        
        # Calculate Trend (last 3 points if available)
        trend_str = "N/A"
        status = "WATCH"
        
        if len(series) >= 2:
            prev = series[-2]
            change = ((prev - current_error) / prev) * 100
            trend_str = f"{change:+.2f}%"
            
            if len(series) >= 3:
                prev2 = series[-3]
                change2 = ((prev2 - prev) / prev2) * 100
                trend_str = f"{change2:+.1f}% -> {change:+.1f}%"

            # Classification Logic
            # 1. PAUSE if error is very high (> 0.15) OR significant regression (> -10%)
            if current_error > 0.15 or change < -10:
                status = "PAUSE"
            # 2. RECOMMENDED if improving (> 0%) AND error is low (< 0.08)
            elif change > 0 and current_error < 0.08:
                status = "RECOMMENDED"
            # 3. WATCH otherwise (stable, slight regression, or mid-range error)
            else:
                status = "WATCH"
        
        else:
            # New symbol or not enough history
            if current_error < 0.08:
                status = "WATCH"
            else:
                status = "PAUSE"

        recommendations[status].append(sym)
        
        # Print row
        # Clean symbol name for display
        display_sym = sym.replace("/USDT:USDT", "").replace("USDT", "")
        print(f"{display_sym:<10} | {trend_str:<20} | {current_error:.5f}         | {status:<15}")

    print(f"{'='*80}\n")
    
    print("📢 TRADING RECOMMENDATIONS:")
    print(f"✅ ACTIVE (High Confidence): {', '.join([s.replace('/USDT:USDT','').replace('USDT','') for s in recommendations['RECOMMENDED']])}")
    print(f"👀 WATCH (Monitor Closely): {', '.join([s.replace('/USDT:USDT','').replace('USDT','') for s in recommendations['WATCH']])}")
    print(f"⏸️ PAUSE (High Risk/Error): {', '.join([s.replace('/USDT:USDT','').replace('USDT','') for s in recommendations['PAUSE']])}")
    print("\n")

if __name__ == "__main__":
    analyze_history()
