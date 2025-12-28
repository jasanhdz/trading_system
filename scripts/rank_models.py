import json
import os

HISTORY_FILE = '/home/jasan/Develop/trading_system/data/model_history.json'

def rank_models():
    if not os.path.exists(HISTORY_FILE):
        print("History file not found.")
        return

    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)

    # Get latest date
    dates = sorted(history.keys())
    if not dates:
        print("No data.")
        return
    
    latest_date = dates[-1]
    print(f"📊 MODEL POWER RANKING (Based on {latest_date} Performance)")
    print(f"The 'Error' (LogLoss) measures uncertainty. Lower is better.")
    print(f"{'='*70}")
    print(f"{'RANK':<5} | {'SYMBOL':<10} | {'ERROR':<10} | {'QUALITY TIER':<20} | {'STATUS'}")
    print(f"{'-'*70}")

    metrics = history[latest_date]
    # Sort by error ascending (lower is better)
    ranked = sorted(metrics.items(), key=lambda x: x[1])

    for i, (sym, error) in enumerate(ranked, 1):
        # Determine Tier
        if error < 0.02:
            tier = "🎯 SNIPER (God Tier)"
            desc = "Perfect"
        elif error < 0.05:
            tier = "🔫 MARKSMAN (Elite)"
            desc = "Excellent"
        elif error < 0.08:
            tier = "⚔️ SOLDIER (Solid)"
            desc = "Good"
        elif error < 0.12:
            tier = "🛡️ RECRUIT (Okay)"
            desc = "Average"
        else:
            tier = "⚠️ CADET (Unstable)"
            desc = "Risky"

        # Clean symbol
        clean_sym = sym.replace("/USDT:USDT", "").replace("USDT", "")
        
        print(f"#{i:<4} | {clean_sym:<10} | {error:.5f}   | {tier:<20} | {desc}")

    print(f"{'='*70}")

if __name__ == "__main__":
    rank_models()
