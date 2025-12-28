import re
import sys

LOG_FILE = '/home/jasan/Develop/trading_system/logs/daily_retrain.log'

def parse_log():
    try:
        with open(LOG_FILE, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("Log file not found.")
        return

    # Split by runs
    runs = content.split("🚀 Starting Daily Retraining...")
    if len(runs) < 2:
        print("Not enough training runs found for comparison.")
        return

    # Get last two runs (ignoring empty first split if any)
    last_run = runs[-1]
    prev_run = runs[-2]

    def extract_metrics(run_text):
        metrics = {}
        # Pattern: "Training model for SYMBOL..." followed eventually by "[ITER] ... eval-mlogloss:VALUE"
        # We need to capture the *last* eval-mlogloss for each symbol block.
        
        # Split by symbol sections
        sections = re.split(r"Starting training for (.+) ->", run_text)
        
        # sections[0] is pre-text. 
        # sections[1] is symbol, sections[2] is content, sections[3] is symbol...
        
        for i in range(1, len(sections), 2):
            symbol = sections[i]
            text = sections[i+1]
            
            # Find all mlogloss values
            matches = re.findall(r"eval-mlogloss:([\d\.]+)", text)
            if matches:
                final_loss = float(matches[-1])
                metrics[symbol] = final_loss
        
        return metrics

    current_metrics = extract_metrics(last_run)
    prev_metrics = extract_metrics(prev_run)

    print(f"{'Symbol':<10} | {'Previous Loss':<15} | {'Current Loss':<15} | {'Improvement':<15}")
    print("-" * 65)

    all_symbols = sorted(list(set(current_metrics.keys()) | set(prev_metrics.keys())))
    
    for sym in all_symbols:
        prev = prev_metrics.get(sym, None)
        curr = current_metrics.get(sym, None)
        
        prev_str = f"{prev:.5f}" if prev is not None else "N/A"
        curr_str = f"{curr:.5f}" if curr is not None else "N/A"
        
        imp_str = "N/A"
        if prev is not None and curr is not None and prev > 0:
            imp = ((prev - curr) / prev) * 100
            imp_str = f"{imp:+.2f}%"
            
        print(f"{sym:<10} | {prev_str:<15} | {curr_str:<15} | {imp_str:<15}")

if __name__ == "__main__":
    parse_log()
