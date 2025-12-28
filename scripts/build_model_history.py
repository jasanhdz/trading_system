import os
import re
import json
import datetime
from pathlib import Path

LOG_DIR = '/home/jasan/Develop/trading_system/logs'
OUTPUT_FILE = '/home/jasan/Develop/trading_system/data/model_history.json'

def parse_log_file(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

    # Try to extract date from filename first
    filename = os.path.basename(filepath)
    date_str = None
    
    # Format: training_YYYYMMDD_HHMMSS.log or similar
    match_date = re.search(r'(\d{8})', filename)
    if match_date:
        d = match_date.group(1)
        # YYYYMMDD -> YYYY-MM-DD
        date_str = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    
    # If not in filename, try to find a timestamp in the log content
    if not date_str:
        # Look for [Day Mon DD HH:MM:SS UTC YYYY] format common in daily_retrain.log
        # or ISO format
        match_ts = re.search(r'\[\w+ (\w+ \d+ \d+:\d+:\d+ UTC \d+)\]', content)
        if match_ts:
            try:
                dt = datetime.datetime.strptime(match_ts.group(1), "%b %d %H:%M:%S UTC %Y")
                date_str = dt.strftime("%Y-%m-%d")
            except:
                pass

    if not date_str:
        # Fallback: use file modification time
        mtime = os.path.getmtime(filepath)
        date_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

    metrics = {}
    
    # Split by runs if multiple runs in one file (like daily_retrain.log)
    # We want to capture ALL runs in the file if possible, but mapping them to dates is tricky if they are all in one file.
    # For daily_retrain.log, we might have multiple days.
    
    if filename == 'daily_retrain.log':
        # Special handling for the main log file which appends
        # Split by "Starting Daily Retraining"
        runs = re.split(r"Starting Daily Retraining", content)
        for run in runs:
            if not run.strip(): continue
            
            # Find date in this specific run block
            run_date = None
            match_run_ts = re.search(r'\[\w+ (\w+ \d+ \d+:\d+:\d+ UTC \d+)\]', run)
            if match_run_ts:
                try:
                    dt = datetime.datetime.strptime(match_run_ts.group(1), "%b %d %H:%M:%S UTC %Y")
                    run_date = dt.strftime("%Y-%m-%d")
                except:
                    pass
            
            if not run_date: continue # Skip if no date found for this block
            
            if run_date not in metrics:
                metrics[run_date] = {}

            # Parse metrics in this block
            # Regex for "Starting training for SYMBOL" ... "eval-mlogloss:VALUE"
            # We look for the LAST eval-mlogloss for each symbol in this block
            sections = re.split(r"Starting training for (.+) ->", run)
            for i in range(1, len(sections), 2):
                symbol = sections[i].strip()
                text = sections[i+1]
                matches = re.findall(r"eval-mlogloss:([\d\.]+)", text)
                if matches:
                    metrics[run_date][symbol] = float(matches[-1])
                    
        return metrics

    else:
        # Single run files
        # Parse metrics
        sections = re.split(r"Starting training for (.+) ->", content)
        if len(sections) < 2:
             # Try old format: "Training model for SYMBOL"
             sections = re.split(r"Training model for (\w+)", content)
        
        run_metrics = {}
        for i in range(1, len(sections), 2):
            symbol = sections[i].strip()
            text = sections[i+1]
            matches = re.findall(r"eval-mlogloss:([\d\.]+)", text)
            if matches:
                run_metrics[symbol] = float(matches[-1])
        
        if run_metrics:
            return {date_str: run_metrics}
        return {}

def main():
    all_history = {}
    
    # 1. Scan directory
    files = [f for f in os.listdir(LOG_DIR) if f.endswith('.log')]
    
    print(f"Scanning {len(files)} log files...")
    
    for f in files:
        if not (f.startswith('daily_retrain') or f.startswith('training_')):
            continue
            
        path = os.path.join(LOG_DIR, f)
        file_data = parse_log_file(path)
        
        for date, data in file_data.items():
            if date not in all_history:
                all_history[date] = {}
            # Merge (latest file overwrites if duplicate date/symbol, but usually distinct)
            all_history[date].update(data)

    # Sort by date
    sorted_history = dict(sorted(all_history.items()))
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(sorted_history, f, indent=2)
        
    print(f"History saved to {OUTPUT_FILE}")
    print(f"Found data for {len(sorted_history)} days.")

if __name__ == "__main__":
    main()
