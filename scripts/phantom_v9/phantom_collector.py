#!/usr/bin/env python3
"""
Phantom V9 Data Collector
Runs update_ml_candles.py in a continuous loop to keep the DB fresh.
"""
import time
import subprocess
import sys
from pathlib import Path

# Ensure we are in the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def run_loop():
    print("👻 Starting Phantom V9 Data Collector Loop")
    
    while True:
        try:
            print(f"\n🔄 [{(time.strftime('%H:%M:%S'))}] Updating Candles...")
            
            # Run update_ml_candles.py with a short lookback (2 days) for speed
            # We assume the DB is initialized. If not, run manually with --days 365 first.
            cmd = [sys.executable, "scripts/update_ml_candles.py", "--days", "2"]
            
            result = subprocess.run(
                cmd, 
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Filter output to show only relevant info
                for line in result.stdout.split('\n'):
                    if "✓" in line or "→" in line:
                        print(line)
            else:
                print(f"❌ Collector Failed:\n{result.stderr}")
                
        except Exception as e:
            print(f"❌ Loop Error: {e}")
        
        # Sleep 60s (Candles are 5m, so 1m is plenty fast)
        time.sleep(60)

if __name__ == "__main__":
    run_loop()
