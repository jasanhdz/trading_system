#!/usr/bin/env python3
"""
THE OCEAN: Phase 3.5 - The Species Splitter
Divides the Bear Legion into specialized sub-datasets based on Volatility.
Thresholds derived from statistical analysis (analyze_bear_biology.py).
"""
import pandas as pd
from pathlib import Path

# Config
DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = DATA_DIR / "dataset_bear_full.csv"

# Thresholds (Determined by Analysis)
THRESH_GRINDER = 0.0020 # 0.20% (Approx Median)
THRESH_CRASH   = 0.0050 # 0.50% (Approx Top 10%)

def main():
    print("🧬 BEAR SPECIES SPLITTER: SEGREGATING DATASETS 🧬")
    
    if not INPUT_FILE.exists():
        print(f"❌ Input not found: {INPUT_FILE}")
        return

    print(f"   Loading Bear Legion: {INPUT_FILE.name}")
    df = pd.read_csv(INPUT_FILE)
    
    # Calculate Volatility
    df['volatility'] = (df['high'] - df['low']) / df['open']
    
    # Split
    grinder_df = df[df['volatility'] < THRESH_GRINDER].copy()
    crash_df   = df[df['volatility'] > THRESH_CRASH].copy()
    
    # Save
    path_grinder = DATA_DIR / "dataset_bear_grinder.csv"
    path_crash   = DATA_DIR / "dataset_bear_crash.csv"
    
    print(f"\n   -----------------------------------------")
    print(f"   🐻 GRINDERS (Low Vol < {THRESH_GRINDER*100:.1f}%)")
    print(f"      Count: {len(grinder_df):,} candles")
    print(f"      Saving to: {path_grinder.name}")
    grinder_df.to_csv(path_grinder, index=False)
    
    print(f"\n   -----------------------------------------")
    print(f"   🦖 CRASHES (High Vol > {THRESH_CRASH*100:.1f}%)")
    print(f"      Count: {len(crash_df):,} candles")
    print(f"      Saving to: {path_crash.name}")
    crash_df.to_csv(path_crash, index=False)
    
    print("\n✅ SPECIES SEGREGATION COMPLETE.")
    print("   Ready for Specialist Training (Phase 4).")

if __name__ == "__main__":
    main()
