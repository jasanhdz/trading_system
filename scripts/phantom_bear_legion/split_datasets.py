#!/usr/bin/env python3
"""
THE OCEAN: Phase 2.5 - The Splitter
Separates the 'regime_labeled_history.csv' into regime-specific datasets.
Focus: Extracting 'dataset_bear_full.csv' for the Bear Legion.
"""
import pandas as pd
from pathlib import Path

# Config
DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = DATA_DIR / "regime_labeled_history.csv"
OUTPUT_BEAR = DATA_DIR / "dataset_bear_full.csv"

def main():
    print("✂️ THE SPLITTER: EXTRACTING REGIMES ✂️")
    
    if not INPUT_FILE.exists():
        print(f"❌ Input not found: {INPUT_FILE}")
        print("   Run classify_regimes.py first.")
        return

    print(f"   Loading Labelled History: {INPUT_FILE.name}")
    df = pd.read_csv(INPUT_FILE)
    
    # Filter Bear Trends
    bear_df = df[df['regime_type'] == 'BEAR_TREND'].copy()
    
    print(f"   Original Rows: {len(df):,}")
    print(f"   Bear Rows:     {len(bear_df):,} ({len(bear_df)/len(df)*100:.2f}%)")
    
    if len(bear_df) == 0:
        print("❌ No Bear Trends found! Check classification thresholds.")
        return
        
    print(f"   Saving Bear Legion Dataset to: {OUTPUT_BEAR.name}...")
    bear_df.to_csv(OUTPUT_BEAR, index=False)
    
    print("\n✅ DATASET SPLIT COMPLETE.")
    print("   The Bear Legion is ready for sub-classification.")

if __name__ == "__main__":
    main()
