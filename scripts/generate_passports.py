#!/usr/bin/env python3
"""
Generate metadata.json (Quality Passports) for existing models.
Reads the Training Diary and creates passports based on latest accuracy.
"""
import json
from pathlib import Path
from datetime import datetime

MODELS_DIR = Path("/home/jasan/Develop/trading_system/models/v2_ensemble")
DIARY_PATH = Path("/home/jasan/Develop/data/training_diary.json")

def main():
    # Load training diary
    if not DIARY_PATH.exists():
        print("❌ Training Diary not found!")
        return
    
    with open(DIARY_PATH, 'r') as f:
        diary_entries = json.load(f)
    
    # Group by symbol and get latest metrics
    symbol_metrics = {}
    
    for entry in diary_entries:
        symbol = entry.get('symbol', '')
        # Convert symbol format: "SOL/USDT:USDT" -> "SOLUSDT"
        clean_symbol = symbol.replace("/", "").replace(":", "").replace("-", "").replace("USDT", "") + "USDT"
        
        model_type = entry.get('model_type', '')
        metrics = entry.get('metrics', {})
        timestamp = entry.get('timestamp', '')
        
        if clean_symbol not in symbol_metrics:
            symbol_metrics[clean_symbol] = {
                'symbol': symbol,
                'clean_symbol': clean_symbol,
                'tcn_accuracy': 0.0,
                'xgb_accuracy': 0.0,
                'tcn_f1': 0.0,
                'xgb_f1': 0.0,
                'samples': 0,
                'timestamp': timestamp
            }
        
        # Update with latest data (diary entries are chronological)
        if model_type == 'TCN':
            symbol_metrics[clean_symbol]['tcn_accuracy'] = metrics.get('accuracy', 0)
            symbol_metrics[clean_symbol]['tcn_f1'] = metrics.get('f1_score', 0)
        elif model_type == 'XGBOOST':
            symbol_metrics[clean_symbol]['xgb_accuracy'] = metrics.get('accuracy', 0)
            symbol_metrics[clean_symbol]['xgb_f1'] = metrics.get('f1_score', 0)
        
        symbol_metrics[clean_symbol]['samples'] = metrics.get('samples', 0)
        symbol_metrics[clean_symbol]['timestamp'] = timestamp
    
    # Generate metadata.json for each model directory
    print("🛂 Generating Quality Passports...")
    print("=" * 60)
    
    generated = 0
    for clean_symbol, data in symbol_metrics.items():
        model_dir = MODELS_DIR / clean_symbol
        
        if not model_dir.exists():
            print(f"⚠️  {clean_symbol}: Model directory not found, skipping")
            continue
        
        best_accuracy = max(data['tcn_accuracy'], data['xgb_accuracy'])
        best_f1 = max(data['tcn_f1'], data['xgb_f1'])
        
        metadata = {
            "symbol": data['symbol'],
            "clean_symbol": clean_symbol,
            "version": "v2.1",
            "timestamp": data['timestamp'],
            "accuracy": best_accuracy,
            "f1_score": best_f1,
            "tcn_accuracy": data['tcn_accuracy'],
            "xgb_accuracy": data['xgb_accuracy'],
            "samples": data['samples'],
            "models": ["TCN", "XGBOOST"],
            "generated_by": "backfill_script"
        }
        
        meta_path = model_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        status = "✅" if best_accuracy >= 0.50 else "⛔"
        print(f"{status} {clean_symbol}: Acc={best_accuracy:.2%} (TCN={data['tcn_accuracy']:.2%}, XGB={data['xgb_accuracy']:.2%})")
        generated += 1
    
    print("=" * 60)
    print(f"🎉 Generated {generated} Quality Passports!")
    print("\n📌 Restart ML Service to apply: pm2 restart 03-ML-Service-V2")

if __name__ == "__main__":
    main()
