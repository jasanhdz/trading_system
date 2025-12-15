import json
from pathlib import Path

def fix_meta(symbol, timeframe):
    path = Path(f"models/advanced/{symbol}/{timeframe}/meta.json")
    if not path.exists():
        print(f"Skipping {symbol} {timeframe} (not found)")
        return

    print(f"Fixing {symbol} {timeframe}...")
    try:
        data = json.loads(path.read_text())
        
        # Update model config
        if 'model_config' in data:
            data['model_config']['hidden_dim'] = 192
            data['model_config']['lstm_layers'] = 3
            data['model_config']['dense_dims'] = [384, 255, 128]
            
            path.write_text(json.dumps(data, indent=2))
            print(f"Updated {path}")
        else:
            print(f"No model_config in {path}")
            
    except Exception as e:
        print(f"Error fixing {path}: {e}")

symbols = ["LINKUSDT", "SOLUSDT", "SNXUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT"]
timeframe = "15m"

for sym in symbols:
    fix_meta(sym, timeframe)
