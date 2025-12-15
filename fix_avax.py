import json
from pathlib import Path

path = Path("models/advanced/AVAXUSDT/15m/meta.json")
data = json.loads(path.read_text())

# Revert to default config (128 hidden)
data['model_config']['hidden_dim'] = 128
data['model_config']['lstm_layers'] = 2
data['model_config']['dense_dims'] = [256, 128]

path.write_text(json.dumps(data, indent=2))
print("Reverted AVAX 15m meta.json")
