#!/usr/bin/env python3
"""Crear meta.json con model_config filtrado (sin parámetros inválidos)."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.nn_pattern.features import ALL_FEATURES

# Parámetros válidos para AdvancedTemporalNet
VALID_MODEL_PARAMS = {
    'hidden_dim', 'lstm_layers', 'dense_dims', 'dropout', 'use_attention',
    'bidirectional', 'num_classes', 'use_regression'
}

def create_meta_json(symbol: str, timeframe: str):
    """Crea meta.json con model_config filtrado."""
    base_dir = Path(f"models/advanced/{symbol}/{timeframe}")
    
    results_path = base_dir / "production_training_results.json"
    if not results_path.exists():
        print(f"❌ No existe: {results_path}")
        return False
    
    with open(results_path) as f:
        results = json.load(f)
    
    # Filtrar model_config para remover parámetros inválidos
    original_config = results["model_config"]
    filtered_config = {k: v for k, v in original_config.items() 
                      if k in VALID_MODEL_PARAMS}
    
    removed = set(original_config.keys()) - set(filtered_config.keys())
    if removed:
        print(f"   ⚠️  Removidos parámetros inválidos: {removed}")
    
    # Crear metadata correcta
    meta = {
        "sequence_length": results["config"]["sequence_length"],
        "selected_features": ALL_FEATURES,
        "model_config": filtered_config,
        "ensemble_size": 1,
        "symbol": symbol,
        "timeframe": timeframe
    }
    
    # Guardar meta.json
    meta_path = base_dir / "meta.json"
    if meta_path.is_symlink():
        meta_path.unlink()
    
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"   ✅ meta.json actualizado ({len(filtered_config)} params)")
    return True

def main():
    print("🔧 Creando meta.json con model_config filtrado...\\n")
    
    for symbol, tf in [("ETHUSDT", "15m"), ("XRPUSDT", "15m"), ("LTCUSDT", "15m")]:
        print(f"📦 {symbol} {tf}...")
        create_meta_json(symbol, tf)
    
    print("\\n✅ Completado!")
    print("El servicio Flask se recargará automáticamente.")

if __name__ == "__main__":
    main()
