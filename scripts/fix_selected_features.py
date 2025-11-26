#!/usr/bin/env python3
"""Generar meta.json con el número correcto de features desde el checkpoint del modelo."""
import json
import torch
from pathlib import Path

def fix_meta_json(symbol: str, timeframe: str):
    """Genera meta.json con el número correcto de features."""
    base_dir = Path(f"models/advanced/{symbol}/{timeframe}")
    
    # Cargar checkpoint para obtener input_dim
    model_path = base_dir / "model.pt"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extraer input_dim de input_proj.0.weight
    if 'input_proj.0.weight' in checkpoint:
        input_dim = checkpoint['input_proj.0.weight'].shape[1]
        print(f"   ✅ Detectadas {input_dim} features del modelo")
    else:
        print(f"   ❌ No se puede detectar input_dim del checkpoint")
        return False
    
    # Cargar production_training_results.json
    results_path = base_dir / "production_training_results.json"
    with open(results_path) as f:
        results = json.load(f)
    
    # Generar lista de features dummy (los nombres no importan, solo el tamaño)
    selected_features = [f"feature_{i}" for i in range(input_dim)]
    
    # Crear metadata
    meta = {
        "sequence_length": results["config"]["sequence_length"],
        "selected_features": selected_features,
        "model_config": {
            'hidden_dim': results["model_config"]["hidden_dim"],
            'lstm_layers': results["model_config"]["lstm_layers"],
            'dense_dims': results["model_config"]["dense_dims"],
            'dropout': results["model_config"]["dropout"],
            'use_attention': results["model_config"]["use_attention"],
            'bidirectional': results["model_config"]["bidirectional"],
            'num_classes': results["model_config"]["num_classes"],
            'use_regression': results["model_config"]["use_regression"]
        },
        "ensemble_size": 1,
        "symbol": symbol,
        "timeframe": timeframe
    }
    
    # Guardar meta.json
    meta_path = base_dir / "meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"   ✅ meta.json actualizado")
    return True

def main():
    print("🔧 Generando meta.json con features correctas...\\n")
    
    for symbol, tf in [("ETHUSDT", "15m"), ("XRPUSDT", "15m"), ("LTCUSDT", "15m")]:
        print(f"📦 {symbol} {tf}...")
        fix_meta_json(symbol, tf)
    
    print("\\n✅ Completado! Reinicia el servicio Flask.")

if __name__ == "__main__":
    main()
