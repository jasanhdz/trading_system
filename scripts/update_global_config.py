#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def update_config(symbol, timeframe, model_dir):
    root = Path(__file__).parents[1]
    config_path = root / "models/advanced/thresholds_config.json"
    model_path = Path(model_dir)
    optimal_path = model_path / "optimal_threshold.json"
    
    if not optimal_path.exists():
        print(f"❌ No se encontró optimal_threshold.json en {model_dir}")
        return

    optimal = json.loads(optimal_path.read_text())
    
    # Cargar config global
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = {}
        
    key = f"{symbol}_{timeframe}"
    
    # Actualizar valores
    if key not in config:
        config[key] = {}
        
    config[key]['threshold'] = optimal['threshold']
    
    # Actualizar métricas informativas
    metrics = optimal.get('backtest_metrics', {})
    config[key]['pnl'] = metrics.get('pnl', 0)
    config[key]['trades'] = metrics.get('n_trades', 0)
    config[key]['sharpe'] = metrics.get('sharpe', 0)
    config[key]['win_rate'] = metrics.get('win_rate', 0)
    
    # Guardar
    config_path.write_text(json.dumps(config, indent=2))
    print(f"✅ Configuración global actualizada para {key}")
    print(f"   Threshold: {optimal['threshold']}")
    print(f"   Sharpe: {metrics.get('sharpe', 0):.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso: python3 update_global_config.py <symbol> <timeframe> <model_dir>")
        sys.exit(1)
    
    update_config(sys.argv[1], sys.argv[2], sys.argv[3])
