#!/usr/bin/env python3
import sys
from pathlib import Path
import json
import shutil
from datetime import datetime
import argparse

# Configuración
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models" / "advanced"
DEPRECATED_DIR = MODELS_DIR / "deprecated"

def load_metrics(model_path):
    """Carga métricas clave de un directorio de modelo."""
    metrics = {}
    
    # 1. Intentar cargar optimal_threshold.json (Mejor fuente)
    thr_path = model_path / "optimal_threshold.json"
    if thr_path.exists():
        try:
            data = json.loads(thr_path.read_text())
            bt = data.get('backtest_metrics', {})
            metrics = {
                'sharpe': bt.get('sharpe', 0),
                'pnl': bt.get('pnl', 0),
                'win_rate': bt.get('win_rate', 0),
                'trades': bt.get('n_trades', 0),
                'threshold': data.get('threshold', 0),
                'source': 'optimal_threshold.json'
            }
            return metrics
        except:
            pass

    # 2. Intentar cargar meta.json (Fuente secundaria)
    meta_path = model_path / "meta.json"
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text())
            # A veces meta.json tiene métricas de test
            test_metrics = data.get('test_metrics', {})
            metrics = {
                'sharpe': 0, # meta.json usualmente no tiene sharpe calculado
                'pnl': 0,
                'win_rate': test_metrics.get('accuracy', 0) * 100, # Approx
                'trades': 0,
                'threshold': 0.5,
                'source': 'meta.json'
            }
            return metrics
        except:
            pass
            
    return None

def compare_models(symbol, timeframe, new_base_dir):
    """Compara modelo viejo vs nuevo."""
    old_path = MODELS_DIR / symbol / timeframe
    new_path = new_base_dir / symbol / timeframe
    
    print(f"\n🔍 Analizando {symbol} [{timeframe}]...")
    
    old_metrics = load_metrics(old_path)
    new_metrics = load_metrics(new_path)
    
    if not new_metrics:
        print(f"❌ No se encontraron métricas para el NUEVO modelo en {new_path}")
        return None
        
    if not old_metrics:
        print(f"⚠️ No se encontraron métricas para el VIEJO modelo en {old_path} (Se asumirá inferior)")
        old_metrics = {'sharpe': -999, 'pnl': -999, 'win_rate': 0, 'trades': 0}

    # Tabla Comparativa
    print(f"\n   {'Métrica':<15} | {'VIEJO':<15} | {'NUEVO':<15} | {'Diferencia':<15}")
    print(f"   {'-'*15}-+-{'-'*15}-+-{'-'*15}-+-{'-'*15}")
    
    metrics_to_compare = ['sharpe', 'pnl', 'win_rate', 'trades']
    
    better = False
    if new_metrics['sharpe'] > old_metrics['sharpe']:
        better = True
        
    for m in metrics_to_compare:
        old_v = old_metrics.get(m, 0)
        new_v = new_metrics.get(m, 0)
        diff = new_v - old_v
        
        # Formato
        if m in ['pnl', 'win_rate']:
            old_s = f"{old_v:.2f}%"
            new_s = f"{new_v:.2f}%"
            diff_s = f"{diff:+.2f}%"
        elif m == 'trades':
            old_s = str(old_v)
            new_s = str(new_v)
            diff_s = f"{diff:+d}"
        else:
            old_s = f"{old_v:.4f}"
            new_s = f"{new_v:.4f}"
            diff_s = f"{diff:+.4f}"
            
        print(f"   {m.capitalize():<15} | {old_s:<15} | {new_s:<15} | {diff_s:<15}")
        
    print(f"\n   🏆 Veredicto: {'NUEVO ES MEJOR' if better else 'VIEJO ES MEJOR (O IGUAL)'}")
    return better

def migrate_model(symbol, timeframe, new_base_dir):
    """Mueve el viejo a deprecated y el nuevo a prod."""
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    dep_path = DEPRECATED_DIR / timestamp / symbol / timeframe
    old_path = MODELS_DIR / symbol / timeframe
    new_path = new_base_dir / symbol / timeframe
    
    print(f"\n📦 Migrando {symbol} {timeframe}...")
    
    # 1. Crear backup del viejo
    if old_path.exists():
        print(f"   ➡️  Moviendo viejo a: {dep_path}")
        dep_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(dep_path))
        
    # 1.1 Backup de thresholds_config.json global si existe (solo una vez por sesión de migración idealmente, pero mejor prevenir)
    # Lo guardaremos en la raíz de la carpeta deprecated con timestamp
    threshold_config_path = MODELS_DIR / "thresholds_config.json"
    if threshold_config_path.exists():
        dep_thr_path = DEPRECATED_DIR / timestamp / "thresholds_config.json"
        dep_thr_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(threshold_config_path), str(dep_thr_path))
        print(f"   ➡️  Backup de thresholds_config.json guardado en: {dep_thr_path}")
    
    # 2. Mover nuevo a prod
    print(f"   ➡️  Instalando nuevo en: {old_path}")
    old_path.parent.mkdir(parents=True, exist_ok=True)
    # Copiar en lugar de mover para mantener el experimento? No, mover es mejor para limpiar.
    # Pero el usuario dijo "mover a carpeta deprecada".
    shutil.copytree(str(new_path), str(old_path))
    
    print("   ✅ Migración completada.")

def main():
    parser = argparse.ArgumentParser(description="Comparador y Migrador de Modelos ML")
    parser.add_argument('--new-dir', required=True, help="Directorio base de los nuevos modelos (ej: models/advanced/models_2025_12_19)")
    parser.add_argument('--symbol', required=True, help="Símbolo a comparar")
    parser.add_argument('--timeframe', default="1h", help="Timeframe")
    parser.add_argument('--migrate', action='store_true', help="Ejecutar migración si el nuevo es mejor")
    
    args = parser.parse_args()
    
    new_base = Path(args.new_dir)
    
    is_better = compare_models(args.symbol, args.timeframe, new_base)
    
    if args.migrate:
        if is_better:
            migrate_model(args.symbol, args.timeframe, new_base)
        else:
            print("\n⚠️  No se migró porque el modelo nuevo NO superó al viejo (o no se encontraron métricas).")
            print("    Usa --force (no implementado) si realmente quieres forzarlo.")

if __name__ == "__main__":
    main()
