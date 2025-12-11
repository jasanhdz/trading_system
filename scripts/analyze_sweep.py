#!/usr/bin/env python3
"""
Analiza los resultados del hyperparameter sweep y genera reporte.
"""
import json
from pathlib import Path
import sys

def analyze_sweep(sweep_file: Path):
    """Analiza un archivo de sweep y genera reporte."""
    
    if not sweep_file.exists():
        print(f"❌ No se encontró {sweep_file}")
        return
    
    with open(sweep_file) as f:
        results = json.load(f)
    
    if not results:
        print("❌ No hay resultados en el sweep")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 ANÁLISIS DE SWEEP: {sweep_file.name}")
    print(f"{'='*80}\n")
    
    print(f"Total de experimentos: {len(results)}\n")
    
    # Estadísticas generales
    avg_f1 = sum(r['metrics']['avg_test_f1'] for r in results) / len(results)
    avg_long_f1 = sum(r['metrics']['avg_long_f1'] for r in results) / len(results)
    avg_short_f1 = sum(r['metrics']['avg_short_f1'] for r in results) / len(results)
    
    print(f"📈 Promedios Generales:")
    print(f"  Macro F1: {avg_f1:.3f}")
    print(f"  Long F1: {avg_long_f1:.3f}")
    print(f"  Short F1: {avg_short_f1:.3f}\n")
    
    # Filtrar resultados prometedores
    promising = [r for r in results if 
                 r['metrics']['avg_long_f1'] > 0.30 and 
                 r['metrics']['avg_short_f1'] > 0.30]
    
    print(f"🎯 Configuraciones Prometedoras (Long F1 > 0.30 AND Short F1 > 0.30): {len(promising)}")
    
    if promising:
        # Ordenar por Macro F1
        promising_sorted = sorted(promising, key=lambda r: r['metrics']['avg_test_f1'], reverse=True)
        
        print("\n🏆 Top 3 Configuraciones:")
        for i, r in enumerate(promising_sorted[:3], 1):
            cfg = r['config']
            met = r['metrics']
            print(f"\n{i}. Config:")
            print(f"   Target Return: {cfg['target_return']:.4f} ({cfg['target_return']*100:.2f}%)")
            print(f"   Prediction Horizon: {cfg['prediction_horizon']} períodos")
            print(f"   Hidden Dim: {cfg['hidden_dim']}")
            print(f"   Dropout: {cfg['dropout']}")
            print(f"   Learning Rate: {cfg['lr']}")
            print(f"   Métricas:")
            print(f"     Macro F1: {met['avg_test_f1']:.3f}")
            print(f"     Long F1: {met['avg_long_f1']:.3f}")
            print(f"     Short F1: {met['avg_short_f1']:.3f}")
            print(f"     Accuracy: {met['avg_test_accuracy']:.3f}")
            if met.get('avg_pnl'):
                print(f"     PnL Implícito: {met['avg_pnl']:.4f}")
        
        # Recomendación
        best = promising_sorted[0]
        print(f"\n{'='*80}")
        print("💡 RECOMENDACIÓN")
        print(f"{'='*80}\n")
        print("Usa esta configuración para el entrenamiento final:")
        print(f"""
./venv_rocm62/bin/python scripts/train_production_ready.py \\
    --symbol {best['config']['symbol']} \\
    --timeframe {best['config']['timeframe']} \\
    --target-return {best['config']['target_return']} \\
    --prediction-horizon {best['config']['prediction_horizon']} \\
    --hidden-dim {best['config']['hidden_dim']} \\
    --dropout {best['config']['dropout']} \\
    --lr {best['config']['lr']} \\
    --epochs 200
""")
    else:
        print("\n⚠️  NINGUNA configuración cumple los criterios mínimos (Long/Short F1 > 0.30)")
        print("\nOpciones:")
        print("1. Ejecutar sweep 'balanced' o 'thorough' para explorar más")
        print("2. Revisar MODEL_IMPROVEMENT_PLAN.md para estrategias alternativas")
        print("3. Considerar feature engineering o cambio de arquitectura")
    
    # Buscar peores resultados
    print(f"\n{'='*80}")
    print("🔴 Configuraciones Peores (Para Evitar)")
    print(f"{'='*80}\n")
    
    worst = sorted(results, key=lambda r: r['metrics']['avg_test_f1'])[:3]
    for i, r in enumerate(worst, 1):
        cfg = r['config']
        met = r['metrics']
        print(f"\n{i}. Config:")
        print(f"   Target Return: {cfg['target_return']:.4f}")
        print(f"   Prediction Horizon: {cfg['prediction_horizon']}")
        print(f"   Macro F1: {met['avg_test_f1']:.3f} ❌")
    
    # Análisis por parámetro
    print(f"\n{'='*80}")
    print("📊 Análisis por Parámetro")
    print(f"{'='*80}\n")
    
    # Agrupar por target_return
    by_target = {}
    for r in results:
        tr = r['config']['target_return']
        if tr not in by_target:
            by_target[tr] = []
        by_target[tr].append(r['metrics']['avg_test_f1'])
    
    print("Por Target Return:")
    for tr in sorted(by_target.keys()):
        avg = sum(by_target[tr]) / len(by_target[tr])
        print(f"  {tr:.4f} ({tr*100:.2f}%): Avg F1 = {avg:.3f}")
    
    # Agrupar por prediction_horizon
    by_horizon = {}
    for r in results:
        ph = r['config']['prediction_horizon']
        if ph not in by_horizon:
            by_horizon[ph] = []
        by_horizon[ph].append(r['metrics']['avg_test_f1'])
    
    print("\nPor Prediction Horizon:")
    for ph in sorted(by_horizon.keys()):
        avg = sum(by_horizon[ph]) / len(by_horizon[ph])
        print(f"  {ph} períodos: Avg F1 = {avg:.3f}")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analizar resultados de sweep")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--timeframe", default="15m", help="Timeframe")
    
    args = parser.parse_args()
    
    sweep_file = Path(f"experiments/{args.symbol}_{args.timeframe}_sweep.json")
    analyze_sweep(sweep_file)
