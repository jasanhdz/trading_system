#!/usr/bin/env python3
"""
Analyze completed models and recommend which to use in production vs re-train.
"""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelMetrics:
    symbol: str
    timeframe: str
    avg_long_f1: float
    avg_short_f1: float
    avg_accuracy: float
    min_short_f1: float
    max_short_f1: float
    num_folds: int

def analyze_model(path: Path) -> Optional[ModelMetrics]:
    """Extract metrics from a production_training_results.json file."""
    if not path.exists():
        return None
    
    with open(path) as f:
        data = json.load(f)
    
    results = data.get('results', [])
    if not results:
        return None
    
    symbol = path.parent.parent.name
    timeframe = path.parent.name
    
    long_f1s = [r['test_metrics']['long_f1'] for r in results]
    short_f1s = [r['test_metrics']['short_f1'] for r in results]
    accuracies = [r['test_metrics']['accuracy'] for r in results]
    
    return ModelMetrics(
        symbol=symbol,
        timeframe=timeframe,
        avg_long_f1=sum(long_f1s) / len(long_f1s),
        avg_short_f1=sum(short_f1s) / len(short_f1s),
        avg_accuracy=sum(accuracies) / len(accuracies),
        min_short_f1=min(short_f1s),
        max_short_f1=max(short_f1s),
        num_folds=len(results)
    )

def classify_model(m: ModelMetrics) -> str:
    """Classify model as READY, RETRAIN, or POOR based on metrics."""
    # Criterios:
    # READY: Long F1 > 0.45 AND Short F1 > 0.35
    # RETRAIN: Puede mejorar con Focal Loss (Short F1 < 0.35)
    # POOR: Long F1 < 0.40 (necesita revisión profunda)
    
    if m.avg_long_f1 < 0.40:
        return "POOR"
    elif m.avg_short_f1 < 0.35:
        return "RETRAIN"
    elif m.avg_long_f1 >= 0.45 and m.avg_short_f1 >= 0.35:
        return "READY"
    else:
        return "RETRAIN"

def main():
    base = Path('models/advanced')
    
    # Recopilar todos los modelos
    all_models: List[ModelMetrics] = []
    
    for symbol_dir in sorted(base.iterdir()):
        if not symbol_dir.is_dir():
            continue
        
        for tf in ['5m', '15m']:
            result_file = symbol_dir / tf / 'production_training_results.json'
            metrics = analyze_model(result_file)
            if metrics:
                all_models.append(metrics)
    
    # Agrupar por símbolo
    symbols_with_both = {}
    symbols_15m_only = {}
    
    for m in all_models:
        if m.symbol not in symbols_with_both and m.symbol not in symbols_15m_only:
            # Check if this symbol has both timeframes
            has_5m = any(x.symbol == m.symbol and x.timeframe == '5m' for x in all_models)
            has_15m = any(x.symbol == m.symbol and x.timeframe == '15m' for x in all_models)
            
            if has_5m and has_15m:
                symbols_with_both[m.symbol] = {'5m': None, '15m': None}
            else:
                symbols_15m_only[m.symbol] = None
    
    # Asignar modelos
    for m in all_models:
        if m.symbol in symbols_with_both:
            symbols_with_both[m.symbol][m.timeframe] = m
        else:
            symbols_15m_only[m.symbol] = m
    
    print("=" * 80)
    print("ANÁLISIS DE MODELOS COMPLETADOS")
    print("=" * 80)
    
    # Modelos con ambos timeframes
    print(f"\n📊 SÍMBOLOS CON AMBOS TIMEFRAMES (5m + 15m): {len(symbols_with_both)}")
    print("-" * 80)
    
    for symbol in sorted(symbols_with_both.keys()):
        m_5m = symbols_with_both[symbol]['5m']
        m_15m = symbols_with_both[symbol]['15m']
        
        print(f"\n{symbol}:")
        for m in [m_5m, m_15m]:
            status = classify_model(m)
            emoji = "✅" if status == "READY" else "🔄" if status == "RETRAIN" else "⚠️"
            print(f"  {emoji} {m.timeframe:3s}: Long F1={m.avg_long_f1:.3f} | Short F1={m.avg_short_f1:.3f} | Acc={m.avg_accuracy:.3f} | [{status}]")
    
    # Modelos solo 15m
    print(f"\n\n📊 SÍMBOLOS SOLO 15m: {len(symbols_15m_only)}")
    print("-" * 80)
    
    for symbol in sorted(symbols_15m_only.keys()):
        m = symbols_15m_only[symbol]
        status = classify_model(m)
        emoji = "✅" if status == "READY" else "🔄" if status == "RETRAIN" else "⚠️"
        print(f"{emoji} {symbol:10s}: Long F1={m.avg_long_f1:.3f} | Short F1={m.avg_short_f1:.3f} | Acc={m.avg_accuracy:.3f} | [{status}]")
    
    # Resumen de recomendaciones
    print(f"\n\n{'=' * 80}")
    print("RECOMENDACIONES DE ACCIÓN")
    print("=" * 80)
    
    ready_models = [m for m in all_models if classify_model(m) == "READY"]
    retrain_models = [m for m in all_models if classify_model(m) == "RETRAIN"]
    poor_models = [m for m in all_models if classify_model(m) == "POOR"]
    
    print(f"\n✅ LISTOS PARA PRODUCCIÓN ({len(ready_models)} modelos):")
    print("   Usar inmediatamente sin cambios")
    for m in ready_models:
        print(f"   - {m.symbol} {m.timeframe}")
    
    print(f"\n🔄 RE-ENTRENAR CON MEJORAS ({len(retrain_models)} modelos):")
    print("   Usar Focal Loss + Nuevas Features para mejorar Short F1")
    for m in retrain_models:
        print(f"   - {m.symbol} {m.timeframe} (Short F1: {m.avg_short_f1:.3f} → estimado 0.45+)")
    
    if poor_models:
        print(f"\n⚠️  REQUIEREN REVISIÓN ({len(poor_models)} modelos):")
        print("   Long F1 muy bajo, puede ser problema de datos o configuración")
        for m in poor_models:
            print(f"   - {m.symbol} {m.timeframe} (Long F1: {m.avg_long_f1:.3f})")
    
    print(f"\n{'=' * 80}\n")

if __name__ == "__main__":
    main()
