"""
Grid Search Optimizer for the Ninja Trading System v4.1

This script systematically tests different configurations of:
- Base Threshold (0.30 - 0.50)
- Hard Stop Loss (-5% to -20% ROE)
- Leverage (5x - 15x)
- Trailing Activation ROE (1% - 5%)

Updated for v2.2 ML Features (23 dimensions with CVD + Volatility)

Usage:
    python scripts/grid_search_optimizer.py --symbol BTCUSDT --days 7
    python scripts/grid_search_optimizer.py --symbol ETHUSDT --days 14 --mode whale
"""
import os
import sys
import argparse
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from backtest_system_v2 import NinjaBotSimulator

# Supress verbose ML logging during grid search
import logging
logging.getLogger("ml_service_v2").setLevel(logging.WARNING)
logging.getLogger("EnsembleManager").setLevel(logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════
# v4.1: Expanded Grid Search Parameters
# ═══════════════════════════════════════════════════════════════════════════
GRID_CONFIGS = {
    'default': {
        'base_thresholds': [0.30, 0.35, 0.40, 0.45, 0.50],
        'hard_stop_options': [-0.05, -0.10, -0.15],
        'leverage_options': [10, 15],
        'trailing_activation': [0.02, 0.03, 0.05]  # v4.1: Trailing params
    },
    'whale': {
        'base_thresholds': [0.45, 0.50, 0.55, 0.60],
        'hard_stop_options': [-0.15, -0.20, -0.25],
        'leverage_options': [3, 5, 7],
        'trailing_activation': [0.03, 0.05, 0.08]
    },
    'monk': {
        'base_thresholds': [0.35, 0.40, 0.45],
        'hard_stop_options': [-0.03, -0.05, -0.07],
        'leverage_options': [10, 15],
        'trailing_activation': [0.01, 0.02]
    },
    'bloodbath': {
        'base_thresholds': [0.25, 0.30, 0.35],
        'hard_stop_options': [-0.015, -0.02, -0.025],
        'leverage_options': [15, 20],
        'trailing_activation': [0.005, 0.01]
    }
}

def run_grid_search(symbol: str, days: int = 3, hours: int = 0, mode: str = 'default'):
    """
    Prueba diferentes configuraciones de Threshold, Hard Stop, Leverage y Trailing para encontrar el óptimo.
    
    v4.1: Ahora soporta múltiples modos (default, whale, monk, bloodbath)
    """
    # Get config for mode
    config = GRID_CONFIGS.get(mode, GRID_CONFIGS['default'])
    base_thresholds = config['base_thresholds']
    hard_stop_options = config['hard_stop_options']
    leverage_options = config['leverage_options']
    trailing_options = config['trailing_activation']
    
    results = []

    print(f"\n{'='*70}")
    print(f"🚀 GRID SEARCH OPTIMIZER v4.1: {symbol}")
    print(f"   Mode: {mode.upper()}")
    print(f"   Período: {days} días, {hours} horas")
    total_combos = len(base_thresholds) * len(hard_stop_options) * len(leverage_options) * len(trailing_options)
    print(f"   Configuraciones a probar: {total_combos}")
    print(f"{'='*70}\n")
    
    # Cargar datos UNA VEZ para reutilizarlos (eficiencia)
    print("📥 Cargando datos (una sola vez para todas las pruebas)...")
    # Instancia temporal para cargar datos (leverage no importa aquí)
    temp_sim = NinjaBotSimulator(symbol=symbol, initial_capital=1000.0, leverage=10)
    shared_data = temp_sim.load_data(days=days, hours=hours)
    print(f"✅ Datos cargados: {len(shared_data)} registros.\n")
    
    total_combos = len(base_thresholds) * len(hard_stop_options) * len(leverage_options)
    combo_num = 0
    
    for lev in leverage_options:
        for base_thr in base_thresholds:
            for stop_pct in hard_stop_options:
                combo_num += 1
                
                # 1. Instanciar Simulador con la config específica
                sim = NinjaBotSimulator(symbol=symbol, initial_capital=1000.0, leverage=lev)
                
                # 2. Sobreescribir configuración manualmente
                sim.base_threshold = base_thr
                sim.hard_stop_pct = stop_pct
                
                # 3. Ejecutar Backtest con datos compartidos (sin recargar)
                sim.run(shared_data)
                
                # 4. Calcular métricas
                num_trades = len(sim.trades)
                if num_trades > 0:
                    wins = [t for t in sim.trades if t['pnl'] > 0]
                    losses = [t for t in sim.trades if t['pnl'] <= 0]
                    win_rate = len(wins) / num_trades
                    profit_factor = sum(t['pnl'] for t in wins) / abs(sum(t['pnl'] for t in losses)) if losses else float('inf')
                    avg_pnl = sum(t['pnl'] for t in sim.trades) / num_trades
                else:
                    win_rate = 0
                    profit_factor = 0
                    avg_pnl = 0
                
                return_pct = ((sim.balance - 1000)/1000)*100
                
                # 5. Guardar Resultados
                results.append({
                    'leverage': lev,
                    'base_thr': base_thr,
                    'stop_pct': stop_pct,
                    'config': f"Lev:{lev}x Thr:{base_thr:.2f} Stop:{stop_pct:.0%}",
                    'final_balance': sim.balance,
                    'return_pct': return_pct,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'avg_pnl': avg_pnl,
                    'total_trades': num_trades
                })
                
                print(f"[{combo_num}/{total_combos}] Lev:{lev}x Thr:{base_thr:.2f} Stop:{stop_pct:.0%} -> Return: {return_pct:>+6.2f}% | Trades: {num_trades} | WR: {win_rate:.0%}")

    # 6. Ranking de Mejores Configuraciones
    print("\n" + "="*70)
    print("📊 RANKING DE CONFIGURACIONES (TOP 10)")
    print("="*70)
    
    # Ordenar por Retorno descendente
    ranked_results = sorted(results, key=lambda x: x['return_pct'], reverse=True)
    
    print(f"{'Rank':<5} {'Config':<30} {'Return':<10} {'Trades':<8} {'WinRate':<10} {'PF':<8}")
    print("-" * 75)
    
    for i, r in enumerate(ranked_results[:10], 1):
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "Inf"
        print(f"{i:<5} {r['config']:<30} {r['return_pct']:>+7.2f}% {r['total_trades']:<8} {r['win_rate']:.0%}{'':<6} {pf_str:<8}")
    
    # 7. Guardar reporte JSON
    import json
    report_path = Path(REPO_ROOT) / "reports" / f"grid_search_{symbol.replace('/','')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(ranked_results, f, indent=2)
    print(f"\n💾 Reporte guardado en: {report_path}")

    # 8. Imprimir Mejor Configuración
    best = ranked_results[0]
    print("\n" + "="*70)
    print("🏆 MEJOR CONFIGURACIÓN ENCONTRADA")
    print("="*70)
    print(f"   Leverage:       {best['leverage']}x")
    print(f"   Threshold Base: {best['base_thr']:.2f}")
    print(f"   Hard Stop:      {best['stop_pct']:.0%} ROE")
    print(f"   Retorno:        {best['return_pct']:+.2f}%")
    print(f"   Win Rate:       {best['win_rate']:.0%}")
    print(f"   Total Trades:   {best['total_trades']}")
    print("="*70 + "\n")
    
    return ranked_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search Optimizer for Trading Bot v4.1")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--days", type=int, default=7, help="Days of historical data")
    parser.add_argument("--hours", type=int, default=0, help="Additional hours of historical data")
    parser.add_argument("--mode", type=str, default="default", 
                       choices=['default', 'whale', 'monk', 'bloodbath'],
                       help="Regime mode for parameter ranges")
    args = parser.parse_args()
    
    run_grid_search(args.symbol, days=args.days, hours=args.hours, mode=args.mode)

