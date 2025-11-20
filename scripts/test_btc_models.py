#!/usr/bin/env python3
"""
Script para entrenar y comparar modelos en BTC 5m y 15m.

Este script facilita probar diferentes configuraciones y
comparar resultados entre timeframes.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import subprocess
import json
from datetime import datetime
import click

from utils.logger import setup_logger

logger = setup_logger("btc_tester")

MODEL_DIR = (REPO_ROOT / "models" / "advanced").resolve()


def run_command(cmd, description):
    """Ejecuta comando y muestra progreso."""
    logger.info(f"\n{'='*80}")
    logger.info(f"▶️  {description}")
    logger.info(f"{'='*80}")
    logger.info(f"Comando: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=REPO_ROOT)

    if result.returncode != 0:
        logger.error(f"❌ Error en: {description}")
        return False

    logger.info(f"\n✅ Completado: {description}")
    return True


def compare_results(symbol, timeframes):
    """Compara resultados entre timeframes."""

    logger.info(f"\n{'='*80}")
    logger.info("📊 COMPARACIÓN DE RESULTADOS")
    logger.info(f"{'='*80}\n")

    results = {}

    for tf in timeframes:
        model_dir = MODEL_DIR / symbol / tf

        # Buscar archivo de resultados
        result_files = [
            model_dir / "production_training_results.json",
            model_dir / "ensemble" / "ensemble_metadata.json",
            model_dir / "ensemble" / "hybrid" / "hybrid_metadata.json",
        ]

        for result_file in result_files:
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                    results[tf] = data
                break

    if not results:
        logger.warning("⚠️  No se encontraron resultados para comparar")
        return

    # Tabla comparativa
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)

    for tf in timeframes:
        if tf not in results:
            continue

        print(f"\n📊 {symbol} {tf}:")
        data = results[tf]

        # Extraer métricas según el tipo de archivo
        if 'results' in data:  # production_training_results.json
            avg_acc = sum(r['test_metrics']['accuracy'] for r in data['results']) / len(data['results'])
            avg_f1 = sum(r['test_metrics']['macro_f1'] for r in data['results']) / len(data['results'])
            avg_long_f1 = sum(r['test_metrics']['long_f1'] for r in data['results']) / len(data['results'])
            avg_short_f1 = sum(r['test_metrics']['short_f1'] for r in data['results']) / len(data['results'])

            print(f"  Tipo: Modelo Individual")
            print(f"  Accuracy: {avg_acc:.4f}")
            print(f"  Macro F1: {avg_f1:.4f}")
            print(f"  Long F1:  {avg_long_f1:.4f}")
            print(f"  Short F1: {avg_short_f1:.4f}")

        elif 'avg_val_f1' in data:  # ensemble_metadata.json
            print(f"  Tipo: Ensemble Neural")
            print(f"  N modelos: {data['n_models']}")
            print(f"  Avg Val F1: {data['avg_val_f1']:.4f}")
            print(f"  Best Val F1: {data['best_val_f1']:.4f}")

        elif 'test_metrics' in data:  # hybrid_metadata.json
            metrics = data['test_metrics']
            print(f"  Tipo: Ensemble Híbrido")
            print(f"  Neural F1:  {metrics['neural_f1']:.4f}")
            print(f"  XGBoost F1: {metrics['xgb_f1']:.4f}")
            print(f"  Hybrid F1:  {metrics['hybrid_f1']:.4f}")
            print(f"  Mejora: {metrics['improvement_pct']:+.2f}%")

    print("\n" + "="*80)


@click.command()
@click.option("--mode",
              type=click.Choice(['single', 'ensemble', 'hybrid', 'all']),
              default='single',
              help="Modo de entrenamiento")
@click.option("--timeframes",
              default="5m,15m",
              help="Timeframes separados por coma (ej: 5m,15m)")
@click.option("--symbol",
              default="BTCUSDT",
              help="Símbolo a entrenar")
@click.option("--device",
              default="cuda",
              help="Device (cuda/cpu)")
@click.option("--epochs",
              default=200,
              help="Epochs para modelo individual")
@click.option("--ensemble-epochs",
              default=150,
              help="Epochs para cada modelo del ensemble")
@click.option("--n-models",
              default=5,
              help="Número de modelos en ensemble")
def main(
    mode: str,
    timeframes: str,
    symbol: str,
    device: str,
    epochs: int,
    ensemble_epochs: int,
    n_models: int,
):
    """
    Entrena y prueba modelos en BTC.

    Ejemplos:

    \b
    # Entrenar modelo individual en ambos timeframes:
    python scripts/test_btc_models.py --mode single --timeframes 5m,15m

    \b
    # Entrenar ensemble solo en 5m:
    python scripts/test_btc_models.py --mode ensemble --timeframes 5m

    \b
    # Pipeline completo (single + ensemble + hybrid) en 15m:
    python scripts/test_btc_models.py --mode all --timeframes 15m

    \b
    # Solo hybrid (requiere ensemble previo):
    python scripts/test_btc_models.py --mode hybrid --timeframes 5m
    """

    timeframes_list = [tf.strip() for tf in timeframes.split(',')]

    print("\n" + "="*80)
    print(f"🚀 ENTRENAMIENTO DE MODELOS: {symbol}")
    print("="*80)
    print(f"\nModo: {mode}")
    print(f"Timeframes: {', '.join(timeframes_list)}")
    print(f"Device: {device}")
    print(f"Epochs (single): {epochs}")
    print(f"Epochs (ensemble): {ensemble_epochs}")
    print(f"N modelos (ensemble): {n_models}")

    start_time = datetime.now()

    for tf in timeframes_list:
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"# PROCESANDO {symbol} {tf}")
        logger.info(f"{'#'*80}")

        # 1. Modelo Individual
        if mode in ['single', 'all']:
            cmd = [
                "python", "scripts/train_production_ready.py",
                "--symbol", symbol,
                "--timeframe", tf,
                "--epochs", str(epochs),
                "--device", device,
            ]

            success = run_command(
                cmd,
                f"Entrenamiento individual {symbol} {tf}"
            )

            if not success and mode == 'all':
                logger.error("❌ Entrenamiento individual falló. Abortando.")
                continue

        # 2. Ensemble
        if mode in ['ensemble', 'all']:
            cmd = [
                "python", "scripts/train_ensemble.py",
                "--symbol", symbol,
                "--timeframe", tf,
                "--n-models", str(n_models),
                "--epochs", str(ensemble_epochs),
                "--device", device,
            ]

            success = run_command(
                cmd,
                f"Entrenamiento ensemble {symbol} {tf}"
            )

            if not success and mode == 'all':
                logger.error("❌ Entrenamiento ensemble falló. Abortando.")
                continue

        # 3. Híbrido (requiere ensemble)
        if mode in ['hybrid', 'all']:
            ensemble_dir = MODEL_DIR / symbol / tf / "ensemble"

            if not ensemble_dir.exists():
                logger.warning(
                    f"⚠️  No existe ensemble en {ensemble_dir}\n"
                    f"   Ejecuta primero: --mode ensemble"
                )
                continue

            cmd = [
                "python", "scripts/train_hybrid_ensemble.py",
                "--symbol", symbol,
                "--timeframe", tf,
                "--ensemble-dir", str(ensemble_dir),
                "--device", device,
            ]

            success = run_command(
                cmd,
                f"Entrenamiento híbrido {symbol} {tf}"
            )

    # Comparar resultados
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60

    logger.info(f"\n{'='*80}")
    logger.info(f"⏱️  Tiempo total: {duration:.1f} minutos")
    logger.info(f"{'='*80}")

    compare_results(symbol, timeframes_list)

    # Recomendaciones
    print("\n" + "="*80)
    print("📝 PRÓXIMOS PASOS")
    print("="*80)
    print("\n1. Revisar métricas:")
    for tf in timeframes_list:
        print(f"   - models/advanced/{symbol}/{tf}/")

    print("\n2. Comparar modelos:")
    print("   - ¿Qué timeframe tiene mejor F1?")
    print("   - ¿El ensemble mejora sobre modelo individual?")
    print("   - ¿El híbrido supera al ensemble neural?")

    print("\n3. Si los resultados son buenos (F1 > 0.45):")
    print("   - Paper trading por 2 semanas")
    print("   - Monitorear Sharpe ratio y drawdown")

    print("\n4. Si los resultados son débiles (F1 < 0.40):")
    print("   - Probar con target_return más alto (0.008)")
    print("   - Aumentar sequence_length (96)")
    print("   - Considerar solo el mejor timeframe")

    print("\n5. Para producción:")
    print("   - Reduce leverage a 10-20x")
    print("   - Implementa circuit breakers")
    print("   - Setup monitoreo (Grafana)")

    print("\n" + "="*80)
    print("✅ PROCESO COMPLETADO")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
