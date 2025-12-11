#!/usr/bin/env python3
"""
Hyperparameter Sweep para Trading Models

Experimenta con diferentes configuraciones para encontrar la mejor combinación.
Enfoque: maximizar métricas de trading (PnL, Sharpe), no solo F1-Score.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import json
import itertools
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import subprocess
import os


def parse_csv_floats(value: str) -> List[float]:
    """Parsea una lista separada por comas a floats."""
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_csv_ints(value: str) -> List[int]:
    """Parsea una lista separada por comas a ints."""
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def apply_filter(options: List, allowed: Optional[List], name: str) -> List:
    """Restringe opciones si se provee un filtro."""
    if allowed is None:
        return options
    filtered = [o for o in options if o in allowed]
    if not filtered:
        raise ValueError(f"Filtro para {name} vacío; revisa valores permitidos: {allowed}")
    return filtered

@dataclass
class ExperimentConfig:
    """Configuración de un experimento."""
    symbol: str
    timeframe: str
    target_return: float
    prediction_horizon: int
    sequence_length: int
    hidden_dim: int
    lstm_layers: int
    dropout: float
    lr: float
    batch_size: int
    gap_multiplier: int  # Multiplicador para el gap (ej. 2 = 2*prediction_horizon)
    
    def to_dict(self):
        return asdict(self)
    
    def get_name(self):
        """Nombre único para este experimento."""
        return (f"{self.symbol}_{self.timeframe}_"
                f"tr{self.target_return}_ph{self.prediction_horizon}_"
                f"hd{self.hidden_dim}_dr{self.dropout}_lr{self.lr}")

@dataclass
class ExperimentResult:
    """Resultados de un experimento."""
    config: ExperimentConfig
    avg_test_f1: float
    avg_test_accuracy: float
    avg_long_f1: float
    avg_short_f1: float
    avg_pnl: float = 0.0  # Si está disponible
    avg_mse: float = 0.0  # Si está disponible
    
def generate_experiments(
    symbol: str,
    timeframe: str,
    mode: str = "fast",
    target_returns_filter: Optional[List[float]] = None,
    prediction_horizons_filter: Optional[List[int]] = None,
    sequence_lengths_filter: Optional[List[int]] = None,
    hidden_dims_filter: Optional[List[int]] = None,
    dropouts_filter: Optional[List[float]] = None,
    gap_multipliers_filter: Optional[List[int]] = None,
) -> List[ExperimentConfig]:
    """
    Genera grid de experimentos a probar.
    
    Args:
        symbol: Símbolo a entrenar
        timeframe: Timeframe a entrenar
        mode: "fast" (9 exp), "balanced" (27 exp), "thorough" (81 exp)
    
    Prioriza los cambios más impactantes:
    1. Target return y horizonte (afectan balance de clases) ← MÁS IMPORTANTE
    2. Architecture (hidden_dim, dropout)
    3. Training (lr, batch_size)
    """
    
    experiments = []
    
    if mode == "fast":
        # Solo variar objetivo (9 experimentos, ~3-4 horas)
        target_returns = apply_filter([0.002, 0.003, 0.005], target_returns_filter, "target_return")  # 0.2%, 0.3%, 0.5%
        prediction_horizons = apply_filter([3, 4, 6], prediction_horizons_filter, "prediction_horizon")
        
        for tr in target_returns:
            for ph in prediction_horizons:
                exp = ExperimentConfig(
                    symbol=symbol,
                    timeframe=timeframe,
                    target_return=tr,
                    prediction_horizon=ph,
                    sequence_length=48,
                    hidden_dim=192,
                    lstm_layers=3,
                    dropout=0.35,
                    lr=3e-4,
                    batch_size=128,
                    gap_multiplier=1
                )
                experiments.append(exp)
    
    elif mode == "balanced":
        # Variar objetivo + arquitectura básica (27 experimentos, ~8-10 horas)
        target_returns = apply_filter([0.002, 0.003, 0.005], target_returns_filter, "target_return")
        prediction_horizons = apply_filter([3, 4, 6], prediction_horizons_filter, "prediction_horizon")
        hidden_dims = apply_filter([128, 192, 256], hidden_dims_filter, "hidden_dim")
        
        for tr in target_returns:
            for ph in prediction_horizons:
                for hd in hidden_dims:
                    exp = ExperimentConfig(
                        symbol=symbol,
                        timeframe=timeframe,
                        target_return=tr,
                        prediction_horizon=ph,
                        sequence_length=48,
                        hidden_dim=hd,
                        lstm_layers=3,
                        dropout=0.35,
                        lr=3e-4,
                        batch_size=128,
                        gap_multiplier=1
                    )
                    experiments.append(exp)
    
    elif mode == "thorough":
        # Grid completo (81+ experimentos, ~24-30 horas)
        target_returns = apply_filter([0.002, 0.003, 0.005], target_returns_filter, "target_return")
        prediction_horizons = apply_filter([3, 4, 6], prediction_horizons_filter, "prediction_horizon")
        hidden_dims = apply_filter([128, 192, 256], hidden_dims_filter, "hidden_dim")
        dropouts = apply_filter([0.3, 0.35, 0.4], dropouts_filter, "dropout")
        
        for tr in target_returns:
            for ph in prediction_horizons:
                for hd in hidden_dims:
                    for dr in dropouts:
                        exp = ExperimentConfig(
                            symbol=symbol,
                            timeframe=timeframe,
                            target_return=tr,
                            prediction_horizon=ph,
                            sequence_length=48,
                            hidden_dim=hd,
                            lstm_layers=3,
                            dropout=dr,
                            lr=3e-4,
                            batch_size=128,
                            gap_multiplier=1
                        )
                        experiments.append(exp)
    elif mode == "aggressive":
        # Grid agresivo para señales más frecuentes y mayor separación temporal
        target_returns = apply_filter([0.0015, 0.002, 0.0025, 0.003], target_returns_filter, "target_return")
        prediction_horizons = apply_filter([2, 3, 4], prediction_horizons_filter, "prediction_horizon")
        sequence_lengths = apply_filter([48, 64], sequence_lengths_filter, "sequence_length")
        hidden_dims = apply_filter([160, 224], hidden_dims_filter, "hidden_dim")
        dropouts = apply_filter([0.25, 0.30], dropouts_filter, "dropout")
        gap_multipliers = apply_filter([2], gap_multipliers_filter, "gap_multiplier")  # gap = 2 * horizon
        
        for tr in target_returns:
            for ph in prediction_horizons:
                for sl in sequence_lengths:
                    for hd in hidden_dims:
                        for dr in dropouts:
                            for gapm in gap_multipliers:
                                exp = ExperimentConfig(
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    target_return=tr,
                                    prediction_horizon=ph,
                                    sequence_length=sl,
                                    hidden_dim=hd,
                                    lstm_layers=3,
                                    dropout=dr,
                                    lr=3e-4,
                                    batch_size=96,  # más seguro para VRAM 8 GB
                                    gap_multiplier=gapm
                                )
                                experiments.append(exp)
    
    else:
        raise ValueError(f"Mode desconocido: {mode}. Usa 'fast', 'balanced', o 'thorough'")
    
    print(f"📊 Generados {len(experiments)} experimentos (mode={mode})")
    print(f"⏱️  Tiempo estimado: {len(experiments) * 25} minutos (~{len(experiments) * 25 / 60:.1f} horas)")
    return experiments

def run_experiment(config: ExperimentConfig, epochs: int = 50, device: str = "cuda") -> ExperimentResult:
    """
    Ejecuta un experimento y retorna resultados.
    """
    print(f"\n🧪 Ejecutando: {config.get_name()}")
    sys.stdout.flush()
    
    # Crear directorio temporal para este experimento
    exp_dir = REPO_ROOT / "experiments" / config.get_name()
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar configuración
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Preparar comando
    # Guardar los artefactos en un subdirectorio específico para no pisarlos
    model_subdir = REPO_ROOT / "models" / "advanced" / config.symbol / config.timeframe / config.get_name()
    model_subdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,  # Usa el intérprete actual (ROCm o CUDA)
        "scripts/train_production_ready.py",
        "--symbol", config.symbol,
        "--timeframe", config.timeframe,
        "--epochs", str(epochs),
        "--batch-size", str(config.batch_size),
        "--lr", str(config.lr),
        "--sequence-length", str(config.sequence_length),
        "--prediction-horizon", str(config.prediction_horizon),
        "--target-return", str(config.target_return),
        "--hidden-dim", str(config.hidden_dim),
        "--lstm-layers", str(config.lstm_layers),
        "--dropout", str(config.dropout),
        "--device", device,
        # Guardar resultados en subcarpeta específica
        "--model-dir", str(model_subdir),
    ]
    
    # Ejecutar
    env = os.environ.copy()
    if "rocm" in cmd[0]:
        env["HSA_OVERRIDE_GFX_VERSION"] = env.get("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
        # Respetar HIP_VISIBLE_DEVICES si ya viene del entorno; default a 0
        env["HIP_VISIBLE_DEVICES"] = env.get("HIP_VISIBLE_DEVICES", "0")
    
    log_file = exp_dir / "training.log"
    
    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=REPO_ROOT
        )
    
    # Mover artefactos del subdirectorio específico de este experimento
    model_dir = model_subdir
    results_file = model_dir / "production_training_results.json"
    meta_file = model_dir / "meta.json"

    if results_file.exists():
        dest_results = exp_dir / "production_training_results.json"
        dest_meta = exp_dir / "meta.json"
        dest_results.write_text(results_file.read_text())
        if meta_file.exists():
            dest_meta.write_text(meta_file.read_text())
        else:
            print(f"⚠️  Meta no encontrada para {config.get_name()}, continuando sin meta.json")
    else:
        print(f"❌ No se encontraron resultados para {config.get_name()}")
        return ExperimentResult(
            config=config,
            avg_test_f1=0.0,
            avg_test_accuracy=0.0,
            avg_long_f1=0.0,
            avg_short_f1=0.0
        )
    
    with open(exp_dir / "production_training_results.json") as f:
        results = json.load(f)
    
    # Extraer métricas
    fold_results = results.get("results", [])
    if not fold_results:
        print(f"❌ Resultados vacíos para {config.get_name()}")
        return ExperimentResult(
            config=config,
            avg_test_f1=0.0,
            avg_test_accuracy=0.0,
            avg_long_f1=0.0,
            avg_short_f1=0.0
        )
    
    avg_f1 = sum(r["test_metrics"]["macro_f1"] for r in fold_results) / len(fold_results)
    avg_acc = sum(r["test_metrics"]["accuracy"] for r in fold_results) / len(fold_results)
    avg_long_f1 = sum(r["test_metrics"]["long_f1"] for r in fold_results) / len(fold_results)
    avg_short_f1 = sum(r["test_metrics"]["short_f1"] for r in fold_results) / len(fold_results)
    
    # PnL si está disponible
    avg_pnl = 0.0
    if "total_pnl" in fold_results[0].get("test_metrics", {}):
        avg_pnl = sum(r["test_metrics"].get("total_pnl", 0.0) for r in fold_results) / len(fold_results)
    
    # MSE si está disponible
    avg_mse = 0.0
    if "mse" in fold_results[0].get("test_metrics", {}):
        avg_mse = sum(r["test_metrics"].get("mse", 0.0) for r in fold_results) / len(fold_results)
    
    result = ExperimentResult(
        config=config,
        avg_test_f1=avg_f1,
        avg_test_accuracy=avg_acc,
        avg_long_f1=avg_long_f1,
        avg_short_f1=avg_short_f1,
        avg_pnl=avg_pnl,
        avg_mse=avg_mse
    )
    
    # Guardar resultados
    with open(exp_dir / "results.json", "w") as f:
        json.dump({
            "config": config.to_dict(),
            "metrics": {
                "avg_test_f1": avg_f1,
                "avg_test_accuracy": avg_acc,
                "avg_long_f1": avg_long_f1,
                "avg_short_f1": avg_short_f1,
                "avg_pnl": avg_pnl,
                "avg_mse": avg_mse
            }
        }, f, indent=2)
    
    print(f"✅ Resultados: F1={avg_f1:.3f}, Long F1={avg_long_f1:.3f}, Short F1={avg_short_f1:.3f}, PnL={avg_pnl:.4f}")
    
    return result

def run_sweep(
    symbol: str,
    timeframe: str,
    epochs: int = 50,
    mode: str = "fast",
    target_returns_filter: Optional[List[float]] = None,
    prediction_horizons_filter: Optional[List[int]] = None,
    sequence_lengths_filter: Optional[List[int]] = None,
    hidden_dims_filter: Optional[List[int]] = None,
    dropouts_filter: Optional[List[float]] = None,
    gap_multipliers_filter: Optional[List[int]] = None,
    device: str = "cuda",
):
    """Ejecuta el sweep completo."""
    
    print(f"\n{'='*80}")
    print(f"🔬 HYPERPARAMETER SWEEP - {symbol} {timeframe}")
    print(f"   Mode: {mode.upper()}")
    print(f"{'='*80}\n")
    
    # Generar experimentos
    experiments = generate_experiments(
        symbol,
        timeframe,
        mode=mode,
        target_returns_filter=target_returns_filter,
        prediction_horizons_filter=prediction_horizons_filter,
        sequence_lengths_filter=sequence_lengths_filter,
        hidden_dims_filter=hidden_dims_filter,
        dropouts_filter=dropouts_filter,
        gap_multipliers_filter=gap_multipliers_filter,
    )
    
    # Ejecutar experimentos
    results = []
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n📊 Experimento {i}/{len(experiments)}")
        result = run_experiment(exp_config, epochs=epochs, device=device)
        results.append(result)
        
        # Guardar resultados parciales
        summary_file = REPO_ROOT / "experiments" / f"{symbol}_{timeframe}_sweep.json"
        with open(summary_file, "w") as f:
            json.dump([
                {
                    "config": r.config.to_dict(),
                    "metrics": {
                        "avg_test_f1": r.avg_test_f1,
                        "avg_test_accuracy": r.avg_test_accuracy,
                        "avg_long_f1": r.avg_long_f1,
                        "avg_short_f1": r.avg_short_f1,
                        "avg_pnl": r.avg_pnl,
                        "avg_mse": r.avg_mse
                    }
                }
                for r in results
            ], f, indent=2)
    
    # Analizar resultados
    print(f"\n{'='*80}")
    print("📊 RESUMEN DE RESULTADOS")
    print(f"{'='*80}\n")
    
    # Ordenar por F1 macro
    results_sorted = sorted(results, key=lambda r: r.avg_test_f1, reverse=True)
    
    print("Top 5 por Macro F1:")
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"\n{i}. {r.config.get_name()}")
        print(f"   Target Return: {r.config.target_return:.3f}%, Horizon: {r.config.prediction_horizon}")
        print(f"   Hidden Dim: {r.config.hidden_dim}, Dropout: {r.config.dropout}, LR: {r.config.lr}")
        print(f"   Macro F1: {r.avg_test_f1:.3f}, Long F1: {r.avg_long_f1:.3f}, Short F1: {r.avg_short_f1:.3f}")
        print(f"   PnL: {r.avg_pnl:.4f}, MSE: {r.avg_mse:.6f}")
    
    # Ordenar por Long F1
    results_by_long = sorted(results, key=lambda r: r.avg_long_f1, reverse=True)
    
    print("\n\nTop 5 por Long F1:")
    for i, r in enumerate(results_by_long[:5], 1):
        print(f"\n{i}. {r.config.get_name()}")
        print(f"   Target Return: {r.config.target_return:.3f}%, Horizon: {r.config.prediction_horizon}")
        print(f"   Long F1: {r.avg_long_f1:.3f}, Short F1: {r.avg_short_f1:.3f}")
    
    # Recomendación
    print(f"\n{'='*80}")
    print("💡 RECOMENDACIÓN")
    print(f"{'='*80}\n")
    
    best = results_sorted[0]
    print(f"La mejor configuración es:")
    print(f"  Target Return: {best.config.target_return:.3f}%")
    print(f"  Prediction Horizon: {best.config.prediction_horizon}")
    print(f"  Sequence Length: {best.config.sequence_length}")
    print(f"  Hidden Dim: {best.config.hidden_dim}")
    print(f"  Dropout: {best.config.dropout}")
    print(f"  Learning Rate: {best.config.lr}")
    print(f"\nMétricas:")
    print(f"  Macro F1: {best.avg_test_f1:.3f}")
    print(f"  Long F1: {best.avg_long_f1:.3f}")
    print(f"  Short F1: {best.avg_short_f1:.3f}")
    print(f"  PnL Implícito: {best.avg_pnl:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter Sweep")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--timeframe", default="15m", help="Timeframe")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per experiment")
    parser.add_argument("--mode", default="fast", choices=["fast", "balanced", "thorough", "aggressive"],
                       help="Sweep mode: fast (9), balanced (27), thorough (81), aggressive (custom)")
    parser.add_argument("--device", default="cuda", help="Dispositivo torch (cuda/hip/cpu)")
    parser.add_argument("--target-returns", help="Lista separada por comas de target returns a usar (ej: 0.0015,0.002)", default=None)
    parser.add_argument("--prediction-horizons", help="Lista separada por comas de horizons (ej: 2,3,4)", default=None)
    parser.add_argument("--sequence-lengths", help="Lista separada por comas de sequence lengths (ej: 48,64)", default=None)
    parser.add_argument("--hidden-dims", help="Lista separada por comas de hidden dims (ej: 160,224)", default=None)
    parser.add_argument("--dropouts", help="Lista separada por comas de dropouts (ej: 0.25,0.3)", default=None)
    parser.add_argument("--gap-multipliers", help="Lista separada por comas de gap multipliers (ej: 2)", default=None)
    
    args = parser.parse_args()
    
    target_returns_filter = parse_csv_floats(args.target_returns) if args.target_returns else None
    prediction_horizons_filter = parse_csv_ints(args.prediction_horizons) if args.prediction_horizons else None
    sequence_lengths_filter = parse_csv_ints(args.sequence_lengths) if args.sequence_lengths else None
    hidden_dims_filter = parse_csv_ints(args.hidden_dims) if args.hidden_dims else None
    dropouts_filter = parse_csv_floats(args.dropouts) if args.dropouts else None
    gap_multipliers_filter = parse_csv_ints(args.gap_multipliers) if args.gap_multipliers else None
    
    run_sweep(
        args.symbol,
        args.timeframe,
        args.epochs,
        args.mode,
        target_returns_filter=target_returns_filter,
        prediction_horizons_filter=prediction_horizons_filter,
        sequence_lengths_filter=sequence_lengths_filter,
        hidden_dims_filter=hidden_dims_filter,
        dropouts_filter=dropouts_filter,
        gap_multipliers_filter=gap_multipliers_filter,
        device=args.device,
    )
