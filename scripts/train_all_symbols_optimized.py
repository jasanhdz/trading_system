#!/usr/bin/env python3
"""
Train optimized models for all symbols in .env

This script:
1. Reads symbols from .env
2. Determines optimal hyperparameters per symbol
3. Trains models for 5m and 15m timeframes
4. Optimized for current bear market (last 10 days heavy dump)

Usage:
    python scripts/train_all_symbols_optimized.py
    python scripts/train_all_symbols_optimized.py --timeframes 5m,15m
    python scripts/train_all_symbols_optimized.py --symbols BTCUSDT,ETHUSDT --fast
    python scripts/train_all_symbols_optimized.py --target-return 0.002 --epochs 150
"""
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import click
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# Load .env
load_dotenv(REPO_ROOT / ".env")


@dataclass
class SymbolConfig:
    """Configuration for a symbol's training."""
    symbol: str
    leverage: int
    confidence: float
    
    # Determined automatically based on symbol characteristics
    tier: str  # "top", "mid", "low"
    hidden_dim: int
    dropout: float
    target_return: float
    sequence_length_5m: int
    sequence_length_15m: int
    horizon_5m: int
    horizon_15m: int
    epochs: int
    lr: float


# Symbol tiers based on liquidity and market cap
TOP_TIER = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
MID_TIER = ["XRPUSDT", "ADAUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT", 
            "LTCUSDT", "BCHUSDT", "UNIUSDT", "TRXUSDT", "ETCUSDT"]
# Everything else is LOW_TIER


MODEL_ROOT = REPO_ROOT / "models" / "advanced"


def symbol_key(symbol: str) -> str:
    """Normalize symbols to filesystem-safe keys."""
    return symbol.replace("/", "").replace(":", "").replace("-", "").upper()


def model_artifacts_exist(symbol: str, timeframe: str) -> bool:
    """Check whether a trained model already exists for the symbol/timeframe."""
    symbol_dir = MODEL_ROOT / symbol_key(symbol) / timeframe
    return (symbol_dir / "model.pt").exists() and (symbol_dir / "meta.json").exists()


def normalize_force_symbol(value: str) -> Optional[str]:
    """Normalize user-provided symbols for force retraining."""
    cleaned = value.strip().upper()
    if not cleaned:
        return None
    cleaned = cleaned.replace("/", "").replace(":", "").replace("-", "")
    if cleaned.endswith("USDT"):
        return cleaned
    if cleaned.endswith("USD"):
        return f"{cleaned}T"
    if len(cleaned) <= 6:
        return f"{cleaned}USDT"
    return cleaned


def parse_symbols_from_env() -> List[SymbolConfig]:
    """Parse SYMBOLS from .env file."""
    symbols_str = os.getenv("SYMBOLS", "")
    
    if not symbols_str:
        raise ValueError("SYMBOLS not found in .env")
    
    # Parse format: SYMBOL:LEVERAGE:CONFIDENCE
    pattern = r'([A-Z0-9]+)USDT:(\d+):([\d.]+)'
    matches = re.findall(pattern, symbols_str)
    
    configs = []
    for base, leverage, confidence in matches:
        symbol = f"{base}USDT"
        
        # Determine tier
        if symbol in TOP_TIER:
            tier = "top"
        elif symbol in MID_TIER:
            tier = "mid"
        else:
            tier = "low"
        
        config = SymbolConfig(
            symbol=symbol,
            leverage=int(leverage),
            confidence=float(confidence),
            tier=tier,
            # Will be set by optimize_hyperparameters()
            hidden_dim=0,
            dropout=0,
            target_return=0,
            sequence_length_5m=0,
            sequence_length_15m=0,
            horizon_5m=0,
            horizon_15m=0,
            epochs=0,
            lr=0,
        )
        
        configs.append(config)
    
    return configs


def optimize_hyperparameters(config: SymbolConfig, bear_market: bool = True) -> SymbolConfig:
    """
    Determine optimal hyperparameters based on symbol tier and market conditions.
    
    Bear market adjustments (current situation - heavy dump last 10 days):
    - Higher target_return (avoid false signals in volatility)
    - Longer sequences (see longer-term downtrend)
    - Moderate dropout (not too high, model needs to learn new patterns)
    - Higher LR (adapt faster to changing market)
    """
    
    # Base parameters by tier
    if config.tier == "top":
        # BTC, ETH, BNB, SOL - highest liquidity
        config.hidden_dim = 128
        config.dropout = 0.20
        config.target_return = 0.0025  # 0.25% (bear: more strict)
        config.sequence_length_5m = 36  # 3 hours
        config.sequence_length_15m = 40  # 10 hours
        config.horizon_5m = 18  # 1.5 hours
        config.horizon_15m = 12  # 3 hours
        config.epochs = 100
        config.lr = 0.003 if bear_market else 0.002
        
    elif config.tier == "mid":
        # Top 10-15 altcoins
        config.hidden_dim = 96
        config.dropout = 0.22
        config.target_return = 0.0035  # 0.35% (more volatile)
        config.sequence_length_5m = 32  # 2.7 hours
        config.sequence_length_15m = 36  # 9 hours
        config.horizon_5m = 15  # 1.25 hours
        config.horizon_15m = 10  # 2.5 hours
        config.epochs = 90
        config.lr = 0.003
        
    else:  # low tier
        # Low-cap altcoins - less data, more volatile
        config.hidden_dim = 64
        config.dropout = 0.25
        config.target_return = 0.0045  # 0.45% (very volatile)
        config.sequence_length_5m = 28  # 2.3 hours
        config.sequence_length_15m = 32  # 8 hours
        config.horizon_5m = 12  # 1 hour
        config.horizon_15m = 8  # 2 hours
        config.epochs = 80
        config.lr = 0.003
    
    # Bear market specific adjustments
    if bear_market:
        # Increase target return by 20% to avoid false signals
        config.target_return *= 1.2
        
        # Increase sequence length by 15% to see longer downtrend
        config.sequence_length_5m = int(config.sequence_length_5m * 1.15)
        config.sequence_length_15m = int(config.sequence_length_15m * 1.15)
    
    return config


def train_symbol_timeframe(
    config: SymbolConfig,
    timeframe: str,
    fast_mode: bool = False,
    walk_forward: bool = True,
    force: bool = False,
    job_label: Optional[str] = None,
    overrides: Optional[dict[str, object]] = None,
) -> dict:
    """Train a single symbol-timeframe combination and report status."""
    
    header_label = job_label or f"{config.symbol} {timeframe}"
    artifact_dir = MODEL_ROOT / symbol_key(config.symbol) / timeframe
    already_trained = model_artifacts_exist(config.symbol, timeframe)
    
    if already_trained and not force:
        print(f"\n{header_label}")
        print(f"→ Skipping (artifacts already exist in {artifact_dir})")
        return {"status": "skipped"}
    
    # Select parameters based on timeframe
    if timeframe == "5m":
        sequence_length = config.sequence_length_5m
        horizon = config.horizon_5m
    elif timeframe == "15m":
        sequence_length = config.sequence_length_15m
        horizon = config.horizon_15m
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    overrides = overrides or {}
    override_target = overrides.get("target_return")
    override_lr = overrides.get("lr")
    override_dropout = overrides.get("dropout")
    override_sequence = overrides.get(f"sequence_length_{timeframe}")
    override_horizon = overrides.get(f"horizon_{timeframe}")
    override_hidden = overrides.get("hidden_dim")
    override_epochs = overrides.get("epochs")
    override_n_features = overrides.get("n_features")
    override_batch = overrides.get("batch_size")
    override_dense1 = overrides.get("dense_dim1")
    override_dense2 = overrides.get("dense_dim2")
    override_patience = overrides.get("patience")
    override_lstm_layers = overrides.get("lstm_layers")

    hidden_dim_effective = int(override_hidden) if override_hidden is not None else config.hidden_dim
    target_return_effective = float(override_target) if override_target is not None else config.target_return
    dropout_effective = float(override_dropout) if override_dropout is not None else config.dropout
    lr_effective = float(override_lr) if override_lr is not None else config.lr
    sequence_effective = int(override_sequence) if override_sequence is not None else sequence_length
    horizon_effective = int(override_horizon) if override_horizon is not None else horizon
    epochs_effective = (
        int(override_epochs)
        if override_epochs is not None
        else (30 if fast_mode else config.epochs)
    )
    patience_effective = (
        int(override_patience)
        if override_patience is not None
        else min(20, max(1, epochs_effective // 5))
    )
    n_features_effective = int(override_n_features) if override_n_features is not None else 32
    batch_effective = int(override_batch) if override_batch is not None else 512
    lstm_layers_effective = int(override_lstm_layers) if override_lstm_layers is not None else 2

    dense_dim1 = int(override_dense1) if override_dense1 is not None else hidden_dim_effective * 2
    dense_dim2 = int(override_dense2) if override_dense2 is not None else hidden_dim_effective
    dense_dims = f"{dense_dim1},{dense_dim2}"
    
    # Fast mode: reduce epochs and no walk-forward
    n_folds = 3 if walk_forward and not fast_mode else 5
    
    # Build command
    cmd = [
        "python", "scripts/train_advanced_model.py",
        "--symbol", config.symbol,
        "--timeframe", timeframe,
        "--sequence-length", str(sequence_effective),
        "--horizon", str(horizon_effective),
        "--target-return", f"{target_return_effective}",
        "--hidden-dim", str(hidden_dim_effective),
        "--lstm-layers", str(lstm_layers_effective),
        "--dense-dims", dense_dims,
        "--dropout", f"{dropout_effective}",
        "--epochs", str(epochs_effective),
        "--patience", str(patience_effective),
        "--lr", f"{lr_effective}",
        "--batch-size", str(batch_effective),
        "--feature-selection",
        "--n-features", str(n_features_effective),
        "--use-attention",
        "--bidirectional",
        "--device", "cpu",
        "--seed", "42",
    ]
    
    # Add walk-forward if enabled
    if walk_forward and not fast_mode:
        cmd.extend(["--walk-forward", "--n-folds", str(n_folds)])
    
    banner = "=" * 80
    print(f"\n{banner}")
    print(f"Training {header_label}")
    print(f"{banner}")
    print(f"Tier: {config.tier.upper()}")
    print(f"Config:")
    print(f"  Sequence Length: {sequence_effective} ({timeframe})")
    print(f"  Horizon: {horizon_effective}")
    print(f"  Target Return: {target_return_effective:.4f} ({target_return_effective*100:.2f}%)")
    print(f"  Hidden Dim: {hidden_dim_effective}")
    print(f"  Dense Dims: {dense_dims}")
    print(f"  Dropout: {dropout_effective}")
    print(f"  Epochs: {epochs_effective}")
    print(f"  LR: {lr_effective}")
    print(f"  Walk-Forward: {walk_forward and not fast_mode}")
    print(f"{banner}\n")
    
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ {header_label} completed successfully\n")
        return {"status": "success"}
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {header_label} failed with error code {e.returncode}\n")
        return {"status": "failed", "returncode": e.returncode}
    except KeyboardInterrupt:
        print(f"\n⚠ {header_label} interrupted by user\n")
        return {"status": "failed", "error": "interrupted"}
    except Exception as exc:  # Defensive: surface unexpected issues
        print(f"\n✗ {header_label} failed: {exc}\n")
        return {"status": "failed", "error": str(exc)}


@click.command()
@click.option(
    "--symbols",
    default=None,
    help="Comma-separated list of symbols to train (default: all from .env)",
)
@click.option(
    "--timeframes",
    default="5m,15m",
    help="Comma-separated list of timeframes (default: 5m,15m)",
)
@click.option(
    "--fast",
    is_flag=True,
    help="Fast mode: 30 epochs, no walk-forward (for testing)",
)
@click.option(
    "--walk-forward/--no-walk-forward",
    default=True,
    help="Enable or disable walk-forward validation (enabled by default)",
)
@click.option(
    "--bear-market/--no-bear-market",
    default=True,
    help="Optimize for bear market conditions (default: True)",
)
@click.option(
    "--start-from",
    default=None,
    help="Start from specific symbol (useful if script was interrupted)",
)
@click.option(
    "--force",
    default=None,
    help="Force retraining: 'all' or comma-separated list of symbols",
)
@click.option(
    "--max-workers",
    default=2,
    show_default=True,
    type=int,
    help="Maximum concurrent training jobs",
)
@click.option(
    "--target-return",
    type=float,
    default=None,
    help="Override target return (e.g. 0.002) for all symbols/timeframes",
)
@click.option(
    "--epochs",
    type=int,
    default=None,
    help="Override training epochs (applies even in --fast mode)",
)
@click.option(
    "--lr",
    type=float,
    default=None,
    help="Override learning rate",
)
@click.option(
    "--dropout",
    type=float,
    default=None,
    help="Override dropout probability",
)
@click.option(
    "--sequence-length-5m",
    type=int,
    default=None,
    help="Override sequence length for 5m timeframe",
)
@click.option(
    "--sequence-length-15m",
    type=int,
    default=None,
    help="Override sequence length for 15m timeframe",
)
@click.option(
    "--horizon-5m",
    type=int,
    default=None,
    help="Override horizon steps for 5m timeframe",
)
@click.option(
    "--horizon-15m",
    type=int,
    default=None,
    help="Override horizon steps for 15m timeframe",
)
@click.option(
    "--hidden-dim",
    type=int,
    default=None,
    help="Override hidden dimension size",
)
@click.option(
    "--dense-dim1",
    type=int,
    default=None,
    help="Override first dense layer dimension",
)
@click.option(
    "--dense-dim2",
    type=int,
    default=None,
    help="Override second dense layer dimension",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Override batch size",
)
@click.option(
    "--n-features",
    type=int,
    default=None,
    help="Override number of selected features",
)
@click.option(
    "--patience",
    type=int,
    default=None,
    help="Override early stopping patience",
)
@click.option(
    "--lstm-layers",
    type=int,
    default=None,
    help="Override number of LSTM layers",
)
def main(
    symbols: Optional[str],
    timeframes: str,
    fast: bool,
    walk_forward: bool,
    bear_market: bool,
    start_from: Optional[str],
    force: Optional[str],
    max_workers: int,
    target_return: Optional[float],
    epochs: Optional[int],
    lr: Optional[float],
    dropout: Optional[float],
    sequence_length_5m: Optional[int],
    sequence_length_15m: Optional[int],
    horizon_5m: Optional[int],
    horizon_15m: Optional[int],
    hidden_dim: Optional[int],
    dense_dim1: Optional[int],
    dense_dim2: Optional[int],
    batch_size: Optional[int],
    n_features: Optional[int],
    patience: Optional[int],
    lstm_layers: Optional[int],
):
    """
    Train optimized models for all symbols.
    
    Optimizations:
    - Automatic hyperparameter selection per symbol tier
    - Bear market adjustments (higher target_return, longer sequences)
    - Walk-forward validation for realistic performance
    - Parallel training support with configurable worker pool
    
    Examples:
        # Train all symbols (takes hours)
        python scripts/train_all_symbols_optimized.py
        
        # Fast test on BTC only
        python scripts/train_all_symbols_optimized.py --symbols BTCUSDT --fast
        
        # Top tier symbols only
        python scripts/train_all_symbols_optimized.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT
        
        # 15m only
        python scripts/train_all_symbols_optimized.py --timeframes 15m
        
        # Resume from specific symbol
        python scripts/train_all_symbols_optimized.py --start-from SOLUSDT
    """
    
    max_workers = max(1, max_workers)

    force_all = False
    force_symbols: set[str] = set()
    if force:
        if force.strip().lower() == "all":
            force_all = True
        else:
            parsed_symbols = []
            for token in force.split(","):
                normalized = normalize_force_symbol(token)
                if normalized:
                    parsed_symbols.append(normalized)
            if parsed_symbols:
                force_symbols = set(parsed_symbols)
            else:
                print("⚠ No valid symbols provided for --force; ignoring.")

    print("\n" + "="*80)
    print("OPTIMIZED MULTI-SYMBOL TRAINING")
    print("="*80)
    print(f"\nMode: {'FAST (testing)' if fast else 'FULL (production)'}")
    print(f"Market Condition: {'BEAR MARKET' if bear_market else 'NORMAL'}")
    print(f"Walk-Forward: {walk_forward}")
    print(f"Timeframes: {timeframes}")
    print(f"Parallel Jobs: {max_workers}")

    overrides: dict[str, object] = {}
    if target_return is not None:
        overrides["target_return"] = target_return
    if epochs is not None:
        overrides["epochs"] = epochs
    if lr is not None:
        overrides["lr"] = lr
    if dropout is not None:
        overrides["dropout"] = dropout
    if sequence_length_5m is not None:
        overrides["sequence_length_5m"] = sequence_length_5m
    if sequence_length_15m is not None:
        overrides["sequence_length_15m"] = sequence_length_15m
    if horizon_5m is not None:
        overrides["horizon_5m"] = horizon_5m
    if horizon_15m is not None:
        overrides["horizon_15m"] = horizon_15m
    if hidden_dim is not None:
        overrides["hidden_dim"] = hidden_dim
    if dense_dim1 is not None:
        overrides["dense_dim1"] = dense_dim1
    if dense_dim2 is not None:
        overrides["dense_dim2"] = dense_dim2
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if n_features is not None:
        overrides["n_features"] = n_features
    if patience is not None:
        overrides["patience"] = patience
    if lstm_layers is not None:
        overrides["lstm_layers"] = lstm_layers

    if overrides:
        print("Overrides:")
        for key, value in overrides.items():
            print(f"  {key}: {value}")
        print()
    if force_all:
        print("Force Retrain: all")
    elif force_symbols:
        print(f"Force Retrain: {', '.join(sorted(force_symbols))}")
    else:
        print("Force Retrain: none")
    if bear_market:
        print("\n⚠️  BEAR MARKET OPTIMIZATIONS ACTIVE:")
        print("  • Target return +20% (avoid false signals)")
        print("  • Sequence length +15% (see longer downtrend)")
        print("  • Higher LR (adapt faster)")
    print("="*80 + "\n")
    
    # Parse symbols from .env or CLI
    all_configs = parse_symbols_from_env()
    
    if symbols:
        # Filter to specified symbols
        symbol_list = [s.strip() for s in symbols.split(",")]
        all_configs = [c for c in all_configs if c.symbol in symbol_list]
        
        if not all_configs:
            print(f"✗ No symbols found matching: {symbol_list}")
            return
    
    # Filter by start_from if specified
    if start_from:
        try:
            start_idx = next(i for i, c in enumerate(all_configs) if c.symbol == start_from)
            all_configs = all_configs[start_idx:]
            print(f"Starting from {start_from} (index {start_idx})")
        except StopIteration:
            print(f"✗ Symbol {start_from} not found")
            return
    
    # Optimize hyperparameters for each symbol
    for config in all_configs:
        optimize_hyperparameters(config, bear_market=bear_market)
    
    # Parse timeframes
    timeframe_list = [tf.strip() for tf in timeframes.split(",") if tf.strip()]
    if not timeframe_list:
        print("✗ No valid timeframes provided.")
        return
    
    # Build job list for scheduling
    jobs = [(config, tf) for config in all_configs for tf in timeframe_list]
    total_jobs = len(jobs)
    
    # Summary
    print("Training Plan:")
    print(f"  Symbols: {len(all_configs)}")
    print(f"  Timeframes: {len(timeframe_list)}")
    print(f"  Total Models: {total_jobs}")
    
    if total_jobs == 0:
        print("✗ No symbol/timeframe combinations to process.")
        return
    
    if not force_all:
        pending_skips = sum(
            1
            for config in all_configs
            for tf in timeframe_list
            if model_artifacts_exist(config.symbol, tf) and symbol_key(config.symbol) not in force_symbols
        )
        if pending_skips:
            print(f"  Existing Artifacts: {pending_skips} (will skip unless forced)")
    
    # Estimate time
    minutes_per_model = 5 if fast else 25
    if walk_forward and not fast:
        minutes_per_model *= 1.5  # Walk-forward takes longer
    
    total_minutes = total_jobs * minutes_per_model
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    
    print(f"  Estimated Time: {hours}h {minutes}m")
    print()
    
    # Group symbols by tier for display
    tier_counts = {"top": 0, "mid": 0, "low": 0}
    for config in all_configs:
        tier_counts[config.tier] += 1
    
    print("Symbol Distribution:")
    print(f"  Top Tier: {tier_counts['top']} (BTC, ETH, etc.)")
    print(f"  Mid Tier: {tier_counts['mid']} (Top altcoins)")
    print(f"  Low Tier: {tier_counts['low']} (Small cap)")
    print()
    
    # Confirm
    if not fast:
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Train all combinations
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_meta = {}
        for index, (config, tf) in enumerate(jobs, start=1):
            job_label = f"[{index}/{total_jobs}] {config.symbol} {tf}"
            force_job = force_all or symbol_key(config.symbol) in force_symbols
            future = executor.submit(
                train_symbol_timeframe,
                config,
                tf,
                fast_mode=fast,
                walk_forward=walk_forward,
                force=force_job,
                job_label=job_label,
                overrides=overrides if overrides else None,
            )
            future_to_meta[future] = {
                "symbol": config.symbol,
                "timeframe": tf,
                "tier": config.tier,
                "job_label": job_label,
                "index": index,
                "forced": force_job,
            }
        
        for future in as_completed(future_to_meta):
            meta = future_to_meta[future]
            try:
                job_result = future.result()
            except Exception as exc:
                job_result = {"status": "failed", "error": str(exc)}
            meta.update(job_result)
            results.append(meta)
    
    results.sort(key=lambda r: r["index"])
    
    # Summary
    elapsed = time.time() - start_time
    elapsed_hours = int(elapsed // 3600)
    elapsed_minutes = int((elapsed % 3600) // 60)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nTotal Time: {elapsed_hours}h {elapsed_minutes}m")
    print(f"Models Processed: {len(results)}")
    
    successful = sum(1 for r in results if r.get("status") == "success")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    failed = sum(1 for r in results if r.get("status") == "failed")
    
    print(f"  ✓ Successful: {successful}")
    print(f"  - Skipped: {skipped}")
    print(f"  ✗ Failed: {failed}")
    
    if failed > 0:
        print("\nFailed models:")
        for r in results:
            if r.get("status") == "failed":
                detail = r.get("error") or f"returncode={r.get('returncode')}"
                print(f"  - {r['symbol']} {r['timeframe']} ({detail})")
    
    # Save results
    results_file = REPO_ROOT / "models" / "advanced" / "training_results.txt"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, "w") as f:
        f.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total models: {len(results)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Skipped: {skipped}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Time elapsed: {elapsed_hours}h {elapsed_minutes}m\n\n")
        f.write("Results:\n")
        for r in results:
            status_icon = {
                "success": "✓",
                "failed": "✗",
                "skipped": "-",
            }.get(r.get("status"), "?")
            forced_flag = " [forced]" if r.get("forced") else ""
            note = ""
            if r.get("status") == "failed":
                detail = r.get("error") or f"returncode={r.get('returncode')}"
                note = f" - {detail}"
            elif r.get("status") == "skipped":
                note = " - already trained"
            f.write(f"{status_icon}{forced_flag} {r['symbol']} {r['timeframe']} ({r['tier']}){note}\n")
    
    print(f"\nResults saved to: {results_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
