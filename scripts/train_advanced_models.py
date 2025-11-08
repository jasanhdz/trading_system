#!/usr/bin/env python3
"""
Batch trainer for the advanced temporal models.

Loads all symbols from an .env file and trains the advanced model for each
requested timeframe, allowing custom history windows per timeframe via
`--timeframes 5m:180 15m:270`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import click
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts import train_advanced_model  # noqa: E402


def _parse_symbols(env_symbols: str, overrides: Iterable[str]) -> List[str]:
    """
    Parse symbol entries of the form 'BTCUSDT:20:0.8' and return unique symbols.
    """
    tokens = [token.strip() for token in overrides if token.strip()]
    if not tokens:
        raw = env_symbols or ""
        raw_tokens = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
        tokens.extend(raw_tokens)

    symbols: List[str] = []
    seen = set()
    for token in tokens:
        symbol = token.split(":", 1)[0].strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


def _parse_timeframes(values: Tuple[str, ...]) -> Dict[str, int]:
    """
    Convert timeframe tokens like '5m:180' into a mapping of timeframe -> days.
    """
    if not values:
        raise click.BadParameter("Provide at least one --timeframes entry (e.g. --timeframes 5m:180).")

    mapping: Dict[str, int] = {}
    for raw in values:
        token = (raw or "").strip()
        if not token:
            continue

        if ":" not in token:
            raise click.BadParameter(
                f"Invalid timeframe token '{token}'. Use the format timeframe:days (e.g. 5m:180)."
            )

        timeframe_part, days_part = token.split(":", 1)
        timeframe = timeframe_part.strip()
        days_str = days_part.strip()

        if not timeframe or not days_str:
            raise click.BadParameter(
                f"Invalid timeframe token '{token}'. Use the format timeframe:days (e.g. 5m:180)."
            )

        try:
            history_days = int(days_str)
        except ValueError as exc:
            raise click.BadParameter(
                f"Invalid history days '{days_str}' for timeframe '{timeframe}'. Provide an integer."
            ) from exc

        if history_days <= 0:
            raise click.BadParameter(
                f"History days must be positive for timeframe '{timeframe}' (got {history_days})."
            )

        mapping[timeframe] = history_days

    if not mapping:
        raise click.BadParameter("No valid timeframe entries were provided.")

    return mapping


def _resolve_env_path(env_path: Path) -> Path:
    if env_path.is_file():
        return env_path
    if env_path.is_dir():
        candidate = env_path / ".env"
        if candidate.is_file():
            return candidate
    raise click.BadParameter(f"Could not find an .env file at '{env_path}'.")


def _load_environment(env_path: Path) -> None:
    load_dotenv(dotenv_path=env_path, override=False)


@click.command()
@click.option(
    "--env-path",
    type=click.Path(path_type=Path),
    default=REPO_ROOT / ".env",
    show_default=True,
    help="Path to the .env file that contains the SYMBOLS definition.",
)
@click.option(
    "--symbols",
    multiple=True,
    help="Optional symbol overrides. Use SYMBOL:LEV:CAP format or plain SYMBOL.",
)
@click.option(
    "--timeframes",
    multiple=True,
    required=True,
    help="Timeframe and history window pair (format: timeframe:days), e.g. --timeframes 5m:180 15m:270.",
)
@click.option("--sequence-length", default=24, show_default=True, help="Lookback window")
@click.option("--horizon", default=12, show_default=True, help="Prediction horizon")
@click.option("--target-return", default=0.002, show_default=True, help="Target return threshold")
@click.option("--epochs", default=30, show_default=True, help="Training epochs")
@click.option("--batch-size", default=512, show_default=True, help="Batch size")
@click.option("--lr", default=1e-3, show_default=True, help="Learning rate")
@click.option("--hidden-dim", default=128, show_default=True, help="LSTM hidden dimension")
@click.option("--lstm-layers", default=2, show_default=True, help="Number of LSTM layers")
@click.option("--dense-dims", default="256,128", show_default=True, help="Dense layer dimensions")
@click.option("--dropout", default=0.3, show_default=True, help="Dropout rate")
@click.option("--use-attention/--no-attention", default=True, help="Enable attention mechanism")
@click.option("--bidirectional/--unidirectional", default=True, help="Use bidirectional LSTM")
@click.option("--feature-selection/--no-feature-selection", default=True, help="Enable feature selection")
@click.option("--n-features", default=32, show_default=True, help="Number of features to select")
@click.option("--walk-forward/--single-split", default=False, help="Use walk-forward validation")
@click.option("--n-folds", default=5, show_default=True, help="Number of walk-forward folds")
@click.option("--ensemble", default=0, show_default=True, help="Number of models in ensemble (0=single)")
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]), help="Device")
@click.option("--seed", default=42, show_default=True, help="Random seed")
def main(
    env_path: Path,
    symbols: Tuple[str, ...],
    timeframes: Tuple[str, ...],
    sequence_length: int,
    horizon: int,
    target_return: float,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    lstm_layers: int,
    dense_dims: str,
    dropout: float,
    use_attention: bool,
    bidirectional: bool,
    feature_selection: bool,
    n_features: int,
    walk_forward: bool,
    n_folds: int,
    ensemble: int,
    device: str,
    seed: int,
) -> None:
    """
    Train advanced models for every symbol defined in the .env file across the
    requested timeframes.
    """

    resolved_env = _resolve_env_path(env_path)
    _load_environment(resolved_env)

    env_symbols = os.getenv("SYMBOLS", "")
    symbol_list = _parse_symbols(env_symbols, symbols)
    if not symbol_list:
        raise click.BadParameter(
            "No symbols found. Ensure the .env file defines SYMBOLS or pass --symbols overrides."
        )

    timeframe_days = _parse_timeframes(timeframes)

    total_jobs = len(symbol_list) * len(timeframe_days)
    click.echo("=" * 80)
    click.echo("BATCH TRAINING - ADVANCED MODELS")
    click.echo("=" * 80)
    click.echo(f"Env file        : {resolved_env}")
    click.echo(f"Symbols         : {', '.join(symbol_list)}")
    click.echo("Timeframes      : " + ", ".join(f"{tf} ({days} days)" for tf, days in timeframe_days.items()))
    click.echo(f"Total jobs      : {total_jobs}")
    click.echo("-" * 80)

    train_fn = train_advanced_model.main.callback  # type: ignore[attr-defined]
    successes: List[Tuple[str, str]] = []
    failures: List[Tuple[str, str, str]] = []

    for symbol in symbol_list:
        for timeframe, history_days in timeframe_days.items():
            click.echo(f"\n>>> Training {symbol} @ {timeframe} ({history_days} days)")
            try:
                train_fn(
                    symbol=symbol,
                    timeframe=timeframe,
                    sequence_length=sequence_length,
                    horizon=horizon,
                    target_return=target_return,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    hidden_dim=hidden_dim,
                    lstm_layers=lstm_layers,
                    dense_dims=dense_dims,
                    dropout=dropout,
                    use_attention=use_attention,
                    bidirectional=bidirectional,
                    feature_selection=feature_selection,
                    n_features=n_features,
                    walk_forward=walk_forward,
                    n_folds=n_folds,
                    ensemble=ensemble,
                    device=device,
                    seed=seed,
                    history_days=history_days,
                )
                successes.append((symbol, timeframe))
            except SystemExit as exit_exc:
                if exit_exc.code not in (None, 0):
                    message = f"Exited with status {exit_exc.code}"
                    failures.append((symbol, timeframe, message))
                    click.echo(f"!!! Failed {symbol} @ {timeframe}: {message}", err=True)
                else:
                    successes.append((symbol, timeframe))
            except Exception as exc:  # noqa: BLE001
                failures.append((symbol, timeframe, str(exc)))
                click.echo(f"!!! Failed {symbol} @ {timeframe}: {exc}", err=True)

    click.echo("\n" + "=" * 80)
    click.echo("BATCH SUMMARY")
    click.echo("=" * 80)
    click.echo(f"Successful jobs : {len(successes)}")
    click.echo(f"Failed jobs     : {len(failures)}")
    if successes:
        click.echo("\nCompleted:")
        for symbol, timeframe in successes:
            click.echo(f"  - {symbol} @ {timeframe}")
    if failures:
        click.echo("\nFailures:")
        for symbol, timeframe, message in failures:
            click.echo(f"  - {symbol} @ {timeframe} :: {message}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
