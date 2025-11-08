#!/usr/bin/env python3
"""Update OHLCV history for all ML symbols defined in the bot .env."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import click
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
BOT_ENV_PATH = REPO_ROOT / "binance_futures_bot_py" / ".env"

# Ensure environment reflects the bot configuration
if BOT_ENV_PATH.exists():
    load_dotenv(dotenv_path=BOT_ENV_PATH, override=False)

# Import project modules
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "binance_futures_bot_py" / "src"))

from data.collectors.binance_collector import BinanceDataCollector
from data.storage.database_manager import db_manager
from infra.config import Config
from utils.logger import setup_logger

logger = setup_logger("update_ml_candles")

QUOTE_TOKENS = ("USDT", "BUSD", "USDC", "BTC", "ETH")
DEFAULT_EXTRA_TIMEFRAMES = ("15m",)


def to_ccxt_symbol(symbol: str) -> str:
    """Convert Binance-style symbol (XRPUSDT) to CCXT pair (XRP/USDT)."""
    clean = symbol.replace("/", "").upper()
    for quote in QUOTE_TOKENS:
        if clean.endswith(quote):
            base = clean[: -len(quote)]
            return f"{base}/{quote}"
    return clean


def timeframe_to_minutes(timeframe: str) -> int:
    """Translate timeframe strings to minutes."""
    timeframe = timeframe.strip().lower()
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 1440
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def fmt_ts(ts) -> str:
    """Format timestamp (aware or naive) as UTC string."""
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.strftime("%Y-%m-%d %H:%M")


def _filter_symbols(symbols: Iterable[str], only: Optional[List[str]]) -> List[str]:
    if not only:
        return list(symbols)
    wanted = {s.upper().replace("/", "") for s in only}
    result = [s for s in symbols if s.upper().replace("/", "") in wanted]
    missing = wanted.difference({s.upper().replace("/", "") for s in result})
    if missing:
        logger.warning("Requested symbols not present in config", {"missing": sorted(missing)})
    return result


def _normalize_timeframes(extra: Iterable[str] | str | None) -> List[str]:
    if not extra:
        return []
    if isinstance(extra, str):
        items = [token.strip() for token in extra.split(",")]
        return [token for token in items if token]
    return [token for token in extra if token]


def _resolve_timeframes(
    sym_timeframe: Optional[str],
    cfg: Config,
    timeframe_override: Optional[str],
) -> List[str]:
    if timeframe_override:
        return [timeframe_override]

    timeframes: List[str] = []
    seen = set()

    def add(tf: Optional[str]) -> None:
        if not tf:
            return
        norm = tf.strip()
        if not norm:
            return
        key = norm.lower()
        if key not in seen:
            seen.add(key)
            timeframes.append(norm)

    add(sym_timeframe or cfg.ML_DEFAULT_TIMEFRAME)

    cfg_extra = getattr(cfg, "ML_EXTRA_TIMEFRAMES", None)
    if not cfg_extra:
        cfg_extra = getattr(cfg, "ML_ADDITIONAL_TIMEFRAMES", None)
    extras = _normalize_timeframes(cfg_extra) or list(DEFAULT_EXTRA_TIMEFRAMES)
    for item in extras:
        add(item)

    return timeframes


@click.command()
@click.option("--symbol", "symbols", multiple=True, help="Limit update to the given symbols.")
@click.option(
    "--timeframe",
    "timeframe_override",
    default=None,
    help="Override timeframe (defaults to per-symbol config).",
)
@click.option(
    "--days",
    type=int,
    default=None,
    help="Backfill window in days when database has no data (defaults to ML_HISTORY_DAYS).",
)
def main(symbols: List[str], timeframe_override: Optional[str], days: Optional[int]) -> None:
    """Download / refresh OHLCV history for all configured ML symbols."""
    cfg = Config()
    all_symbols = cfg.SYMBOLS or [cfg.SYMBOL]
    target_symbols = _filter_symbols(all_symbols, list(symbols))

    if not target_symbols:
        logger.error("No symbols to update; check SYMBOLS in .env")
        return

    collector = BinanceDataCollector()
    if not collector.connect():
        logger.error("Failed to connect to Binance; aborting.")
        return

    db_manager.create_tables()

    end_date = datetime.now(timezone.utc)
    default_days = days or getattr(cfg, "ML_HISTORY_DAYS", 180)

    for symbol in target_symbols:
        pair = to_ccxt_symbol(symbol)
        sym_cfg = cfg.get_symbol_config(symbol)
        timeframes = _resolve_timeframes(sym_cfg.timeframe, cfg, timeframe_override)

        for timeframe in timeframes:
            tf_minutes = timeframe_to_minutes(timeframe)

            latest = db_manager.get_latest_timestamp(pair, timeframe)
            if latest is not None:
                latest_utc = latest if latest.tzinfo else latest.replace(tzinfo=timezone.utc)
                start_candidate = latest_utc + timedelta(minutes=tf_minutes)
            else:
                start_candidate = end_date - timedelta(days=default_days)

            start_window = end_date - timedelta(days=default_days)
            start_date = max(start_candidate, start_window)

            if start_date >= end_date:
                click.echo(f"✔ {symbol} [{timeframe}] ya estaba actualizado")
                logger.info("No update required", {"symbol": pair, "timeframe": timeframe})
                continue

            click.echo(
                f"→ {symbol} ({pair}) [{timeframe}] {fmt_ts(start_date)} → {fmt_ts(end_date)} UTC"
            )
            logger.info(
                "Collecting candles",
                {
                    "symbol": pair,
                    "timeframe": timeframe,
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
            )

            try:
                df = collector.get_historical_data(pair, timeframe, start_date, end_date)
            except Exception as exc:
                logger.error(
                    "Collection failed", {"symbol": pair, "timeframe": timeframe, "err": str(exc)}
                )
                continue

            if latest is not None and not df.empty:
                latest_utc = latest if latest.tzinfo else latest.replace(tzinfo=timezone.utc)
                df = df[df["timestamp"] > pd.Timestamp(latest_utc)]

            if df.empty:
                click.echo(f"… {symbol} [{timeframe}] sin nuevas velas")
                logger.info("No new candles", {"symbol": pair, "timeframe": timeframe})
                continue

            first_ts = df["timestamp"].min()
            last_ts = df["timestamp"].max()
            inserted = db_manager.insert_ohlcv_data(df, pair, timeframe)
            click.echo(
                f"✓ {symbol} [{timeframe}] +{inserted} velas "
                f"({fmt_ts(first_ts)} → {fmt_ts(last_ts)} UTC)"
            )
            logger.info(
                "Database updated",
                {
                    "symbol": pair,
                    "timeframe": timeframe,
                    "rows": inserted,
                    "from": first_ts.isoformat() if hasattr(first_ts, "isoformat") else str(first_ts),
                    "to": last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts),
                },
            )


if __name__ == "__main__":
    main()
