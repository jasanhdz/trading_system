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

from data.collectors.binance_collector import BinanceDataCollector
from data.storage.database_manager import DatabaseManager
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("update_ml_candles")

# NEW CONFIGURATION
CANDLES_DB_URL = "sqlite:///data/binance_candles.db"
TARGET_SYMBOLS = [
    'ETH/USDT',  # Only trading ETH for now
]
TARGET_TIMEFRAMES = ["5m"]

def to_ccxt_symbol(symbol: str) -> str:
    """Convert Binance-style symbol (XRPUSDT) to CCXT pair (XRP/USDT)."""
    if "/" in symbol:
        return symbol
    # Simple heuristic for now, assuming USDT pairs
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USDT"
    return symbol

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

@click.command()
@click.option(
    "--days",
    type=int,
    default=1460,
    help="Backfill window in days (default 1460 = 4 years).",
)
def main(days: int) -> None:
    """Download / refresh OHLCV history for Berzerker symbols."""
    
    # Instantiate new DB Manager
    candles_db = DatabaseManager(CANDLES_DB_URL)
    candles_db.create_tables()
    
    collector = BinanceDataCollector()
    if not collector.connect():
        logger.error("Failed to connect to Binance; aborting.")
        return

    end_date = datetime.now(timezone.utc)
    
    print(f"--- Updating Binance Candles DB ({CANDLES_DB_URL}) ---")
    print(f"Symbols: {TARGET_SYMBOLS}")
    print(f"Timeframes: {TARGET_TIMEFRAMES}")

    for symbol in TARGET_SYMBOLS:
        pair = to_ccxt_symbol(symbol)
        
        for timeframe in TARGET_TIMEFRAMES:
            tf_minutes = timeframe_to_minutes(timeframe)

            latest = candles_db.get_latest_timestamp(pair, timeframe)
            if latest is not None:
                latest_utc = latest if latest.tzinfo else latest.replace(tzinfo=timezone.utc)
                start_candidate = latest_utc + timedelta(minutes=tf_minutes)
            else:
                start_candidate = end_date - timedelta(days=days)

            start_window = end_date - timedelta(days=days)
            start_date = max(start_candidate, start_window)

            if start_date >= end_date:
                click.echo(f"✔ {symbol} [{timeframe}] ya estaba actualizado")
                continue

            click.echo(
                f"→ {symbol} ({pair}) [{timeframe}] {fmt_ts(start_date)} → {fmt_ts(end_date)} UTC"
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
                continue

            first_ts = df["timestamp"].min()
            last_ts = df["timestamp"].max()
            inserted = candles_db.insert_ohlcv_data(df, pair, timeframe)
            click.echo(
                f"✓ {symbol} [{timeframe}] +{inserted} velas "
                f"({fmt_ts(first_ts)} → {fmt_ts(last_ts)} UTC)"
            )

if __name__ == "__main__":
    main()
