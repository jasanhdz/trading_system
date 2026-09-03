import sys
import os
import argparse
import pandas as pd
from pathlib import Path
import ccxt
from datetime import datetime, timedelta, timezone

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from data.collectors.binance_collector import BinanceDataCollector
from data.storage.database_manager import DatabaseManager

def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    return symbol if "/" in symbol else symbol.replace("USDT", "/USDT")


def validate_continuity(df: pd.DataFrame, timeframe: str) -> dict:
    if df.empty or "timestamp" not in df:
        return {"checked": False, "gaps": 0}
    minutes = int(timeframe.rstrip("m")) if timeframe.endswith("m") else 5
    expected = pd.Timedelta(minutes=minutes)
    diffs = df["timestamp"].sort_values().diff().dropna()
    gaps = int((diffs > expected).sum())
    return {"checked": True, "gaps": gaps, "expected_seconds": int(expected.total_seconds())}


def parse_utc_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def update_db(
    symbol: str | None = None,
    timeframe: str = "5m",
    limit: int = 1000,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    days: int | None = None,
):
    print("🔄 Updating Market Data...")
    collector = BinanceDataCollector()
    
    # We need to update the main source used by load_hybrid_data
    # load_hybrid_data reads from database_manager or CSV?
    # Let's check data_loader.py logic.
    # Assuming standard DB management.
    
    db_url = settings.DATABASE_URL
    db = DatabaseManager(db_url)
    db.create_tables() # Ensure tables exist
    
    symbol = normalize_symbol(symbol or settings.SYMBOL)
    
    if end_date is None:
        end_date = datetime.now(timezone.utc)

    if start_date is None and days is not None:
        start_date = end_date - timedelta(days=days)

    if start_date is not None:
        print(f"   Historical backfill for {symbol} {timeframe}: {start_date.isoformat()} → {end_date.isoformat()}")
        df = collector.get_historical_data(symbol, timeframe, start_date, end_date)
    else:
        # Fetch last candles for incremental refresh.
        df = collector.get_ohlcv(symbol, timeframe, limit=limit)
    
    if not df.empty:
        print(f"   Fetched {len(df)} candles via API for {symbol}.")
        continuity = validate_continuity(df, timeframe)
        if continuity["checked"]:
            print(f"   Continuity gaps in fetched batch: {continuity['gaps']}")
        
        # Filter duplicates
        last_ts = db.get_latest_timestamp(symbol, timeframe)
        if last_ts:
            # Ensure timezone awareness matches
            if df['timestamp'].dt.tz is None and last_ts.tzinfo:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            elif df['timestamp'].dt.tz and last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=pd.Timestamp(df['timestamp'].dt.tz).tz)
                
            new_df = df[df['timestamp'] > last_ts]
        else:
            new_df = df
            
        if not new_df.empty:
            count = db.insert_ohlcv_data(new_df, symbol, timeframe)
            print(f"✅ inserted {count} new candles.")
        else:
            print("✨ Database up to date (No new candles).")
    else:
        print("⚠️ No data fetched.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=settings.SYMBOL, help="Symbol to update, e.g. ETHUSDT or BTCUSDT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--start-date", help="UTC start date for historical backfill, e.g. 2020-01-01")
    parser.add_argument("--end-date", help="UTC end date for historical backfill; defaults to now")
    parser.add_argument("--days", type=int, help="Historical backfill window in days. Ignored if --start-date is set.")
    args = parser.parse_args()
    update_db(
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
        start_date=parse_utc_datetime(args.start_date),
        end_date=parse_utc_datetime(args.end_date),
        days=args.days,
    )
