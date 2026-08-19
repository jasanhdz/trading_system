#!/usr/bin/env python3
"""Build immutable 1m Parquet sources from checksum-verified public archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

SANDBOX = Path(__file__).resolve().parents[1]
REPOSITORY = SANDBOX.parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis.utils import sha256_file


COLUMNS = (
    "open_time_ms", "open", "high", "low", "close", "volume",
    "close_time_ms", "quote_volume", "trade_count", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore",
)
OUTPUT_COLUMNS = COLUMNS[:-1]


def load_archive(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = tuple(item for item in archive.infolist() if not item.is_dir())
        if len(members) != 1:
            raise RuntimeError(f"ARCHIVE_MEMBER_COUNT_INVALID:{path}")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(handle, header=None, names=COLUMNS)
    numeric_open = pd.to_numeric(frame["open_time_ms"], errors="coerce")
    frame = frame.loc[numeric_open.notna()].copy()
    frame["open_time_ms"] = numeric_open.loc[numeric_open.notna()].astype("int64")
    for column in OUTPUT_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame.loc[:, OUTPUT_COLUMNS]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    start_ms = int(pd.Timestamp(args.start).timestamp() * 1_000)
    end_ms = int(pd.Timestamp(args.end).timestamp() * 1_000)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "aegis-strategy-router-retrospective-candles-v1",
        "start_inclusive": pd.Timestamp(args.start).isoformat(),
        "end_exclusive": pd.Timestamp(args.end).isoformat(),
        "symbols": {},
        "sealed_holdout_rows_loaded": 0,
    }
    for symbol in sorted(value.strip().upper() for value in args.symbols.split(",") if value.strip()):
        directory = args.archive_root / "futures" / "um" / "monthly" / "klines" / symbol / "1m"
        archives = tuple(sorted(directory.glob(f"{symbol}-1m-*.zip")))
        if not archives:
            raise RuntimeError(f"NO_ARCHIVES:{symbol}")
        frame = pd.concat((load_archive(path) for path in archives), ignore_index=True)
        frame = frame.loc[
            frame["open_time_ms"].ge(start_ms) & frame["open_time_ms"].lt(end_ms)
        ].sort_values("open_time_ms", kind="mergesort", ignore_index=True)
        duplicates = int(frame["open_time_ms"].duplicated().sum())
        if duplicates:
            raise RuntimeError(f"DUPLICATE_OPEN_TIME:{symbol}:{duplicates}")
        gaps = int(frame["open_time_ms"].diff().dropna().ne(60_000).sum())
        if gaps:
            raise RuntimeError(f"CANDLE_GAPS:{symbol}:{gaps}")
        output = args.output_root / f"{symbol}_1m.parquet"
        temporary = output.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(output)
        manifest["symbols"][symbol] = {
            "archives": [
                {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in archives
            ],
            "rows": len(frame),
            "first_open_ms": int(frame.iloc[0].open_time_ms),
            "last_open_ms": int(frame.iloc[-1].open_time_ms),
            "duplicates": duplicates,
            "gaps": gaps,
            "parquet": str(output),
            "parquet_sha256": sha256_file(output),
        }
        print(json.dumps({"symbol": symbol, "rows": len(frame), "gaps": gaps}), flush=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path = args.output_root / "dataset_manifest.json"
    manifest_path.write_text(payload, encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
