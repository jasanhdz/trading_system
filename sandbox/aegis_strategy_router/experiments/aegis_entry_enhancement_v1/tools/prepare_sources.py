#!/usr/bin/env python3
"""Merge public warmup archives with the existing neutral Live candle source."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from io import TextIOWrapper
from pathlib import Path

import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parents[3]
COLUMNS = [
    "open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms",
    "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_frame(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise RuntimeError(f"ARCHIVE_MEMBER_COUNT:{path}")
        with archive.open(members[0]) as raw:
            frame = pd.read_csv(TextIOWrapper(raw), names=COLUMNS, header=None)
    if not pd.to_numeric(frame.open_time_ms, errors="coerce").notna().all():
        frame = frame.iloc[1:].copy()
    frame[COLUMNS[:-1]] = frame[COLUMNS[:-1]].apply(pd.to_numeric, errors="raise")
    frame["open_time_ms"] = frame.open_time_ms.astype("int64")
    frame["close_time_ms"] = frame.close_time_ms.astype("int64")
    frame["trade_count"] = frame.trade_count.astype("int64")
    return frame.drop(columns="ignore")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "config/preregistration_v1.json")
    parser.add_argument("--raw", type=Path, default=REPOSITORY / "data/aegis_entry_enhancement_v1/raw")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "data/aegis_entry_enhancement_v1/candles_1m")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    live_root = REPOSITORY / config["sources"]["live_candles"]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "aegis-entry-enhancement-v1-source-manifest", "symbols": {}}
    for symbol in config["symbols"]:
        paths = sorted((args.raw / "futures/um/monthly/klines" / symbol / "1m").glob("*.zip"))
        if len(paths) != len(config["sources"]["warmup_public_months"]):
            raise RuntimeError(f"WARMUP_ARCHIVE_COVERAGE:{symbol}:{len(paths)}")
        warmup = pd.concat([archive_frame(path) for path in paths], ignore_index=True)
        live_path = live_root / f"{symbol}_1m.parquet"
        live = pd.read_parquet(live_path)
        frame = pd.concat([warmup, live], ignore_index=True).sort_values("open_time_ms", kind="mergesort")
        duplicates = frame.loc[frame.open_time_ms.duplicated(keep=False)]
        if len(duplicates):
            value_columns = [column for column in frame if column != "open_time_ms"]
            conflicts = duplicates.groupby("open_time_ms")[value_columns].nunique(dropna=False).gt(1).any(axis=1)
            if conflicts.any():
                raise RuntimeError(f"CONFLICTING_DUPLICATE:{symbol}:{int(conflicts.sum())}")
        frame = frame.drop_duplicates("open_time_ms", keep="last")
        gaps = int(frame.open_time_ms.diff().dropna().ne(60_000).sum())
        if gaps:
            raise RuntimeError(f"SOURCE_GAPS:{symbol}:{gaps}")
        destination = args.output / f"{symbol}_1m.parquet"
        frame.to_parquet(destination, index=False, compression="zstd")
        manifest["symbols"][symbol] = {
            "rows": len(frame), "first_open_ms": int(frame.open_time_ms.iloc[0]),
            "last_open_ms": int(frame.open_time_ms.iloc[-1]), "gaps": gaps,
            "parquet": str(destination.relative_to(REPOSITORY)), "parquet_sha256": sha256(destination),
            "live_source_sha256": sha256(live_path),
            "archives": [{"path": str(path.relative_to(REPOSITORY)), "sha256": sha256(path)} for path in paths],
        }
    destination = args.output / "dataset_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
