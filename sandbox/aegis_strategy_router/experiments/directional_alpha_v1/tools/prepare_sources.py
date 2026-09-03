#!/usr/bin/env python3
"""Normalize frozen Binance archives into causal one-minute parquet sources."""

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
sys.path.insert(0, str(REPOSITORY / "src"))

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


def load_archive(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise RuntimeError(f"ARCHIVE_MEMBER_COUNT:{path}")
        with archive.open(members[0]) as raw:
            frame = pd.read_csv(TextIOWrapper(raw), names=COLUMNS, header=None)
    if not pd.to_numeric(frame.open_time_ms, errors="coerce").notna().all():
        frame = frame.iloc[1:].copy()
    numeric = COLUMNS[:-1]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    frame.open_time_ms = frame.open_time_ms.astype("int64")
    frame.close_time_ms = frame.close_time_ms.astype("int64")
    frame.trade_count = frame.trade_count.astype("int64")
    return frame.drop(columns="ignore")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "config/preregistration_v1.json")
    parser.add_argument("--raw", type=Path, default=REPOSITORY / "data/directional_alpha_v1/raw")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "data/directional_alpha_v1/candles_1m")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "directional-alpha-v1-source-manifest", "symbols": {}}
    for symbol in config["symbols"]:
        paths = sorted((args.raw / "futures/um/monthly/klines" / symbol / "1m").glob("*.zip"))
        if len(paths) != 15:
            raise RuntimeError(f"ARCHIVE_COVERAGE:{symbol}:{len(paths)}")
        frame = pd.concat([load_archive(path) for path in paths], ignore_index=True)
        duplicate_count = int(frame.open_time_ms.duplicated().sum())
        frame = frame.drop_duplicates("open_time_ms", keep="first").sort_values("open_time_ms", kind="mergesort")
        deltas = frame.open_time_ms.diff().dropna()
        gaps = int(deltas.ne(60_000).sum())
        if duplicate_count:
            raise RuntimeError(f"SOURCE_DUPLICATES:{symbol}:{duplicate_count}")
        original_rows = len(frame)
        cropped_before_ms = None
        if gaps:
            # Start a new genuinely contiguous segment after the last gap.
            # No missing observation is filled. Warmup must rebuild naturally.
            last_gap_index = int(deltas[deltas.ne(60_000)].index.max())
            cropped_before_ms = int(frame.loc[last_gap_index, "open_time_ms"])
            frame = frame.loc[frame.open_time_ms.ge(cropped_before_ms)].copy()
            if frame.open_time_ms.diff().dropna().ne(60_000).any():
                raise RuntimeError(f"SOURCE_CROP_NOT_CONTIGUOUS:{symbol}")
        destination = args.output / f"{symbol}_1m.parquet"
        frame.to_parquet(destination, index=False, compression="zstd")
        manifest["symbols"][symbol] = {
            "rows": len(frame), "original_rows": original_rows,
            "source_gaps_before_causal_crop": gaps, "cropped_before_ms": cropped_before_ms,
            "first_open_ms": int(frame.open_time_ms.iloc[0]),
            "last_open_ms": int(frame.open_time_ms.iloc[-1]), "gaps": 0,
            "duplicates": duplicate_count, "parquet": str(destination.relative_to(REPOSITORY)),
            "parquet_sha256": sha256(destination),
            "archives": [{"path": str(path.relative_to(REPOSITORY)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in paths],
        }
    manifest_path = args.output / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
