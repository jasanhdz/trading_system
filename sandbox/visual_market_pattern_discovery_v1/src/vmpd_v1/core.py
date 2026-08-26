from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write(canonical_json(row))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def load_candles(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"open_time_ms", "close_time_ms", *OHLCV}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    out = df[["open_time_ms", "close_time_ms", *OHLCV]].copy()
    out["open_time_ms"] = out["open_time_ms"].astype("int64")
    out["close_time_ms"] = out["close_time_ms"].astype("int64")
    out = out.sort_values("open_time_ms", kind="stable").reset_index(drop=True)
    return out


def audit_candles(df: pd.DataFrame) -> dict[str, Any]:
    opens = df.open_time_ms.to_numpy(np.int64)
    unique, counts = np.unique(opens, return_counts=True)
    duplicates = int(np.maximum(counts - 1, 0).sum())
    expected = np.arange(unique.min(), unique.max() + 60_000, 60_000, dtype=np.int64)
    gaps = np.setdiff1d(expected, unique, assume_unique=True)
    invalid = int((df.close_time_ms.to_numpy(np.int64) != opens + 59_999).sum())
    return {
        "rows": int(len(df)), "duplicates": duplicates, "gaps": int(len(gaps)),
        "first_open": iso_utc(int(opens.min())), "last_close": iso_utc(int(df.close_time_ms.max())),
        "invalid_close_times": invalid,
        "first_gap": iso_utc(int(gaps[0])) if len(gaps) else None,
    }


def complete_window(dfs: dict[str, pd.DataFrame], days: int) -> tuple[int, int]:
    common_last_close = min(int(df.close_time_ms.max()) for df in dfs.values())
    last_day = pd.Timestamp(common_last_close, unit="ms", tz="UTC").floor("D")
    if common_last_close < int((last_day + pd.Timedelta(days=1)).timestamp() * 1000) - 1:
        last_day -= pd.Timedelta(days=1)
    end_ms = int((last_day + pd.Timedelta(days=1)).timestamp() * 1000) - 1
    start_ms = int((last_day - pd.Timedelta(days=days - 1)).timestamp() * 1000)
    return start_ms, end_ms


def causal_bars(df: pd.DataFrame, decision_at_ms: int, timeframe_minutes: int, count: int) -> pd.DataFrame:
    """Aggregate only completed 1m bars; last higher-TF candle may be partial."""
    closes = df.close_time_ms.to_numpy(np.int64)
    end = int(np.searchsorted(closes, decision_at_ms, side="right"))
    # One extra bucket covers an arbitrary phase within the current partial candle.
    start = max(0, end - (count + 1) * timeframe_minutes)
    past = df.iloc[start:end].copy()
    if past.empty:
        return past
    bucket_ms = timeframe_minutes * 60_000
    past["bucket"] = (past.open_time_ms // bucket_ms) * bucket_ms
    bars = past.groupby("bucket", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        source_last_close_ms=("close_time_ms", "max"), source_minutes=("open", "size"),
    ).reset_index(names="open_time_ms")
    return bars.tail(count).reset_index(drop=True)


def normalize_panel(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    lo, hi = float(out.low.min()), float(out.high.max())
    span = max(hi - lo, max(abs(hi), 1.0) * 1e-9)
    for col in ["open", "high", "low", "close"]:
        out[col] = (out[col].astype(float) - lo) / span
    vmax = float(out.volume.max()) if len(out) else 0.0
    out["volume"] = out.volume.astype(float) / max(vmax, 1e-12)
    out.attrs.update({"price_min": lo, "price_max": hi, "price_span": span})
    return out


def frame_id(decision_at_ms: int) -> str:
    return f"SUIUSDT_{decision_at_ms}"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def temporal_split_days(times: Iterable[str]) -> np.ndarray:
    ts = pd.to_datetime(list(times), utc=True)
    days = np.array(sorted(pd.unique(ts.floor("D"))))
    n = len(days)
    train_end, valid_end = math.floor(n * .70), math.floor(n * .85)
    mapping = {d: ("TRAIN" if i < train_end else "VALIDATION" if i < valid_end else "TEST") for i, d in enumerate(days)}
    return np.array([mapping[d] for d in ts.floor("D")], dtype=object)


def collapse_episodes(assignments: list[dict[str, Any]], gap_minutes: int = 15) -> list[dict[str, Any]]:
    rows = sorted((r for r in assignments if r["pattern_id"] != "NOISE"), key=lambda r: r["decision_at"])
    active: dict[str, dict[str, Any]] = {}
    episodes: list[dict[str, Any]] = []
    gap_ms = gap_minutes * 60_000
    for row in rows:
        pid, t = row["pattern_id"], int(pd.Timestamp(row["decision_at"]).timestamp() * 1000)
        prior = active.get(pid)
        if prior is None or t - prior["last_ms"] > gap_ms:
            episode = {"episode_id": f"{pid}_E{sum(e['pattern_id']==pid for e in episodes)+1:05d}",
                       "pattern_id": pid, "started_at": row["decision_at"], "ended_at": row["decision_at"],
                       "frame_count": 1, "onset_frame_id": row["frame_id"], "last_ms": t}
            episodes.append(episode); active[pid] = episode
        else:
            prior["ended_at"] = row["decision_at"]; prior["frame_count"] += 1; prior["last_ms"] = t
    for episode in episodes:
        start = pd.Timestamp(episode["started_at"]); end = pd.Timestamp(episode["ended_at"])
        episode["duration_minutes"] = int((end - start).total_seconds() / 60) + 3
        episode.pop("last_ms", None)
    return episodes


def artifact_manifest(root: Path, config_hash: str, stage: str) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "m1_manifest.json":
            files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema_version": 1, "stage": stage, "config_sha256": config_hash,
            "trading_authority": False, "files": files}
