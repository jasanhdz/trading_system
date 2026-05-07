from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG


TURBO_MAX_FEATURE_AGE_SECONDS = int(os.getenv("TURBO_MAX_FEATURE_AGE_SECONDS", "900"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for candidate in (
        text,
        text.replace("Z", "+00:00"),
    ):
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            continue
    return None


def _normalize_path(path: Path | str) -> Path:
    return Path(path)


def normalize_turbo_symbol(symbol: str | None = None) -> str:
    raw = symbol or DEFAULT_TURBO_CONFIG.symbol
    return str(raw).replace("/", "").strip().upper()


def turbo_symbol_data_dir(symbol: str | None = None) -> Path:
    return DEFAULT_TURBO_CONFIG.data_dir / "turbo" / normalize_turbo_symbol(symbol)


def turbo_symbol_model_dir(symbol: str | None = None) -> Path:
    return DEFAULT_TURBO_CONFIG.model_dir / normalize_turbo_symbol(symbol)


def turbo_snapshot_path(lookback_days: int, symbol: str | None = None) -> Path:
    legacy_path = DEFAULT_TURBO_CONFIG.data_dir / f"turbo_recent_{lookback_days}d.npz"
    if symbol is None:
        return legacy_path
    symbol_path = turbo_symbol_data_dir(symbol) / f"turbo_recent_{lookback_days}d.npz"
    if normalize_turbo_symbol(symbol) == normalize_turbo_symbol(DEFAULT_TURBO_CONFIG.symbol) and legacy_path.exists():
        return legacy_path
    return symbol_path


def load_turbo_snapshot_status(path: Path | str, include_sample_count: bool = False) -> dict[str, Any]:
    snapshot_path = _normalize_path(path)
    exists = snapshot_path.exists()
    snapshot_mtime = None
    snapshot_age_seconds = None
    feature_timestamp = None
    feature_age_seconds = None
    sample_count = 0
    last_ts = None
    error = None
    is_fresh = False

    if exists:
        try:
            stat = snapshot_path.stat()
            snapshot_mtime_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            snapshot_mtime = snapshot_mtime_dt.isoformat()
            snapshot_age_seconds = max(0.0, (_utc_now() - snapshot_mtime_dt).total_seconds())
        except Exception as exc:
            error = f"snapshot_stat_error:{exc!r}"
        try:
            with np.load(snapshot_path, allow_pickle=True) as data:
                feature_timestamp_value = data.get("feature_timestamp")
                timestamps = data.get("timestamp")
                if include_sample_count:
                    x = data.get("X")
                    if x is not None:
                        sample_count = int(len(x))
                if feature_timestamp_value is not None:
                    feature_dt = _parse_timestamp(feature_timestamp_value.item() if hasattr(feature_timestamp_value, "item") else feature_timestamp_value)
                    if feature_dt is not None:
                        feature_timestamp = feature_dt.isoformat()
                        feature_age_seconds = max(0.0, (_utc_now() - feature_dt).total_seconds())
                        is_fresh = feature_age_seconds <= TURBO_MAX_FEATURE_AGE_SECONDS
                if timestamps is not None and len(timestamps) > 0:
                    last_ts = str(timestamps[-1])
                    if feature_timestamp is None:
                        feature_dt = _parse_timestamp(last_ts)
                    else:
                        feature_dt = None
                    if feature_dt is not None:
                        feature_timestamp = feature_dt.isoformat()
                        feature_age_seconds = max(0.0, (_utc_now() - feature_dt).total_seconds())
                        is_fresh = feature_age_seconds <= TURBO_MAX_FEATURE_AGE_SECONDS
        except Exception as exc:
            error = f"snapshot_read_error:{exc!r}"

    freshness = {
        "path": str(snapshot_path),
        "exists": exists,
        "snapshot_mtime": snapshot_mtime,
        "snapshot_age_seconds": snapshot_age_seconds,
        "feature_timestamp": feature_timestamp,
        "feature_age_seconds": feature_age_seconds,
        "max_feature_age_seconds": TURBO_MAX_FEATURE_AGE_SECONDS,
        "is_fresh": bool(is_fresh),
        "stale": not bool(is_fresh),
        "sample_count": sample_count,
    }
    if error:
        freshness["error"] = error
    freshness["last_ts"] = last_ts
    return freshness


def load_turbo_freshness(lookback_days: int = 7, symbol: str | None = None) -> dict[str, Any]:
    return load_turbo_snapshot_status(turbo_snapshot_path(lookback_days, symbol))


def select_freshest_turbo_snapshot(symbol: str | None = None) -> dict[str, Any]:
    candidates = [load_turbo_snapshot_status(turbo_snapshot_path(int(lookback_days), symbol), include_sample_count=False) for lookback_days in DEFAULT_TURBO_CONFIG.lookback_days]
    candidates = [candidate for candidate in candidates if candidate.get("exists")]
    if not candidates:
        return {
            "path": None,
            "freshness": load_turbo_snapshot_status(turbo_snapshot_path(int(DEFAULT_TURBO_CONFIG.lookback_days[0]), symbol), include_sample_count=False)
            if DEFAULT_TURBO_CONFIG.lookback_days
            else {
                "path": None,
                "exists": False,
                "snapshot_mtime": None,
                "snapshot_age_seconds": None,
                "feature_timestamp": None,
                "feature_age_seconds": None,
                "max_feature_age_seconds": TURBO_MAX_FEATURE_AGE_SECONDS,
                "is_fresh": False,
                "sample_count": 0,
                "last_ts": None,
            },
        }
    candidates.sort(
        key=lambda item: (
            item.get("feature_timestamp") is not None,
            item.get("feature_timestamp") or "",
            item.get("snapshot_mtime") or "",
            -int(item.get("lookback_days") or 0),
        ),
        reverse=True,
    )
    selected = candidates[0]
    selected["path"] = selected.get("path") or str(turbo_snapshot_path(int(selected.get("lookback_days") or 7), symbol))
    return {"path": selected["path"], "freshness": selected}


def save_npz_atomic(path: Path | str, **arrays: Any) -> Path:
    target = _normalize_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".{target.name}.tmp"
    try:
        with tmp_path.open("wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp_path, target)
        return target
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
