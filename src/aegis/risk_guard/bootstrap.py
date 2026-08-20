"""Operational bootstrap for the frozen E4 one-minute candle cache.

Run with ``python -m aegis.risk_guard.bootstrap``.  This module deliberately
has no import-time network or filesystem side effects.
"""

from __future__ import annotations

import argparse
import email.utils
import fcntl
import json
import logging
import os
import random
import tempfile
import time
import urllib.error
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import pandas as pd
import requests

from aegis.research.binance_public_archive import (
    ArchiveRequest,
    BinancePublicArchiveClient,
    PublicArchiveError,
)

from .feature_bridge import FROZEN_E4_UNIVERSE
from .market_snapshot import (
    BINANCE_FUTURES_REST,
    BINANCE_LIMIT_PER_REQUEST,
    DEFAULT_DURABLE_CACHE_ROOT,
    DEFAULT_HISTORY_SEED_ROOT,
    KLINE_ENDPOINT,
    REQUEST_TIMEOUT_S,
)

MINUTE_MS = 60_000
REST_GAP_SECONDS = 0.5
BINANCE_WEIGHT_LIMIT_PER_MINUTE = 2_400
BINANCE_WEIGHT_SOFT_LIMIT = 2_000
MANIFEST_NAME = "bootstrap_manifest.json"
LOCK_NAME = ".bootstrap.lock"
KLINE_SCHEMA = (
    "open_time_ms", "open", "high", "low", "close", "volume",
    "close_time_ms", "quote_volume", "trades", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore",
)
CANDLE_COLUMNS = (
    "open_time_ms", "open", "high", "low", "close", "volume",
    "taker_buy_volume",
)
UNIVERSE = tuple(sorted(FROZEN_E4_UNIVERSE))
logger = logging.getLogger(__name__)


class BootstrapError(RuntimeError):
    """A bootstrap invariant or external data boundary failed."""


def target_decision_at(now: datetime | None = None) -> datetime:
    """Return current UTC time floored to E4's five-minute boundary."""
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    return value.replace(minute=value.minute // 5 * 5, second=0, microsecond=0)


def _timestamp_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _month(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m")


def _next_month_start(ms: int) -> int:
    value = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    if value.month == 12:
        result = datetime(value.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        result = datetime(value.year, value.month + 1, 1, tzinfo=timezone.utc)
    return _timestamp_ms(result)


def _retry_after(headers: Mapping[str, Any], now: Callable[[], float]) -> float | None:
    raw = next((str(v) for k, v in headers.items() if k.lower() == "retry-after"), None)
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(raw)
            return max(0.0, parsed.timestamp() - now())
        except (TypeError, ValueError, OverflowError):
            return None


class RateLimitedRestClient:
    """Sequential Binance REST client with bounded, observable retries."""

    def __init__(
        self,
        *,
        http_get: Callable[..., Any] = requests.get,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
        now: Callable[[], float] = time.time,
        max_retries: int = 5,
        gap_seconds: float = REST_GAP_SECONDS,
        base_backoff: float = 1.0,
        ban_delay: float = 120.0,
    ) -> None:
        self.http_get = http_get
        self.sleep = sleep
        self.jitter = jitter
        self.now = now
        self.max_retries = max_retries
        self.gap_seconds = gap_seconds
        self.base_backoff = base_backoff
        self.ban_delay = ban_delay
        self.requests = self.retries = self.count_429 = self.count_418 = 0
        self.last_used_weight = 0
        self._last_request_at: float | None = None

    @property
    def counters(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "retries": self.retries,
            "429_count": self.count_429,
            "418_count": self.count_418,
        }

    def _wait_for_slot(self) -> None:
        if self._last_request_at is not None:
            delay = self.gap_seconds - (self.now() - self._last_request_at)
            if delay > 0:
                self.sleep(delay)

    def get_json(self, url: str, *, params: Mapping[str, Any]) -> Any:
        for attempt in range(self.max_retries + 1):
            self._wait_for_slot()
            self.requests += 1
            try:
                response = self.http_get(url, params=dict(params), timeout=REQUEST_TIMEOUT_S)
                self._last_request_at = self.now()
            except (requests.Timeout, requests.ConnectionError) as exc:
                self._last_request_at = self.now()
                if attempt >= self.max_retries:
                    raise BootstrapError("REST_RETRY_EXHAUSTED") from exc
                self._retry(attempt, None, False)
                continue

            status = int(response.status_code)
            used_weight = next(
                (
                    int(value)
                    for key, value in response.headers.items()
                    if key.lower() == "x-mbx-used-weight-1m" and str(value).isdigit()
                ),
                0,
            )
            self.last_used_weight = max(self.last_used_weight, used_weight)
            if 200 <= status < 300:
                if used_weight >= BINANCE_WEIGHT_SOFT_LIMIT:
                    logger.warning(
                        "Binance weight soft limit reached: %d/%d",
                        used_weight,
                        BINANCE_WEIGHT_LIMIT_PER_MINUTE,
                    )
                    self.sleep(60.0)
                return response.json()
            if status == 429:
                self.count_429 += 1
            if status == 418:
                self.count_418 += 1
            transient = status in (418, 429) or 500 <= status < 600
            if not transient:
                raise BootstrapError(f"REST_FATAL_HTTP:{status}")
            if attempt >= self.max_retries:
                raise BootstrapError(f"REST_RETRY_EXHAUSTED:{status}")
            retry_after = _retry_after(response.headers, self.now)
            self._retry(attempt, retry_after, status == 418)
        raise AssertionError("unreachable")

    def _retry(self, attempt: int, retry_after: float | None, banned: bool) -> None:
        self.retries += 1
        exponential = self.base_backoff * (2**attempt) + self.jitter()
        delay = max(exponential, retry_after or 0.0, self.ban_delay if banned else 0.0)
        logger.warning(
            "Retrying Binance REST attempt=%d retry_after=%s backoff=%.3f banned=%s",
            attempt + 1,
            retry_after,
            delay,
            banned,
        )
        self.sleep(delay)


def parse_klines(raw: Any) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    frame = pd.DataFrame(raw, columns=KLINE_SCHEMA)
    for column in CANDLE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["open_time_ms"] = frame["open_time_ms"].astype("int64")
    return frame.loc[:, CANDLE_COLUMNS].copy()


def parse_archive(path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            with archive.open(member) as handle:
                frame = pd.read_csv(handle, header=None, dtype=str)
            if frame.shape[1] != len(KLINE_SCHEMA):
                raise BootstrapError(f"ARCHIVE_SCHEMA:{path}")
            if str(frame.iloc[0, 0]).strip().lower() in {"open_time", "open time"}:
                frame = frame.iloc[1:].reset_index(drop=True)
            frame.columns = KLINE_SCHEMA
            frames.append(parse_klines(frame.values.tolist()))
    if not frames:
        raise BootstrapError(f"ARCHIVE_EMPTY:{path}")
    return pd.concat(frames, ignore_index=True)


def _canonical(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = set(CANDLE_COLUMNS) - set(frame.columns)
    if missing:
        raise BootstrapError(f"COLUMNS_MISSING:{source}:{sorted(missing)}")
    result = frame.loc[:, CANDLE_COLUMNS].copy()
    numeric = list(CANDLE_COLUMNS)
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="raise")
    if not np.equal(result["open_time_ms"], np.floor(result["open_time_ms"])).all():
        raise BootstrapError(f"TIMESTAMP_NOT_INTEGER:{source}")
    result["open_time_ms"] = result["open_time_ms"].astype("int64")
    return result


def merge_candles(existing: pd.DataFrame, incoming: pd.DataFrame, source: str = "merge") -> tuple[pd.DataFrame, int]:
    """Merge idempotently, rejecting two different candles for one minute."""
    left = _canonical(existing, source) if not existing.empty else pd.DataFrame(columns=CANDLE_COLUMNS)
    right = _canonical(incoming, source) if not incoming.empty else pd.DataFrame(columns=CANDLE_COLUMNS)
    if left["open_time_ms"].duplicated().any() or right["open_time_ms"].duplicated().any():
        raise BootstrapError(f"DUPLICATE_ROWS:{source}")
    overlap = left.merge(right, on="open_time_ms", suffixes=("_old", "_new"))
    for column in CANDLE_COLUMNS[1:]:
        if not np.allclose(
            overlap[f"{column}_old"].to_numpy(dtype=float),
            overlap[f"{column}_new"].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            raise BootstrapError(f"CONFLICTING_CANDLE:{source}")
    only_new = right.loc[~right["open_time_ms"].isin(left["open_time_ms"])]
    if left.empty:
        merged = only_new.copy()
    elif only_new.empty:
        merged = left.copy()
    else:
        merged = pd.concat([left, only_new], ignore_index=True)
    return merged.sort_values("open_time_ms").reset_index(drop=True), len(overlap)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Publish validated parquet durably with a unique same-directory temp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        checked = pd.read_parquet(temporary)
        expected = _canonical(frame, str(path)).reset_index(drop=True)
        actual = _canonical(checked, str(path)).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _read_parts(seed_root: Path, durable_root: Path, symbol: str) -> tuple[pd.DataFrame, list[tuple[Path, pd.DataFrame]]]:
    seed_path = seed_root / f"{symbol}_1m.parquet"
    if not seed_path.is_file():
        raise BootstrapError(f"SEED_MISSING:{symbol}")
    try:
        seed = _canonical(pd.read_parquet(seed_path), str(seed_path))
        paths = sorted((durable_root / symbol).glob("*.parquet")) if (durable_root / symbol).is_dir() else []
        durable = [(path, _canonical(pd.read_parquet(path), str(path))) for path in paths]
    except (OSError, ValueError, TypeError) as exc:
        raise BootstrapError(f"PARQUET_CORRUPT:{symbol}") from exc
    return seed, durable


def _validate_frame(frame: pd.DataFrame, symbol: str) -> None:
    if frame.empty:
        raise BootstrapError(f"EMPTY_CANDLES:{symbol}")
    timestamps = frame["open_time_ms"].to_numpy(dtype=np.int64)
    if np.any(timestamps % MINUTE_MS) or np.any(np.diff(timestamps) <= 0):
        raise BootstrapError(f"TIMESTAMP_INVALID:{symbol}")
    if frame["open_time_ms"].duplicated().any():
        raise BootstrapError(f"DUPLICATE_ROWS:{symbol}")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) != MINUTE_MS):
        raise BootstrapError(f"CANDLE_GAP:{symbol}")
    values = frame.loc[:, CANDLE_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise BootstrapError(f"NON_FINITE_CANDLE:{symbol}")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise BootstrapError(f"NON_POSITIVE_CANDLE:{symbol}")
    if (frame["volume"] < 0).any():
        raise BootstrapError(f"NEGATIVE_VOLUME:{symbol}")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any() or (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise BootstrapError(f"OHLC_INCOHERENT:{symbol}")
    if (frame["taker_buy_volume"] < 0).any() or (frame["taker_buy_volume"] > frame["volume"]).any():
        raise BootstrapError(f"TAKER_VOLUME_INVALID:{symbol}")


def validate_bootstrap(seed_root: Path, durable_root: Path, target: datetime) -> dict[str, Any]:
    """Validate truth from parquet, never from the diagnostics manifest."""
    target_ms = _timestamp_ms(target) - MINUTE_MS
    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for symbol in UNIVERSE:
        try:
            seed, durable_parts = _read_parts(seed_root, durable_root, symbol)
            _validate_frame(seed.sort_values("open_time_ms").reset_index(drop=True), symbol)
            combined = seed
            duplicates = 0
            for path, part in durable_parts:
                _validate_frame(part.sort_values("open_time_ms").reset_index(drop=True), symbol)
                combined, count = merge_candles(combined, part, str(path))
                duplicates += count
            _validate_frame(combined, symbol)
            last = int(combined["open_time_ms"].iloc[-1])
            if last != target_ms:
                raise BootstrapError(f"TARGET_NOT_REACHED:{symbol}:{last}:{target_ms}")
            results[symbol] = {
                "ready": True,
                "first_timestamp": _iso(int(combined["open_time_ms"].iloc[0])),
                "last_timestamp": _iso(last),
                "row_count": int(len(combined)),
                "month_count": int(len(durable_parts)),
                "gaps": 0,
                "duplicate_rows": duplicates,
            }
        except Exception as exc:  # all corrupt parquet/schema failures become closed diagnostics
            failures[symbol] = str(exc)
            results[symbol] = {"ready": False, "error": str(exc)}
    ready = not failures and set(results) == set(UNIVERSE)
    return {
        "ready": ready,
        "target_decision_at": target.isoformat(),
        "symbols_complete": sum(item.get("ready") is True for item in results.values()),
        "symbols_expected": len(UNIVERSE),
        "total_candles": sum(int(item.get("row_count", 0)) for item in results.values()),
        "gaps": sum(int(item.get("gaps", 0)) for item in results.values()),
        "duplicates": sum(int(item.get("duplicate_rows", 0)) for item in results.values()),
        "symbols": results,
        "errors": failures,
    }


@contextmanager
def process_lock(durable_root: Path) -> Iterator[None]:
    durable_root.mkdir(parents=True, exist_ok=True)
    path = durable_root / LOCK_NAME
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BootstrapError("BOOTSTRAP_ALREADY_RUNNING") from exc
        yield


class Bootstrapper:
    def __init__(
        self,
        seed_root: Path,
        durable_root: Path,
        *,
        target: datetime | None = None,
        rest: RateLimitedRestClient | None = None,
        archive: BinancePublicArchiveClient | None = None,
    ) -> None:
        self.seed_root = Path(seed_root)
        self.durable_root = Path(durable_root)
        self.target = target_decision_at(target)
        self.target_ms = _timestamp_ms(self.target) - MINUTE_MS
        self.rest = rest or RateLimitedRestClient()
        self.archive = archive or BinancePublicArchiveClient()
        self.archive_root = self.durable_root.parent / f"{self.durable_root.name}_archives"
        self.manifest_path = self.durable_root / MANIFEST_NAME
        self.manifest: dict[str, Any] = {"ready": False, "symbols": {}}
        self._counter_starts: dict[str, dict[str, int]] = {}
        self.archive_requests = 0
        self.archive_retries = 0

    def _publish_manifest(self) -> None:
        self.manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(self.manifest_path, self.manifest)

    def _state(self, symbol: str, status: str, seed_last: int | None, durable_last: int | None, **extra: Any) -> None:
        baseline = self._counter_starts.setdefault(symbol, self.rest.counters.copy())
        counters = {
            key: value - baseline[key] for key, value in self.rest.counters.items()
        }
        item = {
            "seed_last_timestamp": _iso(seed_last),
            "durable_last_timestamp": _iso(durable_last),
            "target_timestamp": _iso(self.target_ms),
            "months_complete": extra.pop("months_complete", []),
            "current_month": _month(self.target_ms),
            **counters,
            "archive_requests": self.archive_requests,
            "archive_retries": self.archive_retries,
            "gaps_detected": extra.pop("gaps_detected", 0),
            "duplicate_rows": extra.pop("duplicate_rows", 0),
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        self.manifest["symbols"][symbol] = item
        self._publish_manifest()

    def _write_month(self, symbol: str, month: str, incoming: pd.DataFrame) -> int:
        path = self.durable_root / symbol / f"{month}.parquet"
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=CANDLE_COLUMNS)
        merged, duplicates = merge_candles(existing, incoming, f"{symbol}:{month}")
        _validate_frame(merged, symbol)
        atomic_write_parquet(path, merged)
        return duplicates

    def _archive_month(self, symbol: str, month: str, start: int, end: int) -> pd.DataFrame:
        request = ArchiveRequest("futures/um", "klines", symbol, month, "1m")
        for attempt in range(self.rest.max_retries + 1):
            self.archive_requests += 1
            try:
                evidence = self.archive.download(
                    request,
                    self.archive_root,
                )
                break
            except PublicArchiveError:
                raise
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                if status == 429:
                    self.rest.count_429 += 1
                if status == 418:
                    self.rest.count_418 += 1
                if status not in (418, 429) and not 500 <= status < 600:
                    raise BootstrapError(
                        f"ARCHIVE_FATAL_HTTP:{symbol}:{month}:{status}"
                    ) from exc
                if attempt >= self.rest.max_retries:
                    raise BootstrapError(
                        f"ARCHIVE_RETRY_EXHAUSTED:{symbol}:{month}"
                    ) from exc
                self.archive_retries += 1
                delay = max(
                    self.rest.base_backoff * (2**attempt) + self.rest.jitter(),
                    _retry_after(exc.headers or {}, self.rest.now) or 0.0,
                    self.rest.ban_delay if status == 418 else 0.0,
                )
                self.rest.sleep(delay)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt >= self.rest.max_retries:
                    raise BootstrapError(
                        f"ARCHIVE_RETRY_EXHAUSTED:{symbol}:{month}"
                    ) from exc
                self.archive_retries += 1
                delay = self.rest.base_backoff * (2**attempt) + self.rest.jitter()
                self.rest.sleep(delay)
        frame = parse_archive(Path(evidence.file))
        return frame.loc[frame["open_time_ms"].between(start, end)].reset_index(drop=True)

    def _rest_range(self, symbol: str, start: int, end: int) -> Iterator[pd.DataFrame]:
        cursor = start
        url = f"{BINANCE_FUTURES_REST}{KLINE_ENDPOINT}"
        while cursor <= end:
            raw = self.rest.get_json(url, params={
                "symbol": symbol, "interval": "1m", "limit": BINANCE_LIMIT_PER_REQUEST,
                "startTime": cursor, "endTime": end,
            })
            page = parse_klines(raw)
            page = page.loc[page["open_time_ms"].between(cursor, end)].reset_index(drop=True)
            if page.empty:
                raise BootstrapError(f"REST_EMPTY:{symbol}:{cursor}")
            yield page
            next_cursor = int(page["open_time_ms"].max()) + MINUTE_MS
            if next_cursor <= cursor:
                raise BootstrapError(f"REST_NON_ADVANCING:{symbol}:{cursor}")
            cursor = next_cursor

    def sync_symbol(self, symbol: str) -> None:
        if symbol not in UNIVERSE:
            raise BootstrapError(f"UNIVERSE_SYMBOL_INVALID:{symbol}")
        seed, durable_parts = _read_parts(self.seed_root, self.durable_root, symbol)
        _validate_frame(seed.sort_values("open_time_ms").reset_index(drop=True), symbol)
        seed_last = int(seed["open_time_ms"].max())
        combined = seed
        duplicate_rows = 0
        months_complete: list[str] = []
        for path, part in durable_parts:
            _validate_frame(part.sort_values("open_time_ms").reset_index(drop=True), symbol)
            combined, count = merge_candles(combined, part, str(path))
            duplicate_rows += count
            if _next_month_start(int(part["open_time_ms"].min())) - MINUTE_MS == int(part["open_time_ms"].max()):
                months_complete.append(path.stem)
        durable_last = max((int(part["open_time_ms"].max()) for _, part in durable_parts if not part.empty), default=None)
        next_ms = int(combined["open_time_ms"].max()) + MINUTE_MS
        print(f"[{symbol}] seed through {_iso(seed_last)}", flush=True)
        self._state(symbol, "SYNCING", seed_last, durable_last, months_complete=months_complete, duplicate_rows=duplicate_rows)
        while next_ms <= self.target_ms:
            month = _month(next_ms)
            print(f"[{symbol}] syncing {month}", flush=True)
            month_end = _next_month_start(next_ms) - MINUTE_MS
            end = min(month_end, self.target_ms)
            if month < _month(self.target_ms):
                chunk = self._archive_month(symbol, month, next_ms, end)
                if chunk.empty:
                    raise BootstrapError(f"ARCHIVE_MISSING_RANGE:{symbol}:{month}")
                duplicate_rows += self._write_month(symbol, month, chunk)
                next_ms = int(chunk["open_time_ms"].max()) + MINUTE_MS
                if next_ms > month_end:
                    months_complete.append(month)
                durable_last = next_ms - MINUTE_MS
                self._state(symbol, "SYNCING", seed_last, durable_last, months_complete=sorted(set(months_complete)), duplicate_rows=duplicate_rows)
            else:
                for page in self._rest_range(symbol, next_ms, end):
                    duplicate_rows += self._write_month(symbol, month, page)
                    next_ms = int(page["open_time_ms"].max()) + MINUTE_MS
                    durable_last = next_ms - MINUTE_MS
                    self._state(symbol, "SYNCING", seed_last, durable_last, months_complete=sorted(set(months_complete)), duplicate_rows=duplicate_rows)
        self._state(symbol, "COMPLETE", seed_last, durable_last, months_complete=sorted(set(months_complete)), duplicate_rows=duplicate_rows)
        print(f"[{symbol}] caught up {_iso(self.target_ms)}", flush=True)

    def run(self, *, validate: bool = True) -> dict[str, Any]:
        self.durable_root.mkdir(parents=True, exist_ok=True)
        for symbol in UNIVERSE:
            try:
                self._state(symbol, "PENDING", None, None)
                self.sync_symbol(symbol)
            except Exception as exc:
                prior = self.manifest["symbols"].get(symbol, {})
                message = str(exc)
                self._state(
                    symbol,
                    "ERROR",
                    _parse_iso_ms(prior.get("seed_last_timestamp")),
                    _parse_iso_ms(prior.get("durable_last_timestamp")),
                    gaps_detected=int("GAP" in message),
                    duplicate_rows=int("DUPLICATE" in message),
                    error=message,
                )
                raise
        result = validate_bootstrap(self.seed_root, self.durable_root, self.target) if validate else {"ready": False}
        for symbol, error in result.get("errors", {}).items():
            item = self.manifest["symbols"][symbol]
            item.update({
                "status": "ERROR",
                "error": error,
                "gaps_detected": int("GAP" in error),
                "duplicate_rows": max(item["duplicate_rows"], int("DUPLICATE" in error)),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
        self.manifest["ready"] = bool(result["ready"])
        self.manifest["target_decision_at"] = self.target.isoformat()
        self.manifest["symbols_complete"] = sum(
            item.get("status") == "COMPLETE"
            for item in self.manifest["symbols"].values()
        )
        self.manifest["requests"] = self.rest.requests + self.archive_requests
        self.manifest["retries"] = self.rest.retries + self.archive_retries
        self.manifest["429_count"] = self.rest.count_429
        self.manifest["418_count"] = self.rest.count_418
        self._publish_manifest()
        return result


def read_status(durable_root: Path) -> dict[str, Any]:
    path = durable_root / MANIFEST_NAME
    if not path.is_file():
        return {"ready": False, "status": "NOT_STARTED"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapError("MANIFEST_CORRUPT") from exc


def _parse_iso_ms(value: Any) -> int | None:
    if not value:
        return None
    return int(pd.Timestamp(value).timestamp() * 1000)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, default=DEFAULT_HISTORY_SEED_ROOT)
    parser.add_argument("--durable-root", type=Path, default=DEFAULT_DURABLE_CACHE_ROOT)
    parser.add_argument("--resume", action="store_true", help="resume from durable parquet (the default behavior)")
    parser.add_argument("--validate", action="store_true", help="strictly validate after synchronization")
    parser.add_argument("--validate-only", action="store_true", help="validate local parquet without network access")
    parser.add_argument("--status", action="store_true", help="print diagnostics without modifying data")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parser().parse_args(argv)
    target = target_decision_at()
    if args.status:
        result = read_status(args.durable_root)
    elif args.validate_only:
        result = validate_bootstrap(args.seed_root, args.durable_root, target)
    else:
        with process_lock(args.durable_root):
            result = Bootstrapper(args.seed_root, args.durable_root, target=target).run(validate=args.validate)
    print(json.dumps(result, sort_keys=True, indent=2))
    if result.get("ready") is True:
        print("E4_BOOTSTRAP_COMPLETE")
    return 0 if result.get("ready", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
