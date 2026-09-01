"""Small cross-process Binance weight budget shared with the TS runtime."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

CAPACITY = 1800.0
REFILL_PER_SECOND = CAPACITY / 60.0
CRITICAL_RESERVE = 300.0
DEFAULT_DB = Path(os.getenv("TMPDIR", "/tmp")) / "trading_system-binance-shared-rate-limit.sqlite3"


class SharedBinanceRateLimiter:
    def __init__(self, process_name: str, db_path: Path | None = None) -> None:
        self.process_name = process_name
        self.db_path = db_path or Path(os.getenv("BINANCE_SHARED_RATE_LIMIT_DB", DEFAULT_DB))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics = {"grants": 0, "blocked": 0, "rate_limit_events": 0}
        with self._connection() as connection:
            connection.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS binance_rate_limit_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    tokens REAL NOT NULL,
                    updated_at INTEGER NOT NULL,
                    cooldown_until INTEGER NOT NULL DEFAULT 0
                );
                INSERT OR IGNORE INTO binance_rate_limit_state(id, tokens, updated_at, cooldown_until)
                VALUES (1, {CAPACITY}, CAST(strftime('%s','now') AS INTEGER) * 1000, 0);
                CREATE TABLE IF NOT EXISTS binance_rate_limit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at INTEGER NOT NULL,
                    process_name TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    weight INTEGER NOT NULL,
                    priority TEXT NOT NULL,
                    outcome TEXT NOT NULL
                );
                """
            )

    def acquire(self, weight: int, endpoint: str, priority: str = "normal") -> None:
        requested = max(1, int(weight))
        while True:
            wait = self._try_acquire(requested, endpoint, priority)
            if wait <= 0:
                return
            self.metrics["blocked"] += 1
            time.sleep(min(wait, 5.0))

    def note_rate_limit(self, until_ms: int, status: int | None = None) -> None:
        now = int(time.time() * 1000)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT cooldown_until FROM binance_rate_limit_state WHERE id = 1"
            ).fetchone()[0]
            connection.execute(
                "UPDATE binance_rate_limit_state SET cooldown_until = ?, updated_at = ? WHERE id = 1",
                (max(int(current), int(until_ms)), now),
            )
            connection.execute(
                "INSERT INTO binance_rate_limit_events(created_at, process_name, endpoint, weight, priority, outcome) VALUES (?, ?, ?, ?, ?, ?)",
                (now, self.process_name, "unknown", 0, "normal", f"rate_limit_{status or 'unknown'}"),
            )
        self.metrics["rate_limit_events"] += 1

    def snapshot(self) -> dict[str, float | int]:
        now = int(time.time() * 1000)
        with self._connection() as connection:
            tokens, updated, cooldown = connection.execute(
                "SELECT tokens, updated_at, cooldown_until FROM binance_rate_limit_state WHERE id = 1"
            ).fetchone()
        available = min(CAPACITY, float(tokens) + max(0, now - int(updated)) / 1000 * REFILL_PER_SECOND)
        return {
            **self.metrics,
            "available_weight": available,
            "capacity_per_minute": CAPACITY,
            "critical_reserve": CRITICAL_RESERVE,
            "cooldown_until": int(cooldown),
        }

    def _try_acquire(self, weight: int, endpoint: str, priority: str) -> float:
        now = int(time.time() * 1000)
        wait = 0.0
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            tokens, updated, cooldown = connection.execute(
                "SELECT tokens, updated_at, cooldown_until FROM binance_rate_limit_state WHERE id = 1"
            ).fetchone()
            tokens = min(CAPACITY, float(tokens) + max(0, now - int(updated)) / 1000 * REFILL_PER_SECOND)
            if int(cooldown) > now:
                wait = (int(cooldown) - now) / 1000
            else:
                available = tokens if priority == "critical" else tokens - CRITICAL_RESERVE
                if available >= weight:
                    connection.execute(
                        "UPDATE binance_rate_limit_state SET tokens = ?, updated_at = ?, cooldown_until = 0 WHERE id = 1",
                        (tokens - weight, now),
                    )
                    connection.execute(
                        "INSERT INTO binance_rate_limit_events(created_at, process_name, endpoint, weight, priority, outcome) VALUES (?, ?, ?, ?, ?, ?)",
                        (now, self.process_name, endpoint, weight, priority, "granted"),
                    )
                    self.metrics["grants"] += 1
                    return 0.0
                wait = (weight - available) / REFILL_PER_SECOND
            connection.execute(
                "UPDATE binance_rate_limit_state SET tokens = ?, updated_at = ? WHERE id = 1",
                (tokens, now),
            )
        return max(0.025, wait)

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection
