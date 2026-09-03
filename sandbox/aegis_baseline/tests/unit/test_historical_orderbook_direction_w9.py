from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from aegis.research.historical_orderbook_direction_w9 import (
    L2Book,
    assess_coverage,
    audit_incremental_l2,
    audit_quotes,
    audit_trades,
    first_barrier_label,
    microprice,
    order_book_imbalance,
    stable_opportunity_episode_id,
)


def _row(*, snapshot: bool, side: str, price: float, amount: float) -> dict[str, str]:
    return {
        "exchange": "binance-futures",
        "symbol": "ADAUSDT",
        "timestamp": "100",
        "local_timestamp": "101",
        "is_snapshot": str(snapshot).lower(),
        "side": side,
        "price": str(price),
        "amount": str(amount),
    }


def test_book_requires_snapshot_and_applies_absolute_updates() -> None:
    book = L2Book()
    with pytest.raises(ValueError, match="UPDATE_BEFORE_SNAPSHOT"):
        book.apply_message([_row(snapshot=False, side="bid", price=99, amount=2)])
    book.apply_message([
        _row(snapshot=True, side="bid", price=99, amount=2),
        _row(snapshot=True, side="ask", price=101, amount=3),
    ])
    book.apply_message([_row(snapshot=False, side="bid", price=99, amount=4)])
    assert book.top() == (99, 4, 101, 3)
    with pytest.raises(ValueError, match="BOOK_SIDE_EMPTY"):
        book.apply_message([_row(snapshot=False, side="bid", price=99, amount=0)])


def test_book_rejects_crossed_state() -> None:
    book = L2Book()
    with pytest.raises(ValueError, match="BOOK_CROSSED"):
        book.apply_message([
            _row(snapshot=True, side="bid", price=101, amount=2),
            _row(snapshot=True, side="ask", price=100, amount=3),
        ])


def test_microprice_and_imbalance() -> None:
    assert order_book_imbalance(3, 1) == pytest.approx(0.5)
    assert microprice(99, 3, 101, 1) == pytest.approx(100.5)


def test_stable_episode_identity() -> None:
    first = stable_opportunity_episode_id("ADAUSDT", 123)
    assert first == stable_opportunity_episode_id("adausdt", 123)
    assert first != stable_opportunity_episode_id("ADAUSDT", 124)


def test_coverage_gate_fails_small_free_sample() -> None:
    result = assess_coverage(
        train_episodes=228,
        validation_episodes=231,
        train_symbols=11,
        validation_symbols=11,
        train_months=4,
        validation_months=4,
    )
    assert result["passes"] is False
    assert any("TRAIN_EPISODES" in item for item in result["blockers"])


def test_first_barrier_uses_path_not_terminal_close() -> None:
    path = [(1, 100.3), (2, 99.9)]
    assert first_barrier_label(path, reference_price=100, barrier_bps=25) == "UP_FIRST"


def test_streaming_audit_reconstructs_snapshot_and_update(tmp_path: Path) -> None:
    path = tmp_path / "l2.csv.gz"
    fields = ["exchange", "symbol", "timestamp", "local_timestamp", "is_snapshot", "side", "price", "amount"]
    rows = [
        _row(snapshot=True, side="bid", price=99, amount=2),
        _row(snapshot=True, side="ask", price=101, amount=3),
        {**_row(snapshot=False, side="bid", price=99, amount=4), "timestamp": "200", "local_timestamp": "201"},
    ]
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = audit_incremental_l2(path)
    assert result["passes_normalized_reconstruction"] is True
    assert result["messages"] == 2
    assert result["snapshot_messages"] == 1


def test_quote_and_trade_audits(tmp_path: Path) -> None:
    quote_path = tmp_path / "quotes.csv.gz"
    quote_fields = ["exchange", "symbol", "timestamp", "local_timestamp", "ask_amount", "ask_price", "bid_price", "bid_amount"]
    with gzip.open(quote_path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=quote_fields)
        writer.writeheader()
        writer.writerow({"exchange": "binance-futures", "symbol": "ADAUSDT", "timestamp": 100, "local_timestamp": 101, "ask_amount": 2, "ask_price": 101, "bid_price": 99, "bid_amount": 3})
    trade_path = tmp_path / "trades.csv.gz"
    trade_fields = ["exchange", "symbol", "timestamp", "local_timestamp", "id", "side", "price", "amount"]
    with gzip.open(trade_path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trade_fields)
        writer.writeheader()
        writer.writerow({"exchange": "binance-futures", "symbol": "ADAUSDT", "timestamp": 100, "local_timestamp": 101, "id": 1, "side": "buy", "price": 100, "amount": 2})
    assert audit_quotes(quote_path)["passes"] is True
    assert audit_trades(trade_path)["passes"] is True
