#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_canary_exec as ex


def test_private_adapter_reports_missing_runtime_credentials_without_env_file() -> None:
    old = {k: os.environ.pop(k, None) for k in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "BINANCE_FUTURES_API_KEY", "BINANCE_FUTURES_API_SECRET")}
    try:
        adapter = ex.BinancePrivateReadOnlyAdapter()
        assert adapter.available is False
        assert adapter.private_snapshot()["reason"] == "PRIVATE_CREDENTIALS_NOT_AVAILABLE"
    finally:
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v


def test_secret_redaction() -> None:
    os.environ["BINANCE_API_KEY"] = "abc-secret-key"
    os.environ["BINANCE_API_SECRET"] = "def-secret-value"
    try:
        text = ex.redact_secret_text("abc-secret-key and def-secret-value")
        assert "abc-secret-key" not in text
        assert "def-secret-value" not in text
        assert text.count("<REDACTED>") == 2
    finally:
        os.environ.pop("BINANCE_API_KEY", None)
        os.environ.pop("BINANCE_API_SECRET", None)


def test_adapter_has_no_order_surface_at_all() -> None:
    adapter = ex.BinancePrivateReadOnlyAdapter()
    forbidden = [name for name in dir(adapter) if any(w in name.lower() for w in ("order", "entry", "submit", "place", "close_position", "cancel")) and callable(getattr(adapter, name)) and not name.startswith("_") and name != "open_orders"]
    assert forbidden == [], f"private adapter must be read-only + kill only, found: {forbidden}"
    assert ex.PYTHON_SUBMITS_ORDERS is False


if __name__ == "__main__":
    test_private_adapter_reports_missing_runtime_credentials_without_env_file()
    test_secret_redaction()
    test_adapter_has_no_order_surface_at_all()
    print("test_gen2_canary_private_adapter: OK")
