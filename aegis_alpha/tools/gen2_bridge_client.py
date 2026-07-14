#!/usr/bin/env python3
"""GEN2 signed HTTP client for the TypeScript execution bridge.

Aegis is the brain; TS is the execution engine. This client signs every request
with HMAC-SHA256 over `<timestamp>.<body>` using GEN2_BRIDGE_SECRET (env only,
never persisted), retries safely thanks to deterministic clientOrderIds
(DUPLICATE acks are success), and ingests execution events into the canary
streams. Python never talks to Binance to execute.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = os.environ.get("GEN2_BRIDGE_URL", "http://127.0.0.1:8791")
SECRET_ENV = "GEN2_BRIDGE_SECRET"
MAX_CLOCK_SKEW_SECONDS = 5


class BridgeUnavailable(RuntimeError):
    pass


def _secret() -> bytes:
    secret = os.environ.get(SECRET_ENV, "")
    if not secret:
        raise BridgeUnavailable("GEN2_BRIDGE_SECRET_NOT_SET")
    return secret.encode()


def sign(body: str, timestamp: str, secret: bytes | None = None) -> str:
    return hmac.new(secret or _secret(), f"{timestamp}.{body}".encode(), hashlib.sha256).hexdigest()


def _request(method: str, path: str, body: dict[str, Any] | None, base_url: str, timeout: float) -> dict[str, Any]:
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")) if body is not None else ""
    timestamp = str(int(time.time() * 1000))
    headers = {
        "Content-Type": "application/json",
        "X-GEN2-TIMESTAMP": timestamp,
        "X-GEN2-SIGNATURE": sign(payload, timestamp),
    }
    req = urllib.request.Request(f"{base_url}{path}", data=payload.encode() if payload else None, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read().decode())
        except Exception:
            raise BridgeUnavailable(f"HTTP_{exc.code}") from None
    except Exception as exc:
        raise BridgeUnavailable(repr(exc)) from None


def get_status(base_url: str = DEFAULT_BASE_URL, timeout: float = 5.0) -> dict[str, Any]:
    return _request("GET", "/gen2/status", None, base_url, timeout)


def get_events(after_sequence: int, base_url: str = DEFAULT_BASE_URL, timeout: float = 10.0) -> list[dict[str, Any]]:
    page = _request("POST", "/gen2/events", {"after_sequence": int(after_sequence)}, base_url, timeout)
    events = page.get("events")
    if not isinstance(events, list):
        raise BridgeUnavailable("EVENTS_PAGE_MALFORMED")
    return events


def post_execute(order: dict[str, Any], base_url: str = DEFAULT_BASE_URL, timeout: float = 10.0, retries: int = 2) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            ack = _request("POST", "/gen2/execute", order, base_url, timeout)
            if ack.get("status") in {"ACCEPTED", "ACCEPTED_DRYRUN", "DUPLICATE", "REJECTED"}:
                return ack
            last = ack
        except BridgeUnavailable as exc:
            last = {"status": "BRIDGE_UNAVAILABLE", "reason": str(exc)}
        if attempt < retries:
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
    return last or {"status": "BRIDGE_UNAVAILABLE", "reason": "NO_RESPONSE"}


def post_kill(candidate_id: str, reason: str, action: str = "BLOCK_NEW_ONLY", base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    return _request("POST", "/gen2/kill", {"schema": "gen2_kill_v1", "candidate_id": candidate_id, "reason": reason, "action": action}, base_url, 10.0)
