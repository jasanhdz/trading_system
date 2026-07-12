#!/usr/bin/env python3
"""Gen2 live-canary execution adapter and safety dry-run.

The module is deliberately fail-closed. It never submits exchange orders unless
an injected adapter implements that behavior and all gates pass. The CLI dry-run
uses local/read-only state plus optional public exchange metadata and reports
`orders_submitted=0`.
"""
from __future__ import annotations

import argparse
import hmac
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aegis_alpha.tools.gen2_canary_core as core  # noqa: E402
from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import GEN2_ROOT, sha256_file, utc_now, validate_gen2_path  # noqa: E402

DEFAULT_CANDIDATE_ID = "gen2-20260711T202935Z"
DEFAULT_ALLOWED_SYMBOLS = ("ADAUSDT", "DOGEUSDT")
MARGIN_ALLOCATION_FRACTION = 0.50
ALLOWED_LEVERAGE = (15, 20)
DEFAULT_LEVERAGE = 20
EQUITY_STOP_FRACTION = 0.50
MAX_CONCURRENT_POSITIONS = 1
FIRST_ARM_MAX_ORDERS = 1
MINIMUM_LIQUIDATION_BUFFER_PCT = 0.010
DEFAULT_STOP_DISTANCE_PCT = 0.015
FEE_RATE_ROUND_TRIP = 0.0016
REAL_ORDER_SUBMISSION_ENABLED = False
BINANCE_FAPI = "https://fapi.binance.com"


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=json_default)


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    validate_gen2_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(_json_dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    validate_gen2_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=json_default) + "\n")


def load_freeze(path: Path = core.FREEZE_PATH) -> dict[str, Any]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("candidate_id") != DEFAULT_CANDIDATE_ID:
        raise ValueError(f"unexpected candidate_id {freeze.get('candidate_id')}")
    return freeze


def phase_o_new_entries_paused(ts_repo: Path | None = None) -> tuple[bool, str]:
    ts_repo = ts_repo or (REPO / "binance-futures-bot-ts")
    yaml_path = ts_repo / "regime_config.live.yaml"
    text = yaml_path.read_text(encoding="utf-8")
    m = re.search(r"phase_o_short_live:\n(?P<body>(?:[ \t]+[^\n]*\n)+)", text)
    if not m:
        return False, "PHASE_O_CONFIG_NOT_FOUND"
    body = m.group("body")
    if re.search(r"^\s*enabled:\s*true\s*$", body, re.MULTILINE) and re.search(r"^\s*allow_orders:\s*false\s*$", body, re.MULTILINE):
        return True, "PHASE_O_NEW_ENTRIES_PAUSED"
    return False, "PHASE_O_NEW_ENTRIES_NOT_PAUSED"


def deterministic_client_order_id(candidate_id: str, signal_id: str, symbol: str, side: str) -> str:
    seed = f"{candidate_id}|{signal_id}|{symbol.upper()}|{side.upper()}".encode("utf-8")
    return "GEN2-" + hashlib.sha256(seed).hexdigest()[:28]


def round_down_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step


@dataclass(frozen=True)
class SymbolFilters:
    symbol: str
    min_notional: float
    step_size: float
    tick_size: float
    min_qty: float = 0.0
    max_leverage: int = DEFAULT_LEVERAGE


@dataclass(frozen=True)
class SizingDecision:
    symbol: str
    available_balance: float
    allocated_margin: float
    leverage: int
    price: float
    quantity: float
    target_notional: float
    actual_notional: float
    required_isolated_margin: float
    estimated_fees: float
    stop_price: float
    liquidation_price: float
    liquidation_buffer_pct: float
    worst_case_loss: float
    equity_after_worst_case: float
    decision: str
    reason: str


def estimate_short_liquidation_price(entry_price: float, leverage: int, maintenance_margin_rate: float = 0.004) -> float:
    return entry_price * (1.0 + (1.0 / leverage) - maintenance_margin_rate)


def stop_price_for_short(entry_price: float, stop_distance_pct: float = DEFAULT_STOP_DISTANCE_PCT) -> float:
    return entry_price * (1.0 + stop_distance_pct)


def liquidation_buffer_pct(stop_price: float, liquidation_price: float) -> float:
    if stop_price <= 0:
        return 0.0
    return (liquidation_price - stop_price) / stop_price


def compute_sizing_v2(symbol: str, filters: SymbolFilters, price: float, available_balance: float,
                      leverage: int = DEFAULT_LEVERAGE, current_equity: float | None = None,
                      minimum_liquidation_buffer_pct: float = MINIMUM_LIQUIDATION_BUFFER_PCT) -> SizingDecision:
    if leverage not in ALLOWED_LEVERAGE:
        return SizingDecision(symbol, available_balance, 0.0, leverage, price, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "NO_TRADE", "LEVERAGE_NOT_ALLOWED")
    if leverage > filters.max_leverage:
        return SizingDecision(symbol, available_balance, 0.0, leverage, price, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "NO_TRADE", "LEVERAGE_NOT_SUPPORTED_BY_SYMBOL")
    allocated_margin = available_balance * MARGIN_ALLOCATION_FRACTION
    target_notional = allocated_margin * leverage
    quantity = round_down_step(target_notional / price, filters.step_size)
    quantity = round(quantity, 12)
    actual_notional = quantity * price
    required_margin = actual_notional / leverage if leverage > 0 else float("inf")
    fees = actual_notional * FEE_RATE_ROUND_TRIP
    while quantity > 0 and required_margin + fees > allocated_margin + 1e-9:
        quantity = round(quantity - filters.step_size, 12)
        actual_notional = quantity * price
        required_margin = actual_notional / leverage if leverage > 0 else float("inf")
        fees = actual_notional * FEE_RATE_ROUND_TRIP
    stop_price = stop_price_for_short(price)
    liquidation_price = estimate_short_liquidation_price(price, leverage)
    buffer_pct = liquidation_buffer_pct(stop_price, liquidation_price)
    worst_loss = actual_notional * DEFAULT_STOP_DISTANCE_PCT + fees
    equity = available_balance if current_equity is None else current_equity
    equity_after = equity - worst_loss
    if quantity <= 0 or quantity < filters.min_qty:
        reason = "MIN_QTY_NOT_MET"
    elif actual_notional > target_notional + 1e-9:
        reason = "NOTIONAL_EXCEEDS_TARGET"
    elif actual_notional < filters.min_notional:
        reason = "MIN_NOTIONAL_NOT_MET"
    elif required_margin > allocated_margin + 1e-9:
        reason = "ALLOCATED_MARGIN_EXCEEDED"
    elif allocated_margin + 1e-9 < required_margin + fees:
        reason = "FEES_EXCEED_MARGIN_REMAINDER"
    elif not (price < stop_price < liquidation_price):
        reason = "STOP_LIQUIDATION_ORDER_INVALID"
    elif buffer_pct < minimum_liquidation_buffer_pct:
        reason = "LIQUIDATION_BUFFER_INSUFFICIENT"
    else:
        reason = "CAPITAL_OK"
    decision = "CANARY_CAPITAL_SUFFICIENT" if reason == "CAPITAL_OK" else "NO_TRADE"
    return SizingDecision(symbol, available_balance, allocated_margin, leverage, price, quantity, target_notional,
                          actual_notional, required_margin, fees, stop_price, liquidation_price, buffer_pct,
                          worst_loss, equity_after, decision, reason)


def select_sizing_v2(symbol: str, filters: SymbolFilters, price: float, available_balance: float,
                     current_equity: float | None = None) -> SizingDecision:
    first = compute_sizing_v2(symbol, filters, price, available_balance, DEFAULT_LEVERAGE, current_equity)
    if first.decision == "CANARY_CAPITAL_SUFFICIENT":
        return first
    if first.reason == "LIQUIDATION_BUFFER_INSUFFICIENT":
        second = compute_sizing_v2(symbol, filters, price, available_balance, 15, current_equity)
        if second.decision == "CANARY_CAPITAL_SUFFICIENT":
            return second
    return first


def capital_feasibility(symbol: str, filters: SymbolFilters, price: float, capital_cap: float | None = None,
                        leverage: int = DEFAULT_LEVERAGE, **_: Any) -> SizingDecision:
    return compute_sizing_v2(symbol, filters, price, capital_cap if capital_cap is not None else 16.24, leverage)


def load_public_exchange_info(symbols: tuple[str, ...] = DEFAULT_ALLOWED_SYMBOLS, timeout: float = 8.0) -> dict[str, SymbolFilters]:
    url = f"{BINANCE_FAPI}/fapi/v1/exchangeInfo"
    with urlopen(url, timeout=timeout) as resp:  # nosec - public market metadata only
        payload = json.loads(resp.read().decode("utf-8"))
    out: dict[str, SymbolFilters] = {}
    for row in payload.get("symbols", []):
        symbol = row.get("symbol")
        if symbol not in symbols:
            continue
        filters = {item.get("filterType"): item for item in row.get("filters", [])}
        lot = filters.get("LOT_SIZE", {})
        notional = filters.get("MIN_NOTIONAL", {})
        price = filters.get("PRICE_FILTER", {})
        out[symbol] = SymbolFilters(
            symbol=symbol,
            min_notional=float(notional.get("notional", notional.get("minNotional", 5.0))),
            step_size=float(lot.get("stepSize", 1.0)),
            tick_size=float(price.get("tickSize", 0.0001)),
            min_qty=float(lot.get("minQty", 0.0)),
            max_leverage=DEFAULT_LEVERAGE,
        )
    return out


def load_public_prices(symbols: tuple[str, ...] = DEFAULT_ALLOWED_SYMBOLS, timeout: float = 8.0) -> dict[str, float]:
    prices = {}
    for symbol in symbols:
        url = f"{BINANCE_FAPI}/fapi/v1/ticker/price?{urlencode({'symbol': symbol})}"
        with urlopen(url, timeout=timeout) as resp:  # nosec - public market metadata only
            payload = json.loads(resp.read().decode("utf-8"))
        prices[symbol] = float(payload["price"])
    return prices


def load_account_snapshot_from_logs(ts_repo: Path | None = None) -> dict[str, Any]:
    ts_repo = ts_repo or (REPO / "binance-futures-bot-ts")
    logs = ts_repo / "logs" / "aegis"
    latest = sorted(logs.glob("account_snapshots_*.jsonl"))
    if not latest:
        return {"available_balance": None, "source": "NO_ACCOUNT_SNAPSHOT"}
    lines = latest[-1].read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return {"available_balance": None, "source": str(latest[-1]), "empty": True}
    row = json.loads(lines[-1])
    return {
        "available_balance": row.get("availableBalance", row.get("available_balance")),
        "wallet_balance": row.get("walletBalance", row.get("wallet_balance")),
        "account_equity": row.get("equityTotal", row.get("equity_total", row.get("walletBalance", row.get("wallet_balance")))),
        "open_position": bool(row.get("positionOpen", row.get("position_open", False))),
        "source": str(latest[-1]),
        "timestamp": row.get("timestamp", row.get("recorded_at")),
    }


def redact_secret_text(text: str) -> str:
    for name in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "BINANCE_FUTURES_API_KEY", "BINANCE_FUTURES_API_SECRET"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "<REDACTED>")
    return text


class BinancePrivateReadOnlyAdapter:
    """USD-M Futures private read adapter.

    It signs read/account requests if credentials are present in environment.
    Order methods are present for interface completeness but hard-blocked while
    REAL_ORDER_SUBMISSION_ENABLED is false.
    """

    def __init__(self, timeout: float = 8.0) -> None:
        self.api_key = os.environ.get("BINANCE_FUTURES_API_KEY") or os.environ.get("BINANCE_API_KEY")
        self.api_secret = os.environ.get("BINANCE_FUTURES_API_SECRET") or os.environ.get("BINANCE_API_SECRET")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def _signed_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.available:
            raise RuntimeError("PRIVATE_CREDENTIALS_NOT_AVAILABLE")
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        signature = hmac.new(str(self.api_secret).encode(), query.encode(), hashlib.sha256).hexdigest()
        url = f"{BINANCE_FAPI}{path}?{query}&signature={signature}"
        req = Request(url, headers={"X-MBX-APIKEY": str(self.api_key)})
        try:
            with urlopen(req, timeout=self.timeout) as resp:  # nosec - signed read-only endpoint
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(redact_secret_text(repr(exc))) from None

    def account(self) -> dict[str, Any]:
        return self._signed_get("/fapi/v2/account")

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {"symbol": symbol} if symbol else {}
        return self._signed_get("/fapi/v1/openOrders", params)

    def position_risk(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {"symbol": symbol} if symbol else {}
        payload = self._signed_get("/fapi/v2/positionRisk", params)
        return payload if isinstance(payload, list) else [payload]

    def leverage_bracket(self, symbol: str) -> Any:
        return self._signed_get("/fapi/v1/leverageBracket", {"symbol": symbol})

    def place_entry(self, *_: Any, **__: Any) -> None:
        if not REAL_ORDER_SUBMISSION_ENABLED:
            raise RuntimeError("REAL_ORDER_SUBMISSION_DISABLED")
        raise RuntimeError("ORDER_SUBMISSION_NOT_IMPLEMENTED_IN_DRY_RUN")


def init_operational_manifest_v2(candidate_id: str, account: dict[str, Any]) -> dict[str, Any]:
    cdir = core.canary_dir(candidate_id)
    path = cdir / "operational_manifest_v2.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    initial_equity = float(account.get("account_equity") or account.get("wallet_balance") or account.get("available_balance"))
    manifest = {
        "schema": "gen2_canary_operational_manifest_v2",
        "operational_revision": "GEN2_F1_CANARY_CAPITAL_V2",
        "candidate_id": candidate_id,
        "created_at_utc": utc_now().isoformat(),
        "changed_component": "OPERATIONAL_CAPITAL_CONTRACT",
        "unchanged_components": ["MODEL_AND_POLICY_FREEZE"],
        "initial_canary_equity": initial_equity,
        "equity_floor": initial_equity * EQUITY_STOP_FRACTION,
        "margin_allocation_fraction": MARGIN_ALLOCATION_FRACTION,
        "allowed_leverage": list(ALLOWED_LEVERAGE),
        "default_leverage": DEFAULT_LEVERAGE,
        "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
        "first_arm_max_orders": FIRST_ARM_MAX_ORDERS,
        "equity_stop_fraction": EQUITY_STOP_FRACTION,
        "martingale": False,
        "averaging_down": False,
        "pyramiding": False,
        "automatic_compounding": True,
        "real_order_submission_enabled": REAL_ORDER_SUBMISSION_ENABLED,
    }
    atomic_write(path, manifest)
    return manifest


def check_equity_floor(candidate_id: str, current_equity: float) -> tuple[bool, str, dict[str, Any]]:
    manifest = json.loads((core.canary_dir(candidate_id) / "operational_manifest_v2.json").read_text(encoding="utf-8"))
    floor = float(manifest["equity_floor"])
    if current_equity <= floor:
        core.engage_kill_switch(candidate_id, "CANARY_EQUITY_FLOOR_REACHED")
        return False, "CANARY_EQUITY_FLOOR_REACHED", {"current_equity": current_equity, "equity_floor": floor}
    return True, "EQUITY_FLOOR_OK", {"current_equity": current_equity, "equity_floor": floor}


def create_arm_token_v2(candidate_id: str, expiry_hours: int, initial_equity: float,
                        allowed_symbols: tuple[str, ...] = DEFAULT_ALLOWED_SYMBOLS) -> dict[str, Any]:
    body = {
        "schema": "gen2_canary_arm_token_v2",
        "candidate_id": candidate_id,
        "operational_revision": "GEN2_F1_CANARY_CAPITAL_V2",
        "initial_equity": initial_equity,
        "margin_allocation_fraction": MARGIN_ALLOCATION_FRACTION,
        "allowed_leverage": list(ALLOWED_LEVERAGE),
        "default_leverage": DEFAULT_LEVERAGE,
        "allowed_symbols": sorted(allowed_symbols),
        "max_orders": FIRST_ARM_MAX_ORDERS,
        "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
        "expires_at": (utc_now().timestamp() + expiry_hours * 3600),
        "equity_stop_fraction": EQUITY_STOP_FRACTION,
        "consumed": False,
    }
    signed = {k: body[k] for k in sorted(body) if k != "checksum"}
    body["checksum"] = hashlib.sha256(json.dumps(signed, sort_keys=True).encode()).hexdigest()
    return body


def verify_arm_token_v2(candidate_id: str) -> tuple[bool, str, dict[str, Any]]:
    path = core.canary_dir(candidate_id) / "ARM_TOKEN_V2.json"
    if not path.exists():
        return False, "CANARY_UNARMED", {}
    token = json.loads(path.read_text(encoding="utf-8"))
    if token.get("schema") != "gen2_canary_arm_token_v2":
        return False, "TOKEN_V1_INCOMPATIBLE_WITH_SPEC_V2", {}
    checksum = token.get("checksum")
    signed = {k: token[k] for k in sorted(token) if k != "checksum"}
    expected = hashlib.sha256(json.dumps(signed, sort_keys=True).encode()).hexdigest()
    if checksum != expected:
        return False, "TOKEN_CHECKSUM_INVALID", {}
    if token.get("candidate_id") != candidate_id:
        return False, "TOKEN_WRONG_CANDIDATE", {}
    if token.get("operational_revision") != "GEN2_F1_CANARY_CAPITAL_V2":
        return False, "TOKEN_WRONG_OPERATIONAL_REVISION", {}
    if token.get("margin_allocation_fraction") != MARGIN_ALLOCATION_FRACTION:
        return False, "TOKEN_MARGIN_FRACTION_MISMATCH", {}
    if token.get("allowed_leverage") != list(ALLOWED_LEVERAGE):
        return False, "TOKEN_LEVERAGE_MISMATCH", {}
    if token.get("max_orders") != FIRST_ARM_MAX_ORDERS:
        return False, "TOKEN_MAX_ORDERS_MISMATCH", {}
    if token.get("consumed") is True:
        return False, "TOKEN_CONSUMED", {}
    if float(token.get("expires_at", 0)) < utc_now().timestamp():
        return False, "TOKEN_EXPIRED", {}
    return True, "TOKEN_V2_VALID", token


class CanaryExecutionAdapter:
    def __init__(self, candidate_id: str = DEFAULT_CANDIDATE_ID, candidate_dir: Path | None = None,
                 allowed_symbols: tuple[str, ...] = DEFAULT_ALLOWED_SYMBOLS) -> None:
        self.candidate_id = candidate_id
        self.candidate_dir = candidate_dir or core.canary_dir(candidate_id)
        self.allowed_symbols = tuple(s.upper() for s in allowed_symbols)
        self.orders_submitted = 0
        self.state_path = self.candidate_dir / "execution_state.json"
        if not self.state_path.exists():
            atomic_write(self.state_path, {"processed_signal_ids": [], "open_client_order_ids": [], "orders_submitted": 0})

    def _state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save_state(self, state: dict[str, Any]) -> None:
        atomic_write(self.state_path, state)

    def validate(self, opportunity: dict[str, Any], filters: SymbolFilters, price: float, available_balance: float | None,
                 current_equity: float | None = None,
                 now: datetime | None = None) -> tuple[bool, str, dict[str, Any]]:
        now = now or utc_now()
        freeze = load_freeze()
        if opportunity.get("candidate_id") != self.candidate_id or freeze.get("candidate_id") != self.candidate_id:
            return False, "WRONG_CANDIDATE", {}
        symbol = str(opportunity.get("symbol", "")).upper()
        if symbol not in self.allowed_symbols:
            return False, "SYMBOL_NOT_ALLOWED", {}
        if str(opportunity.get("side", "")).upper() != "SHORT":
            return False, "ONLY_SHORT_ALLOWED", {}
        if int(opportunity.get("primary_horizon", 0)) != 12:
            return False, "PRIMARY_H12_REQUIRED", {}
        if opportunity.get("final_candle") is not True:
            return False, "PARTIAL_CANDLE", {}
        signal_id = str(opportunity.get("signal_id", ""))
        state = self._state()
        if signal_id in state.get("processed_signal_ids", []):
            return False, "DUPLICATE_SIGNAL", {}
        if deterministic_client_order_id(self.candidate_id, signal_id, symbol, "SHORT") in state.get("open_client_order_ids", []):
            return False, "DUPLICATE_CLIENT_ORDER_ID", {}
        if state.get("position_open") is True or int(opportunity.get("open_positions", 0)) >= MAX_CONCURRENT_POSITIONS:
            return False, "MAX_CONCURRENT_POSITIONS_REACHED", {}
        phase_paused, phase_reason = phase_o_new_entries_paused()
        if not phase_paused:
            return False, phase_reason, {}
        risk_ok, risk_reason = core.risk_gate(self.candidate_id)
        if not risk_ok:
            return False, risk_reason, {}
        token_ok, token_reason, _token = verify_arm_token_v2(self.candidate_id)
        if not token_ok:
            return False, token_reason, {}
        leverage = int(opportunity.get("leverage", DEFAULT_LEVERAGE))
        if leverage not in ALLOWED_LEVERAGE:
            return False, "LEVERAGE_NOT_ALLOWED", {}
        if str(opportunity.get("margin_type", "ISOLATED")).upper() != "ISOLATED":
            return False, "ISOLATED_MARGIN_NOT_CONFIRMED", {}
        if available_balance is None or available_balance <= 0:
            return False, "BALANCE_UNKNOWN", {}
        equity = current_equity if current_equity is not None else available_balance
        if not (core.canary_dir(self.candidate_id) / "operational_manifest_v2.json").exists():
            init_operational_manifest_v2(self.candidate_id, {"available_balance": available_balance, "account_equity": equity})
        floor_ok, floor_reason, floor_meta = check_equity_floor(self.candidate_id, equity)
        if not floor_ok:
            return False, floor_reason, floor_meta
        sizing = compute_sizing_v2(symbol, filters, price, available_balance, leverage, equity)
        if sizing.decision != "CANARY_CAPITAL_SUFFICIENT":
            return False, sizing.reason, sizing.__dict__
        client_order_id = deterministic_client_order_id(self.candidate_id, signal_id, symbol, "SHORT")
        return True, "READY", {**sizing.__dict__, "client_order_id": client_order_id}

    def submit(self, opportunity: dict[str, Any], filters: SymbolFilters, price: float, available_balance: float | None,
               current_equity: float | None = None, dry_run: bool = True) -> dict[str, Any]:
        ok, reason, sizing = self.validate(opportunity, filters, price, available_balance, current_equity)
        record = {
            "schema": "gen2_live_order_attempt_v1",
            "candidate_id": self.candidate_id,
            "signal_id": opportunity.get("signal_id"),
            "symbol": opportunity.get("symbol"),
            "side": opportunity.get("side"),
            "accepted": ok,
            "reason": reason,
            "dry_run": dry_run,
            "order_action": "NO_ORDER" if dry_run or not ok else "SUBMIT_ORDER",
            "enforcement_action": "NONE" if dry_run or not ok else "LIVE_ORDER",
            "real_order_submission_enabled": REAL_ORDER_SUBMISSION_ENABLED,
            "sizing": sizing,
            "recorded_at_utc": utc_now().isoformat(),
        }
        if dry_run or not ok:
            append_jsonl(self.candidate_dir / "live_orders.jsonl", record)
            return record
        raise RuntimeError("LIVE_ORDER_SUBMISSION_DISABLED_IN_CODEX_TASK")


class BracketManager:
    def __init__(self, candidate_id: str = DEFAULT_CANDIDATE_ID, candidate_dir: Path | None = None) -> None:
        self.candidate_id = candidate_id
        self.candidate_dir = candidate_dir or core.canary_dir(candidate_id)

    def confirm(self, fill: dict[str, Any], stop_confirmed: bool, time_exit_confirmed: bool, elapsed_seconds: float) -> dict[str, Any]:
        ok = bool(fill.get("filled_qty", 0) > 0 and stop_confirmed and time_exit_confirmed and elapsed_seconds <= 60)
        reason = "BRACKETS_CONFIRMED" if ok else "BRACKET_CONFIRMATION_FAILED"
        record = {
            "schema": "gen2_bracket_confirmation_v1",
            "candidate_id": self.candidate_id,
            "client_order_id": fill.get("client_order_id"),
            "reduce_only": True,
            "stop_confirmed": stop_confirmed,
            "time_exit_confirmed": time_exit_confirmed,
            "elapsed_seconds": elapsed_seconds,
            "ok": ok,
            "reason": reason,
            "recorded_at_utc": utc_now().isoformat(),
        }
        append_jsonl(self.candidate_dir / "brackets.jsonl", record)
        if not ok and fill.get("filled_qty", 0) > 0:
            core.engage_kill_switch(self.candidate_id, "CRITICAL_EXECUTION_FAILURE_BRACKET")
        return record


class Reconciler:
    def __init__(self, candidate_id: str = DEFAULT_CANDIDATE_ID, candidate_dir: Path | None = None) -> None:
        self.candidate_id = candidate_id
        self.candidate_dir = candidate_dir or core.canary_dir(candidate_id)

    def reconcile(self, local: dict[str, Any], exchange: dict[str, Any]) -> dict[str, Any]:
        incidents: list[str] = []
        if exchange.get("orphan_position"):
            incidents.append("ORPHAN_POSITION")
        if exchange.get("duplicate_order"):
            incidents.append("DUPLICATE_ORDER")
        if exchange.get("missing_bracket"):
            incidents.append("MISSING_BRACKET")
        if exchange.get("local_exchange_mismatch"):
            incidents.append("LOCAL_EXCHANGE_MISMATCH")
        status = "RECONCILED" if not incidents else "RECONCILIATION_FAIL_CLOSED"
        record = {"schema": "gen2_reconciliation_v1", "candidate_id": self.candidate_id, "status": status,
                  "incidents": incidents, "local": local, "exchange": exchange, "recorded_at_utc": utc_now().isoformat()}
        append_jsonl(self.candidate_dir / "reconciliations.jsonl", record)
        if incidents and exchange.get("exposure", 0):
            core.engage_kill_switch(self.candidate_id, "RECONCILIATION_UNCERTAIN_WITH_EXPOSURE")
        return record


def dry_run(candidate_id: str, use_public: bool = False) -> dict[str, Any]:
    core.init_canary(candidate_id)
    cdir = core.canary_dir(candidate_id)
    filters = {
        "ADAUSDT": SymbolFilters("ADAUSDT", min_notional=5.0, step_size=1.0, tick_size=0.0001),
        "DOGEUSDT": SymbolFilters("DOGEUSDT", min_notional=5.0, step_size=1.0, tick_size=0.00001),
    }
    prices = {"ADAUSDT": 0.70, "DOGEUSDT": 0.12}
    public_error = None
    if use_public:
        try:
            filters.update(load_public_exchange_info())
            prices.update(load_public_prices())
        except Exception as exc:  # public connectivity is diagnostic only
            public_error = repr(exc)
    account = load_account_snapshot_from_logs()
    available = account.get("available_balance")
    if available is None:
        available = 16.24
    adapter = CanaryExecutionAdapter(candidate_id)
    opportunities = [
        {"candidate_id": candidate_id, "signal_id": "eligible-1", "symbol": "ADAUSDT", "side": "SHORT", "primary_horizon": 12, "final_candle": True, "leverage": 5},
        {"candidate_id": candidate_id, "signal_id": "not-eligible-1", "symbol": "BTCUSDT", "side": "SHORT", "primary_horizon": 12, "final_candle": True, "leverage": 5},
        {"candidate_id": candidate_id, "signal_id": "partial-1", "symbol": "DOGEUSDT", "side": "SHORT", "primary_horizon": 12, "final_candle": False, "leverage": 5},
    ]
    attempts = []
    for opp in opportunities:
        symbol = opp["symbol"]
        attempts.append(adapter.submit(opp, filters.get(symbol, filters["ADAUSDT"]), prices.get(symbol, prices["ADAUSDT"]), available, dry_run=True))
    brackets = BracketManager(candidate_id).confirm({"client_order_id": "mock-fill", "filled_qty": 1.0}, True, True, 3.0)
    bracket_fail = BracketManager(candidate_id).confirm({"client_order_id": "mock-fill-fail", "filled_qty": 0.0}, False, False, 61.0)
    reconciliation = Reconciler(candidate_id).reconcile({"open_orders": []}, {"open_orders": [], "exposure": 0})
    feasibility = [_json_dumps(capital_feasibility(s, filters[s], prices[s]).__dict__) for s in DEFAULT_ALLOWED_SYMBOLS]
    report = {
        "candidate_id": candidate_id,
        "armed": core.verify_arm_token(candidate_id)[0],
        "phase_o": phase_o_new_entries_paused()[1],
        "orders_submitted": adapter.orders_submitted,
        "attempts": attempts,
        "bracket_success": brackets,
        "bracket_failure_mock": bracket_fail,
        "reconciliation": reconciliation,
        "capital_feasibility": [json.loads(x) for x in feasibility],
        "account_snapshot": account,
        "public_market_data_error": public_error,
        "FORWARD_OUTCOMES_NOT_EVALUATED": True,
    }
    atomic_write(cdir / "dry_run_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gen2 canary execution safety adapter")
    parser.add_argument("--mode", choices=["dry-run", "capital-audit"], default="dry-run")
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--public-market-data", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "capital-audit":
        payload = dry_run(args.candidate_id, use_public=args.public_market_data)["capital_feasibility"]
    else:
        payload = dry_run(args.candidate_id, use_public=args.public_market_data)
    print(json.dumps(payload, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
