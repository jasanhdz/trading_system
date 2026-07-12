#!/usr/bin/env python3
"""Gen2 live-canary execution adapter and safety dry-run.

The module is deliberately fail-closed. It never submits exchange orders unless
an injected adapter implements that behavior and all gates pass. The CLI dry-run
uses local/read-only state plus optional public exchange metadata and reports
`orders_submitted=0`.
"""
from __future__ import annotations

import argparse
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
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aegis_alpha.tools.gen2_canary_core as core  # noqa: E402
from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import GEN2_ROOT, sha256_file, utc_now, validate_gen2_path  # noqa: E402

DEFAULT_CANDIDATE_ID = "gen2-20260711T202935Z"
DEFAULT_ALLOWED_SYMBOLS = ("ADAUSDT", "DOGEUSDT")
DEFAULT_CAPITAL_CAP = 15.0
DEFAULT_LEVERAGE = 5
DEFAULT_RISK_PER_TRADE_PCT = 0.005
DEFAULT_STOP_DISTANCE_PCT = 0.015
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
    max_leverage: int = DEFAULT_LEVERAGE


@dataclass(frozen=True)
class CapitalFeasibility:
    symbol: str
    price: float
    min_executable_quantity: float
    min_executable_notional: float
    required_isolated_margin: float
    expected_fees: float
    worst_case_loss: float
    capital_for_0_5pct_risk: float
    decision: str


def capital_feasibility(symbol: str, filters: SymbolFilters, price: float, capital_cap: float = DEFAULT_CAPITAL_CAP,
                        leverage: int = DEFAULT_LEVERAGE, fee_rate: float = 0.0008,
                        stop_distance_pct: float = DEFAULT_STOP_DISTANCE_PCT, slippage_pct: float = 0.001) -> CapitalFeasibility:
    if leverage <= 0 or leverage > filters.max_leverage:
        raise ValueError("invalid leverage")
    qty = max(filters.step_size, round_down_step(filters.min_notional / price, filters.step_size))
    while qty * price < filters.min_notional:
        qty += filters.step_size
    qty = round(qty, 12)
    notional = qty * price
    margin = notional / leverage
    fees = notional * fee_rate * 2
    worst_loss = notional * (stop_distance_pct + slippage_pct) + fees
    capital_for_risk = worst_loss / DEFAULT_RISK_PER_TRADE_PCT
    decision = "CANARY_CAPITAL_SUFFICIENT" if margin <= capital_cap and capital_for_risk <= capital_cap else "CANARY_CAPITAL_INSUFFICIENT"
    return CapitalFeasibility(symbol, price, qty, notional, margin, fees, worst_loss, capital_for_risk, decision)


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
        "open_position": bool(row.get("positionOpen", row.get("position_open", False))),
        "source": str(latest[-1]),
        "timestamp": row.get("timestamp", row.get("recorded_at")),
    }


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
        phase_paused, phase_reason = phase_o_new_entries_paused()
        if not phase_paused:
            return False, phase_reason, {}
        risk_ok, risk_reason = core.risk_gate(self.candidate_id)
        if not risk_ok:
            return False, risk_reason, {}
        token_ok, token_reason = core.verify_arm_token(self.candidate_id)
        if not token_ok:
            return False, "CANARY_UNARMED" if token_reason == "CANARY_UNARMED_NO_TOKEN" else token_reason, {}
        leverage = int(opportunity.get("leverage", DEFAULT_LEVERAGE))
        if leverage <= 0 or leverage > DEFAULT_LEVERAGE:
            return False, "LEVERAGE_LIMIT", {}
        if available_balance is None or available_balance <= 0:
            return False, "BALANCE_UNKNOWN", {}
        if available_balance > DEFAULT_CAPITAL_CAP:
            available_balance = DEFAULT_CAPITAL_CAP
        quantity = round_down_step((available_balance * leverage) / price, filters.step_size)
        notional = quantity * price
        if quantity <= 0 or notional < filters.min_notional:
            return False, "MIN_NOTIONAL_NOT_MET", {"quantity": quantity, "notional": notional}
        client_order_id = deterministic_client_order_id(self.candidate_id, signal_id, symbol, "SHORT")
        return True, "READY", {"quantity": quantity, "notional": notional, "client_order_id": client_order_id, "leverage": leverage, "price": price}

    def submit(self, opportunity: dict[str, Any], filters: SymbolFilters, price: float, available_balance: float | None,
               dry_run: bool = True) -> dict[str, Any]:
        ok, reason, sizing = self.validate(opportunity, filters, price, available_balance)
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
