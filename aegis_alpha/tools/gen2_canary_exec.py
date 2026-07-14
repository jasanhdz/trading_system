#!/usr/bin/env python3
"""Gen2 canary execution-side helpers — reconciliation second opinion + emergency kill.

Architecture (approved review, GEN2_OPERATIONAL_MODES_SPEC.md): Aegis is the
brain, the TypeScript bridge is the ONLY order-submission path. This module
contains NO order submission. What lives here:

- market metadata (public exchangeInfo/prices) for sizing previews and the
  decision loop's symbol filters,
- the private READ-ONLY Binance adapter used exclusively for second-opinion
  reconciliation against the bridge's local streams and for engaging the
  emergency kill switch,
- BracketManager / Reconciler stream writers (fail-closed, kill on incident),
- a dry-run report wired to the single operational contract.

Risk limits, sizing and the arm token live ONLY in gen2_operational_contract.py.
"""
from __future__ import annotations

import argparse
import hmac
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aegis_alpha.tools.gen2_canary_core as core  # noqa: E402
import aegis_alpha.tools.gen2_operational_contract as oc  # noqa: E402
from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.gen2_canary_core import phase_o_new_entries_paused  # noqa: E402,F401  (single implementation lives in core)
from aegis_alpha.tools.gen2_d3_common import utc_now, validate_gen2_path  # noqa: E402

DEFAULT_CANDIDATE_ID = "gen2-20260711T202935Z"
DEFAULT_ALLOWED_SYMBOLS = ("ADAUSDT", "DOGEUSDT")
BINANCE_FAPI = "https://fapi.binance.com"
PYTHON_SUBMITS_ORDERS = False  # architectural invariant: only the TS bridge executes


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


def load_freeze(path: Path | None = None) -> dict[str, Any]:
    freeze = json.loads((path or core.FREEZE_PATH).read_text(encoding="utf-8"))
    if freeze.get("candidate_id") != DEFAULT_CANDIDATE_ID:
        raise ValueError(f"unexpected candidate_id {freeze.get('candidate_id')}")
    return freeze


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
    max_leverage: int = 20


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
    """USD-M Futures private READ adapter — second-opinion reconciliation only.

    Signs read/account requests if credentials are present in the environment.
    It has NO order methods by design: Python never submits Binance orders.
    Its only write-shaped power is engaging the local+bridge kill switch.
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

    def private_snapshot(self, symbols: tuple[str, ...] = DEFAULT_ALLOWED_SYMBOLS) -> dict[str, Any]:
        if not self.available:
            return {"private_read_ready": False, "reason": "PRIVATE_CREDENTIALS_NOT_AVAILABLE"}
        account = self.account()
        positions = {s: self.position_risk(s) for s in symbols}
        open_orders = {s: self.open_orders(s) for s in symbols}
        brackets = {s: self.leverage_bracket(s) for s in symbols}
        return {
            "private_read_ready": True,
            "account_equity": account.get("totalWalletBalance"),
            "available_balance": account.get("availableBalance"),
            "positions": positions,
            "open_orders": open_orders,
            "leverage_brackets": brackets,
        }

    def emergency_kill(self, candidate_id: str, reason: str) -> dict[str, Any]:
        """Engage local kill switch and (best effort) the TS bridge kill. No order calls."""
        core.engage_kill_switch(candidate_id, f"EMERGENCY:{reason}")
        bridge_ack: dict[str, Any] | None = None
        try:
            import aegis_alpha.tools.gen2_bridge_client as bridge

            bridge_ack = bridge.post_kill(candidate_id, reason)
        except Exception as exc:
            bridge_ack = {"status": "BRIDGE_KILL_UNREACHABLE", "reason": redact_secret_text(repr(exc))}
        return {"local_kill": True, "bridge_kill_ack": bridge_ack}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # torn trailing line after a power cut; append-only streams tolerate skip
    return rows


def second_opinion_reconciliation(candidate_id: str, adapter: BinancePrivateReadOnlyAdapter | None = None,
                                  symbols: tuple[str, ...] = DEFAULT_ALLOWED_SYMBOLS,
                                  bridge_status: dict[str, Any] | None = None,
                                  balance_tolerance_pct: float = 0.02) -> dict[str, Any]:
    """Compare exchange truth (private read) vs local live_orders/fills/brackets streams.

    Detects: orphan orders/positions, duplicated fills, fills without confirmed
    brackets, balance mismatch vs the bridge's reported balance, and leverage /
    margin-mode drift vs THE operational contract. Fail-closed: any incident
    while exposure exists engages the kill switch.
    """
    adapter = adapter or BinancePrivateReadOnlyAdapter()
    cdir = core.canary_dir(candidate_id)
    local_orders = _read_jsonl(cdir / "live_orders.jsonl")
    local_ids = {r.get("order", {}).get("client_order_id") for r in local_orders if r.get("order")}
    local_ids |= {r.get("client_order_id") for r in local_orders if r.get("client_order_id")}
    local_ids.discard(None)
    incidents: list[str] = []
    details: dict[str, Any] = {}

    # local-stream consistency (does not need exchange access)
    fill_rows = [r for r in _read_jsonl(cdir / "fills.jsonl") if r.get("type") in (None, "FILL")]
    fill_counts: dict[str, int] = {}
    for r in fill_rows:
        coid = str(r.get("client_order_id"))
        fill_counts[coid] = fill_counts.get(coid, 0) + 1
    duplicated_fills = sorted(c for c, n in fill_counts.items() if n > 1)
    if duplicated_fills:
        incidents.append("DUPLICATED_FILLS")
        details["duplicated_fills"] = duplicated_fills
    bracket_ok_ids = {str(r.get("client_order_id")) for r in _read_jsonl(cdir / "brackets.jsonl")
                      if r.get("ok") is True or r.get("type") == "BRACKET_CONFIRMED"}
    missing_brackets = sorted(set(fill_counts) - bracket_ok_ids)
    if missing_brackets:
        incidents.append("FILL_WITHOUT_CONFIRMED_BRACKET")
        details["missing_brackets"] = missing_brackets

    if not adapter.available:
        record = {"schema": "gen2_second_opinion_v2", "candidate_id": candidate_id,
                  "status": "PRIVATE_READ_UNAVAILABLE", "incidents": incidents, **details,
                  "local_order_ids": sorted(local_ids), "recorded_at_utc": utc_now().isoformat()}
        append_jsonl(cdir / "reconciliations.jsonl", record)
        return record

    snapshot = adapter.private_snapshot(symbols)
    exchange_orders = {o.get("clientOrderId") for rows in snapshot.get("open_orders", {}).values() for o in rows}
    exchange_positions = [
        pos for rows in snapshot.get("positions", {}).values() for pos in rows
        if abs(float(pos.get("positionAmt", 0) or 0)) > 0
    ]
    gen2_exchange_orders = {o for o in exchange_orders if o and o.startswith("GEN2-")}
    orphan_orders = sorted(gen2_exchange_orders - local_ids)
    if orphan_orders:
        incidents.append("ORPHAN_ORDER_ON_EXCHANGE")
        details["orphan_orders"] = orphan_orders
    if exchange_positions and not local_ids:
        incidents.append("POSITION_WITHOUT_LOCAL_ORDER")

    # contract consistency on any open position (leverage / margin mode)
    contract: dict[str, Any] | None = None
    try:
        contract = oc.load_contract(candidate_id)
    except Exception:
        pass
    if contract is not None:
        for pos in exchange_positions:
            lev = int(float(pos.get("leverage", 0) or 0))
            if lev and lev > int(contract["max_leverage"]):
                incidents.append("LEVERAGE_ABOVE_CONTRACT")
                details.setdefault("leverage_violations", []).append({"symbol": pos.get("symbol"), "leverage": lev})
            margin_type = str(pos.get("marginType", pos.get("margin_type", ""))).upper()
            if margin_type and margin_type != "ISOLATED":
                incidents.append("MARGIN_MODE_NOT_ISOLATED")
                details.setdefault("margin_violations", []).append({"symbol": pos.get("symbol"), "margin_type": margin_type})

    # balance second opinion vs what the bridge reported to the brain
    exchange_balance = snapshot.get("available_balance")
    if bridge_status is not None and exchange_balance is not None:
        bridge_balance = bridge_status.get("available_balance")
        if bridge_balance is not None:
            ref = max(abs(float(exchange_balance)), 1e-9)
            drift = abs(float(exchange_balance) - float(bridge_balance)) / ref
            details["balance_drift_pct"] = drift
            if drift > balance_tolerance_pct:
                incidents.append("BALANCE_MISMATCH_BRIDGE_VS_EXCHANGE")

    status = "RECONCILED" if not incidents else "RECONCILIATION_FAIL_CLOSED"
    record = {"schema": "gen2_second_opinion_v2", "candidate_id": candidate_id, "status": status,
              "incidents": sorted(set(incidents)), **details,
              "open_position_count": len(exchange_positions),
              "exchange_available_balance": exchange_balance,
              "local_order_ids": sorted(local_ids), "recorded_at_utc": utc_now().isoformat()}
    append_jsonl(cdir / "reconciliations.jsonl", record)
    if incidents and exchange_positions:
        core.engage_kill_switch(candidate_id, "SECOND_OPINION_MISMATCH_WITH_EXPOSURE")
    return record


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


def dry_run(candidate_id: str, use_public: bool = False, private_read: bool = False) -> dict[str, Any]:
    """Read-only architecture rehearsal against the unified operational contract."""
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
    private_snapshot: dict[str, Any] | None = None
    if private_read:
        private_snapshot = BinancePrivateReadOnlyAdapter().private_snapshot()
        if private_snapshot.get("private_read_ready"):
            account = {
                "available_balance": float(private_snapshot.get("available_balance")),
                "wallet_balance": float(private_snapshot.get("account_equity")),
                "account_equity": float(private_snapshot.get("account_equity")),
                "source": "binance_private_read_only",
                "open_position": any(
                    abs(float(pos.get("positionAmt", 0))) > 0
                    for rows in private_snapshot.get("positions", {}).values()
                    for pos in rows
                ),
            }
    available = account.get("available_balance")
    contract: dict[str, Any] | None = None
    contract_error: str | None = None
    try:
        contract = oc.load_contract(candidate_id)
    except Exception as exc:
        contract_error = repr(exc)
    sizing_preview: dict[str, Any] = {}
    if contract is not None and available is not None:
        for symbol in DEFAULT_ALLOWED_SYMBOLS:
            f = filters[symbol]
            sizing_preview[symbol] = oc.compute_sizing(contract, prices[symbol], float(available), f.step_size, f.min_notional, f.min_qty)
    armed, arm_reason, _token = oc.verify_arm_token(candidate_id)
    risk_ok, risk_reason = oc.risk_gate(candidate_id)
    brackets = BracketManager(candidate_id).confirm({"client_order_id": "mock-fill", "filled_qty": 1.0}, True, True, 3.0)
    bracket_fail = BracketManager(candidate_id).confirm({"client_order_id": "mock-fill-fail", "filled_qty": 0.0}, False, False, 61.0)
    reconciliation = Reconciler(candidate_id).reconcile({"open_orders": []}, {"open_orders": [], "exposure": 0})
    report = {
        "candidate_id": candidate_id,
        "armed": armed,
        "arm_reason": arm_reason,
        "risk_gate": risk_reason if not risk_ok else "RISK_OK",
        "operational_mode": contract.get("mode") if contract else None,
        "operational_contract_error": contract_error,
        "phase_o": phase_o_new_entries_paused()[1],
        "python_submits_orders": PYTHON_SUBMITS_ORDERS,
        "orders_submitted": 0,
        "sizing_preview": sizing_preview,
        "bracket_success": brackets,
        "bracket_failure_mock": bracket_fail,
        "reconciliation": reconciliation,
        "account_snapshot": account,
        "private_snapshot_status": private_snapshot,
        "public_market_data_error": public_error,
        "FORWARD_OUTCOMES_NOT_EVALUATED": True,
    }
    atomic_write(cdir / "dry_run_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gen2 canary reconciliation/second-opinion helpers (no order submission)")
    parser.add_argument("--mode", choices=["dry-run", "second-opinion"], default="dry-run")
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--public-market-data", action="store_true")
    parser.add_argument("--private-read", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "second-opinion":
        payload = second_opinion_reconciliation(args.candidate_id)
    else:
        payload = dry_run(args.candidate_id, use_public=args.public_market_data, private_read=args.private_read)
    print(json.dumps(payload, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
