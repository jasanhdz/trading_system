#!/usr/bin/env python3
"""GEN2 Telegram notifier — Python side (watcher, monitor).

Reuses the SAME bot credentials as the TS bot (TELEGRAM_BOT_TOKEN /
TELEGRAM_CHAT_ID): env vars win; otherwise ONLY the TELEGRAM_* keys are read
from binance-futures-bot-ts/.env.gen2 and then .env (never Binance keys).

Severity + anti-spam contract:
- CRITICAL: always sent when the fingerprint is new; identical fingerprints
  are re-sent at most every 10 minutes.
- WARNING: identical fingerprints at most every 30 minutes.
- INFO: identical fingerprints at most every 6 hours (startup uses a unique
  fingerprint per process start, so it always goes out).

Never raises into the caller (a Telegram outage must not affect trading
paths) and never prints secrets — outgoing text is redacted against every
sensitive env var before sending.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aegis_alpha.tools.gen2_canary_core as core  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import utc_now  # noqa: E402

COOLDOWN_SECONDS = {"CRITICAL": 600, "WARNING": 1800, "INFO": 6 * 3600}
SEVERITY_ICON = {"CRITICAL": "🔴", "WARNING": "🟠", "INFO": "🟢"}
SENSITIVE_ENV = ("GEN2_BRIDGE_SECRET", "TELEGRAM_BOT_TOKEN", "TELEGRAM_LOG_BOT_TOKEN",
                 "BINANCE_API_KEY", "BINANCE_API_SECRET",
                 "BINANCE_FUTURES_API_KEY", "BINANCE_FUTURES_API_SECRET")
_TELEGRAM_KEYS = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")


def _load_telegram_env() -> dict[str, str]:
    creds = {k: os.environ.get(k, "") for k in _TELEGRAM_KEYS}
    if all(creds.values()):
        return creds
    for env_file in (REPO / "binance-futures-bot-ts" / ".env.gen2", REPO / "binance-futures-bot-ts" / ".env"):
        if not env_file.exists():
            continue
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            for key in _TELEGRAM_KEYS:
                if not creds[key] and line.startswith(f"{key}="):
                    creds[key] = line.split("=", 1)[1].strip().strip('"').strip("'")
        if all(creds.values()):
            break
    return creds


def redact(text: str) -> str:
    for name in SENSITIVE_ENV:
        value = os.environ.get(name)
        if value and len(value) >= 8:
            text = text.replace(value, "<REDACTED>")
    creds = _load_telegram_env()
    if creds.get("TELEGRAM_BOT_TOKEN") and len(creds["TELEGRAM_BOT_TOKEN"]) >= 8:
        text = text.replace(creds["TELEGRAM_BOT_TOKEN"], "<REDACTED>")
    return text


def _dedup_path(candidate_id: str) -> Path:
    return core.canary_dir(candidate_id) / "telegram_dedup.json"


def _should_send(candidate_id: str, severity: str, fingerprint: str, now_epoch: float) -> bool:
    path = _dedup_path(candidate_id)
    state = {}
    if path.exists():
        try:
            state = json.loads(path.read_text())
        except Exception:
            state = {}
    key = f"{severity}|{fingerprint}"
    last = float(state.get(key, 0))
    if now_epoch - last < COOLDOWN_SECONDS.get(severity, 600):
        return False
    state[key] = now_epoch
    # prune entries older than 24h so the file stays small
    state = {k: v for k, v in state.items() if now_epoch - float(v) < 86400}
    core.atomic_write(path, json.dumps(state))
    return True


def _post(token: str, chat_id: str, text: str, timeout: float = 5.0) -> bool:
    body = json.dumps({"chat_id": chat_id, "text": text[:4000], "parse_mode": "HTML"}).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - telegram API
        return json.loads(resp.read().decode()).get("ok") is True


def notify(candidate_id: str, severity: str, title: str, body: str = "",
           fingerprint: str | None = None, send_fn: Any = None,
           now_epoch: float | None = None) -> dict[str, Any]:
    """Send a severity-classified, deduplicated, redacted Telegram message.

    Returns a status dict; NEVER raises. Every attempt (sent/skipped/failed)
    is appended to telegram_log.jsonl for audit without secrets.
    """
    severity = severity.upper()
    result: dict[str, Any] = {"severity": severity, "title": title, "sent": False, "reason": ""}
    try:
        now_epoch = now_epoch if now_epoch is not None else utc_now().timestamp()
        fp = fingerprint or title
        if not _should_send(candidate_id, severity, fp, now_epoch):
            result["reason"] = "COOLDOWN_DEDUP"
        else:
            creds = _load_telegram_env()
            if not creds["TELEGRAM_BOT_TOKEN"] or not creds["TELEGRAM_CHAT_ID"]:
                result["reason"] = "TELEGRAM_NOT_CONFIGURED"
            else:
                text = redact(f"{SEVERITY_ICON.get(severity, '')} <b>[GEN2 {severity}]</b> {title}\n{body}".strip())
                sender = send_fn or (lambda t: _post(creds["TELEGRAM_BOT_TOKEN"], creds["TELEGRAM_CHAT_ID"], t))
                ok = bool(sender(text))
                result["sent"] = ok
                result["reason"] = "SENT" if ok else "TELEGRAM_API_NOT_OK"
    except Exception as exc:
        result["reason"] = f"NOTIFY_FAILED:{redact(repr(exc))[:160]}"
    try:
        core.append_jsonl(core.canary_dir(candidate_id) / "telegram_log.jsonl",
                          {"severity": severity, "title": redact(title), "sent": result["sent"], "reason": result["reason"]})
    except Exception:
        pass
    return result


def build_startup_message(candidate_id: str, state: dict[str, Any]) -> str:
    """Rich GEN2 startup message from REAL state (see gen2_decision_loop.collect_startup_state)."""
    contract = state.get("contract") or {}
    bridge = state.get("bridge") or {}
    ev = state.get("evidence") or {}
    models = state.get("models") or {}
    sym_analyzed = " ".join(s.replace("USDT", "") for s in state.get("symbols_analyzed", []))
    sym_exec = " ".join(state.get("symbols_executable", [])) or "ninguno (desarmado)"
    positions = bridge.get("open_positions") or []
    lines = [
        "🧠 <b>GEN2 CANARY ONLINE</b> ✅",
        "",
        "🔬 <b>Candidate</b>",
        f"ID: {candidate_id}",
        f"Mode: {contract.get('mode', 'SIN CONTRATO')}",
        f"Scientific freeze: {'VALID' if models.get('freeze_valid') else 'NOT VERIFIED'}",
        f"Env vs freeze: {models.get('environment', 'n/a')}",
        f"Execution config: {'ENABLED' if state.get('config_execution_enabled') else 'DISABLED'}",
        f"Emergency deny override: {'ON' if state.get('emergency_deny_override') else 'OFF'}",
        f"Effective execution: {'ENABLED' if state.get('effective_execution') else 'DISABLED'}",
        f"Armed: {'YES' if state.get('armed') else 'NO'} ({state.get('arm_reason', '')})",
        f"Phase O entries: {'PAUSED' if state.get('phase_o_paused') else '⚠️ NOT PAUSED'}",
        f"Kill switch: {'🔴 ON' if state.get('kill_switch') else 'OFF'}",
        "",
        "🎯 <b>Símbolos</b>",
        f"Analizados ({len(state.get('symbols_analyzed', []))}): {sym_analyzed}",
        f"Ejecutables: {sym_exec}",
        "",
        "💰 <b>Cuenta</b>",
        f"Disponible: ${bridge.get('available_balance') if bridge.get('available_balance') is not None else 'n/a (dry-run)'}",
        f"Posiciones abiertas: {len(positions)}",
        "",
        "🛡️ <b>Riesgo operativo</b>",
        f"Sizing: {contract.get('sizing_kind', 'n/a')}",
        f"Leverage: {contract.get('default_leverage', 'n/a')}x (max {contract.get('max_leverage', 'n/a')}x) | ISOLATED",
        f"Caps: daily {contract.get('daily_loss_pct', 'n/a')} | total {contract.get('total_loss_pct', 'n/a')} | floor {contract.get('equity_floor_fraction', 'n/a')}",
        f"Max posiciones: {contract.get('max_concurrent_positions', 'n/a')} | Riesgo gate: {state.get('risk_reason', 'n/a')}",
        "",
        "🧠 <b>Modelos</b>",
        f"TRRM: {models.get('trrm', 'n/a')} | QMAE: {models.get('qmae', 'n/a')} | EQM: {models.get('eqm', 'n/a')}",
        "",
        "🔌 <b>Servicios</b>",
        f"Bridge TS: {'healthy' if bridge.get('gen2_enabled') else '🔴 unreachable'}",
        "Watcher: starting (este mensaje)",
        "",
        "📊 <b>Evidencia</b>",
        f"Paper decisions: {ev.get('paper_decisions', ev.get('decisions', 0))} | Outcomes maduros: {ev.get('outcomes', 0)}",
        f"Dry-run requests: {ev.get('dryrun_requests', 0)} | Órdenes reales: {ev.get('real_order_submissions', 0)} | Fills reales: {ev.get('real_fills', 0)}",
        f"Incidentes: {ev.get('incidents', 0)}",
        "",
        "💼 <b>Posiciones</b>",
        "Ninguna" if not positions else json.dumps(positions)[:300],
    ]
    return "\n".join(lines)
