#!/usr/bin/env python3
"""Read-only Phase O.4 trade audit from local bot state and structured logs."""
from __future__ import annotations
import argparse, csv, json, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TS = ROOT / "binance-futures-bot-ts"
PROMOTION = "2026-06-01T19:41:00Z"

def parse_time(value: str | None) -> datetime | None:
    if not value: return None
    if value == "now": return datetime.now(timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

def read_jsonl(path: Path):
    if not path.exists(): return []
    out=[]
    for line in path.read_text(errors="replace").splitlines():
        try: out.append(json.loads(line))
        except json.JSONDecodeError: pass
    return out

def pick(d: dict, *keys, default=None):
    for key in keys:
        if key in d and d[key] is not None: return d[key]
    return default

def classify_source(row: dict) -> str:
    side=str(pick(row,"side",default="")).upper()
    strategy=str(pick(row,"finalStrategy","strategy",default="")).lower()
    opened=parse_time(pick(row,"openedAt","opened_at","timestamp"))
    if "momentum" in strategy: return "momentum_ride"
    if opened and opened >= parse_time(PROMOTION):
        if side == "SHORT": return "phase_o_short"
        if side == "LONG": return "legacy_long"
    return "unknown"

def public_mark(symbol: str):
    url=f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            return float(json.loads(r.read())["price"])
    except Exception:
        return None

def normalize(row: dict, read_public=False) -> dict:
    meta=row.get("metadata") or {}
    guard=meta.get("guardTrace") or meta.get("guard_trace") or row.get("guardTrace") or {}
    entry=float(pick(row,"entryPrice","entry_price",default=0) or 0)
    qty=float(pick(row,"quantity","qty",default=0) or 0)
    lev=float(pick(row,"leverage",default=0) or 0)
    notional=float(pick(row,"notional",default=entry*qty) or entry*qty)
    margin=float(pick(row,"margin",default=(notional/lev if lev else 0)) or 0)
    opened=pick(row,"openedAt","opened_at")
    closed=pick(row,"closedAt","closed_at")
    mark=public_mark(str(row.get("symbol",""))) if read_public and not closed else None
    pnl=pick(row,"pnl","pnl_usdt","realizedPnl","realized_pnl","unrealizedPnl","unrealized_pnl")
    roe=pick(row,"roe","realizedRoe","unrealizedRoe")
    return {
      "trade_id": pick(row,"tradeId","trade_id","id",default=""), "symbol": row.get("symbol",""),
      "side": str(row.get("side","")).upper(), "strategy": pick(row,"strategy",default=""),
      "finalStrategy": pick(row,"finalStrategy",default=""), "finalReason": pick(row,"finalReason",default=""),
      "source_phase": classify_source(row), "opened_at": opened, "closed_at": closed, "status": "CLOSED" if closed else "OPEN",
      "entry_price": entry, "current_mark": mark, "exit_price": pick(row,"exitPrice","exit_price"), "qty": qty,
      "leverage": lev, "margin": margin, "notional": notional,
      "position_fraction": pick(row,"positionFraction","position_fraction"),
      "configured_fraction_source": pick(row,"configuredFractionSource","configured_fraction_source",default="local_trade_log"),
      "sl": pick(row,"stopLossPrice","sl","stop_loss"), "tp": pick(row,"takeProfitPrice","tp","take_profit"),
      "brackets_confirmed": bool(pick(row,"bracketsConfirmed","brackets_confirmed",default=False)),
      "breakeven_armed": bool(pick(row,"breakEvenArmed","breakeven_armed",default=False)),
      "breakeven_executed": bool(pick(row,"breakEvenExecuted","breakeven_executed",default=False)),
      "trailing_armed": bool(pick(row,"trailingArmed","trailing_armed",default=False)),
      "pnl": pnl, "roe": roe, "mae": pick(row,"mae","mae_roe"), "mfe": pick(row,"mfe","mfe_roe"),
      "clean_entry": guard.get("clean_entry") or guard.get("cleanEntry"), "event_risk": guard.get("event_risk") or guard.get("eventRisk"),
      "entry_quality": guard.get("entry_quality") or guard.get("entryQuality"), "decision_brain": guard.get("decision_brain") or guard.get("decisionBrain"),
      "regime": guard.get("regime"), "short_gate": guard.get("short_gate") or guard.get("shortGate"),
      "phase_o_shadow_scope": guard.get("phase_o_shadow_scope") or meta.get("phase_o_short_guard_modes_applied"),
      "whether_hard_safety_respected": bool(pick(row,"bracketsConfirmed","brackets_confirmed",default=False)),
      "notes": "ERROR_LINK_ENTRY" if row.get("symbol")=="LINKUSDT" else ""
    }

def load_trades() -> list[dict]:
    records=[]
    for path in sorted((TS/"logs"/"aegis").glob("turbo_trades_*.jsonl")):
        records += read_jsonl(path)
    merged={}
    for row in records:
        tid=pick(row,"tradeId","trade_id","id",default=f"{row.get('symbol')}:{row.get('openedAt')}:{row.get('side')}")
        cur=merged.setdefault(str(tid), {})
        cur.update({k:v for k,v in row.items() if v is not None})
    return list(merged.values())

def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--from",dest="from_ts",default="2026-06-01T19:00:00Z")
    ap.add_argument("--to",default="now"); ap.add_argument("--out-dir",default="/home/jasan/Develop")
    ap.add_argument("--include-open",action="store_true"); ap.add_argument("--include-closed",action="store_true")
    ap.add_argument("--read-binance-public",action="store_true"); ap.add_argument("--read-account-local-only",action="store_true")
    args=ap.parse_args(); start,end=parse_time(args.from_ts),parse_time(args.to)
    include_open=args.include_open or not args.include_closed; include_closed=args.include_closed or not args.include_open
    rows=[]
    for raw in load_trades():
        opened=parse_time(pick(raw,"openedAt","opened_at")); closed=parse_time(pick(raw,"closedAt","closed_at"))
        if not ((opened and start <= opened <= end) or (closed and start <= closed <= end)): continue
        row=normalize(raw,args.read_binance_public)
        if row["status"]=="OPEN" and include_open or row["status"]=="CLOSED" and include_closed: rows.append(row)
    rows.sort(key=lambda x:x.get("opened_at") or "")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    base=out/f"aegis_phase_o4_open_trades_audit_{stamp}"
    summary=base.with_name(f"aegis_phase_o4_open_trades_summary_{stamp}.csv"); traces=base.with_name(f"aegis_phase_o4_trade_traces_{stamp}.csv")
    fields=list(rows[0]) if rows else ["trade_id","symbol","side","source_phase","status"]
    write_csv(summary,rows,fields)
    write_csv(traces,rows,["trade_id","symbol","side","source_phase","clean_entry","event_risk","entry_quality","decision_brain","regime","short_gate","phase_o_shadow_scope","notes"])
    payload={"safety":"READ_ONLY","from":args.from_ts,"to":args.to,"trade_count":len(rows),"open_count":sum(r['status']=='OPEN' for r in rows),"closed_count":sum(r['status']=='CLOSED' for r in rows),"link_entry_errors":sum(r['symbol']=='LINKUSDT' for r in rows),"trades":rows,"sources":[str(p) for p in sorted((TS/'logs'/'aegis').glob('turbo_trades_*.jsonl'))]}
    base.with_suffix(".json").write_text(json.dumps(payload,indent=2,default=str)+"\n")
    md=["# Phase O.4 Open Trades Audit","","## Safety","READ_ONLY. No orders, config changes, or service restarts.","",f"Trades: {len(rows)} | open: {payload['open_count']} | closed: {payload['closed_count']} | LINK entries: {payload['link_entry_errors']}","","| opened | symbol | side | source | status | lev | margin | notional | pnl | roe | brackets |","|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows: md.append(f"| {r['opened_at']} | {r['symbol']} | {r['side']} | {r['source_phase']} | {r['status']} | {r['leverage']} | {r['margin']} | {r['notional']} | {r['pnl']} | {r['roe']} | {r['brackets_confirmed']} |")
    base.with_suffix(".md").write_text("\n".join(md)+"\n")
    print(json.dumps({"report":str(base.with_suffix('.md')),"json":str(base.with_suffix('.json')),"summary_csv":str(summary),"traces_csv":str(traces),"trades":len(rows)},indent=2))
if __name__=="__main__": main()
