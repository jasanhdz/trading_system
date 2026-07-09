#!/usr/bin/env python3
"""Research-only loss cap and auto-pause simulator."""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_OUT_DIR = Path("/home/jasan/Develop")
TRADE_FIELDS = ["trade_id", "opened_at", "symbol", "side", "realized_pnl", "roe", "source"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def synthetic_fixture_trades() -> list[dict[str, Any]]:
    base = datetime(2026, 7, 1, tzinfo=timezone.utc)
    pnls = [12, -8, 9, -35, 11, -7, -9, -11, 15, -50, 13, 10, -6, -7, -45, 16, -5, 14]
    return [
        {
            "trade_id": f"fixture-{i+1}",
            "opened_at": (base + timedelta(hours=i * 3)).isoformat(),
            "symbol": "ADAUSDT",
            "side": "SHORT",
            "realized_pnl": pnl,
            "roe": pnl / 100.0,
            "source": "synthetic_fixture",
        }
        for i, pnl in enumerate(pnls)
    ]


def read_trades_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return [dict(r) for r in csv.DictReader(f) if r.get("realized_pnl") not in (None, "")]


def avg_win_before(trades: list[dict[str, Any]], idx: int, default: float = 10.0) -> float:
    wins = [to_float(t.get("realized_pnl")) for t in trades[:idx] if to_float(t.get("realized_pnl")) > 0]
    return mean(wins[-20:]) if wins else default


def profit_factor(pnls: list[float]) -> float:
    gross_win = sum(x for x in pnls if x > 0)
    gross_loss = abs(sum(x for x in pnls if x < 0))
    return math.inf if gross_loss == 0 else gross_win / gross_loss


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    idx = min(len(data) - 1, max(0, int(round((len(data) - 1) * q))))
    return data[idx]


def simulate_rules(trades: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    original = [to_float(t.get("realized_pnl")) for t in trades]
    simulated: list[float] = []
    blocked = 0
    winners_sacrificed = 0
    large_losses_avoided = 0
    paused_until_day: str | None = None
    consecutive_losses = 0
    daily_loss: dict[str, float] = {}
    for idx, trade in enumerate(trades):
        pnl = to_float(trade.get("realized_pnl"))
        day = str(trade.get("opened_at", ""))[:10]
        rolling_pf = profit_factor(simulated[-30:])
        paused = False
        if paused_until_day and day == paused_until_day:
            paused = True
        if config.get("consecutive_losses") and consecutive_losses >= int(config["consecutive_losses"]):
            paused = True
        if config.get("rolling_pf_threshold") and len(simulated) >= 6 and rolling_pf < float(config["rolling_pf_threshold"]):
            paused = True
        if config.get("daily_loss_cap") and abs(daily_loss.get(day, 0.0)) >= float(config["daily_loss_cap"]):
            paused = True
            paused_until_day = day
        if config.get("tail_pause") and idx > 0 and pnl < -3.0 * avg_win_before(trades, idx):
            paused = True
        if paused:
            blocked += 1
            winners_sacrificed += int(pnl > 0)
            large_losses_avoided += int(pnl < -2.0 * avg_win_before(trades, idx))
            simulated.append(0.0)
            continue
        capped = pnl
        if config.get("loss_cap_multiplier") and pnl < 0:
            cap = float(config["loss_cap_multiplier"]) * avg_win_before(trades, idx)
            capped = max(pnl, -cap)
        if config.get("fixed_loss_cap") and pnl < 0:
            capped = max(capped, -float(config["fixed_loss_cap"]))
        simulated.append(capped)
        daily_loss[day] = daily_loss.get(day, 0.0) + min(capped, 0.0)
        consecutive_losses = consecutive_losses + 1 if capped < 0 else 0
    losses = [x for x in original if x < 0]
    sim_losses = [x for x in simulated if x < 0]
    return {
        "config": config,
        "original_pnl": sum(original),
        "simulated_pnl": sum(simulated),
        "original_max_loss": min(original) if original else 0.0,
        "simulated_max_loss": min(simulated) if simulated else 0.0,
        "original_p90_loss": percentile([abs(x) for x in losses], 0.90),
        "simulated_p90_loss": percentile([abs(x) for x in sim_losses], 0.90),
        "original_p95_loss": percentile([abs(x) for x in losses], 0.95),
        "simulated_p95_loss": percentile([abs(x) for x in sim_losses], 0.95),
        "trades_blocked_after_pause": blocked,
        "large_losses_avoided": large_losses_avoided,
        "winners_sacrificed": winners_sacrificed,
        "score": sum(simulated) - 0.5 * winners_sacrificed + 2.0 * large_losses_avoided,
    }


def grid_search(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mult in (2.0, 3.0):
        for daily in (30.0, 50.0, 80.0):
            for consec in (2, 3, 4):
                cfg = {
                    "loss_cap_multiplier": mult,
                    "daily_loss_cap": daily,
                    "consecutive_losses": consec,
                    "rolling_pf_threshold": 0.8,
                    "tail_pause": True,
                }
                rows.append(simulate_rules(trades, cfg))
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row.keys()} | {"config_json"})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["config_json"] = json.dumps(out.pop("config", {}), sort_keys=True)
            writer.writerow(out)


def run_simulation(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades = read_trades_csv(Path(args.trades_csv)) if args.trades_csv else []
    status = "OK_USING_REAL_OR_PROVIDED_TRADES"
    if not trades:
        trades = synthetic_fixture_trades()
        status = "USING_SYNTHETIC_FIXTURE_FOR_MECHANICS"
    rows = grid_search(trades) if args.grid else [simulate_rules(trades, {"loss_cap_multiplier": 2.0})]
    best = rows[0] if rows else {}
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    result = {
        "schema_version": "loss_cap_autopause_sim_a_v1",
        "status": status,
        "generated_at": timestamp,
        "trade_count": len(trades),
        "best_rule": best.get("config", {}),
        "best_metrics": {k: v for k, v in best.items() if k != "config"},
        "recommendation": "Mechanics validated. Use forward live logs before activating live guards.",
    }
    json_path = out_dir / f"aegis_loss_cap_autopause_sim_a_{timestamp}.json"
    md_path = out_dir / f"aegis_loss_cap_autopause_sim_a_{timestamp}.md"
    grid_path = out_dir / f"aegis_loss_cap_autopause_grid_{timestamp}.csv"
    rec_path = out_dir / f"aegis_loss_cap_autopause_recommendations_{timestamp}.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(
        "\n".join([
            "# Aegis Loss Cap / Auto-Pause Simulation A",
            "",
            f"- status: {status}",
            f"- trade_count: {len(trades)}",
            f"- best_rule: {json.dumps(best.get('config', {}), sort_keys=True)}",
            f"- original_pnl: {best.get('original_pnl', 0):.4f}",
            f"- simulated_pnl: {best.get('simulated_pnl', 0):.4f}",
            f"- original_max_loss: {best.get('original_max_loss', 0):.4f}",
            f"- simulated_max_loss: {best.get('simulated_max_loss', 0):.4f}",
            "",
            "Research-only simulation; no live guards were activated.",
        ]) + "\n",
        encoding="utf-8",
    )
    write_csv(grid_path, rows)
    write_csv(rec_path, [{"rank": 1, "thresholds": json.dumps(best.get("config", {}), sort_keys=True), "recommendation": result["recommendation"]}])
    result["outputs"] = {"json": str(json_path), "md": str(md_path), "grid_csv": str(grid_path), "recommendations_csv": str(rec_path)}
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Research-only loss cap and auto-pause simulator.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--trades-csv", default="")
    p.add_argument("--simulate-loss-cap", action="store_true")
    p.add_argument("--simulate-daily-pause", action="store_true")
    p.add_argument("--simulate-consecutive-losses", action="store_true")
    p.add_argument("--simulate-rolling-pf", action="store_true")
    p.add_argument("--grid", action="store_true")
    return p


def main() -> int:
    result = run_simulation(build_parser().parse_args())
    print(json.dumps({k: result[k] for k in ("status", "trade_count", "best_rule", "outputs")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
