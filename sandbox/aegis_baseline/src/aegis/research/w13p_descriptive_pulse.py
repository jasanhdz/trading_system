"""Outcome-blind descriptive pulse for collected W13-P signal bundles."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from aegis.research.prospective_microstructure_w13p import CURRENT_QUALITY_GATE_VERSION


HORIZONS_SECONDS = (1, 2, 5, 10, 20, 30, 60, 120, 180)
BARRIERS_BPS = (5.0, 10.0, 14.0, 20.0)
INDEPENDENCE_GAP_SECONDS = 15 * 60


def _read_parts(root: Path, kind: str) -> pd.DataFrame:
    paths = sorted((root / kind).rglob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    frames = [pq.read_table(path).to_pandas() for path in paths]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return pd.concat(frames, ignore_index=True)


def _direction(side: str) -> float:
    return 1.0 if side == "LONG" else -1.0


def _mid(payload: dict[str, Any]) -> float:
    return (float(payload["b"]) + float(payload["a"])) / 2.0


def _directional_bps(price: float, reference: float, side: str) -> float:
    return _direction(side) * (price / reference - 1.0) * 10_000.0


def first_barrier(path: Iterable[tuple[int, float]], barrier_bps: float) -> str:
    for _, value in path:
        if value >= barrier_bps:
            return "FAVORABLE_FIRST"
        if value <= -barrier_bps:
            return "ADVERSE_FIRST"
    return "NEITHER"


def _safe_spearman(left: pd.Series, right: pd.Series) -> float | None:
    valid = pd.concat((left, right), axis=1).dropna()
    if len(valid) < 5 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return None
    value = valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman")
    return None if pd.isna(value) else float(value)


def _cluster_ids(frame: pd.DataFrame) -> pd.Series:
    result: dict[int, str] = {}
    for symbol, group in frame.sort_values("signal_timestamp_us").groupby("symbol"):
        cluster = 0
        previous: int | None = None
        for index, row in group.iterrows():
            stamp = int(row["signal_timestamp_us"])
            if previous is None or stamp - previous > INDEPENDENCE_GAP_SECONDS * 1_000_000:
                cluster += 1
            result[index] = f"{symbol}-{cluster:03d}"
            previous = stamp
    return pd.Series(result).reindex(frame.index)


def build_episode_table(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    quality = _read_parts(root, "quality")
    signals = _read_parts(root, "signal")
    events = _read_parts(root, "event")
    if quality.empty or signals.empty or events.empty:
        return pd.DataFrame(), {"reason": "MISSING_COLLECTION_PARTS"}

    quality = quality.drop_duplicates("signal_id", keep="last")
    signals = signals.drop_duplicates("signal_id", keep="last")
    eligible = quality.loc[
        quality["W13_ELIGIBLE"].astype(bool)
        & quality["quality_gate_version"].eq(CURRENT_QUALITY_GATE_VERSION)
    ]
    selected = signals.merge(
        eligible[["signal_id", "window_start_us", "window_end_us", "max_gap_ms"]],
        on="signal_id",
        how="inner",
        validate="one_to_one",
    ).sort_values("signal_timestamp_us")

    events = events.drop_duplicates("event_id", keep="first")
    events["payload"] = events["payload_json"].map(json.loads)
    by_symbol = {symbol: group.sort_values("exchange_event_timestamp_us") for symbol, group in events.groupby("symbol")}
    rows: list[dict[str, Any]] = []
    for _, signal in selected.iterrows():
        symbol = str(signal["symbol"])
        side = str(signal["side"])
        t0 = int(signal["signal_timestamp_us"])
        end = int(signal["window_end_us"])
        reference = float(signal["reference_mid"])
        event_frame = by_symbol.get(symbol, pd.DataFrame())
        window = event_frame.loc[
            event_frame["exchange_event_timestamp_us"].between(t0, end, inclusive="both")
        ]
        quotes = window.loc[window["event_type"].eq("QUOTE")].copy()
        trades = window.loc[window["event_type"].eq("TRADE")].copy()
        quote_path = [
            (int(event.exchange_event_timestamp_us), _directional_bps(_mid(event.payload), reference, side))
            for event in quotes.itertuples()
        ]
        row: dict[str, Any] = {
            "signal_id": str(signal["signal_id"]),
            "timestamp_utc": datetime.fromtimestamp(t0 / 1_000_000, UTC).isoformat(),
            "signal_timestamp_us": t0,
            "symbol": symbol,
            "side": side,
            "reference_mid": reference,
            "max_gap_ms": float(signal["max_gap_ms"]),
            "quote_events": int(len(quotes)),
            "trade_events": int(len(trades)),
            "first_quote_delay_ms": (
                (int(quotes.iloc[0]["exchange_event_timestamp_us"]) - t0) / 1000.0 if len(quotes) else math.nan
            ),
            "first_trade_delay_ms": (
                (int(trades.iloc[0]["exchange_event_timestamp_us"]) - t0) / 1000.0 if len(trades) else math.nan
            ),
        }
        for horizon in HORIZONS_SECONDS:
            cutoff = t0 + horizon * 1_000_000
            values = [value for stamp, value in quote_path if stamp <= cutoff]
            row[f"return_{horizon}s_bps"] = values[-1] if values else 0.0
            row[f"mfe_{horizon}s_bps"] = max((0.0, *values))
            row[f"mae_{horizon}s_bps"] = max((0.0, *(-value for value in values)))

            horizon_trades = trades.loc[trades["exchange_event_timestamp_us"].le(cutoff)]
            buy_notional = 0.0
            sell_notional = 0.0
            for payload in horizon_trades["payload"]:
                notional = float(payload["p"]) * float(payload["q"])
                if bool(payload.get("m")):
                    sell_notional += notional
                else:
                    buy_notional += notional
            total = buy_notional + sell_notional
            raw_imbalance = (buy_notional - sell_notional) / total if total else math.nan
            row[f"directional_flow_{horizon}s"] = _direction(side) * raw_imbalance

        for barrier in BARRIERS_BPS:
            label = str(int(barrier)) if barrier.is_integer() else str(barrier).replace(".", "_")
            row[f"barrier_{label}bps"] = first_barrier(quote_path, barrier)

        source = json.loads(str(signal["source_envelope_json"]))
        d3 = source.get("component_evidence", {}).get("d3", {}).get("output", {})
        upstream = source.get("upstream_model", {})
        row.update({
            "regime": str(d3.get("regime", "UNKNOWN")),
            "regime_confidence": d3.get("regime_confidence"),
            "expected_return_bps": abs(float(upstream.get("expected_return", 0.0))) * 10_000.0,
            "tail_risk_probability": upstream.get("tail_risk_probability"),
            "qmae_q90_bps": float(upstream.get("qmae_q90", 0.0)) * 10_000.0,
        })
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, {"reason": "NO_STRICT_ELIGIBLE_SIGNALS"}
    frame["cluster_id_15m"] = _cluster_ids(frame)
    frame["behavior_14bps"] = frame["barrier_14bps"].map({
        "FAVORABLE_FIRST": "GOOD_PROXY",
        "ADVERSE_FIRST": "BAD_PROXY",
        "NEITHER": "INDETERMINATE",
    })
    audit = {
        "strict_signals": int(len(frame)),
        "independence_clusters_15m": int(frame["cluster_id_15m"].nunique()),
        "symbols": sorted(frame["symbol"].unique().tolist()),
        "directions": frame["side"].value_counts().sort_index().astype(int).to_dict(),
        "start_utc": str(frame["timestamp_utc"].min()),
        "end_utc": str(frame["timestamp_utc"].max()),
    }
    return frame, audit


def summarize(frame: pd.DataFrame, audit: dict[str, Any]) -> dict[str, Any]:
    if frame.empty:
        return {"schema_id": "aegis-w13p-descriptive-pulse-v1", "audit": audit}
    horizons = {}
    for horizon in HORIZONS_SECONDS:
        returns = frame[f"return_{horizon}s_bps"]
        horizons[str(horizon)] = {
            "median_return_bps": float(returns.median()),
            "favorable_fraction": float(returns.gt(0).mean()),
            "median_mfe_bps": float(frame[f"mfe_{horizon}s_bps"].median()),
            "median_mae_bps": float(frame[f"mae_{horizon}s_bps"].median()),
            "mfe_at_least_14bps_fraction": float(frame[f"mfe_{horizon}s_bps"].ge(14.0).mean()),
        }
    barrier_counts = {
        str(int(barrier)): frame[f"barrier_{int(barrier)}bps"].value_counts().sort_index().astype(int).to_dict()
        for barrier in BARRIERS_BPS
    }
    behavior = {}
    for label, group in frame.groupby("behavior_14bps"):
        behavior[label] = {
            "signals": int(len(group)),
            "symbols": int(group["symbol"].nunique()),
            "median_return_10s_bps": float(group["return_10s_bps"].median()),
            "median_return_30s_bps": float(group["return_30s_bps"].median()),
            "median_directional_flow_10s": float(group["directional_flow_10s"].median()) if group["directional_flow_10s"].notna().any() else None,
            "median_directional_flow_30s": float(group["directional_flow_30s"].median()) if group["directional_flow_30s"].notna().any() else None,
            "median_mfe_180s_bps": float(group["mfe_180s_bps"].median()),
            "median_mae_180s_bps": float(group["mae_180s_bps"].median()),
        }
    by_symbol = {}
    for symbol, group in frame.groupby("symbol"):
        by_symbol[symbol] = {
            "signals": int(len(group)),
            "clusters_15m": int(group["cluster_id_15m"].nunique()),
            "median_return_180s_bps": float(group["return_180s_bps"].median()),
            "median_mfe_180s_bps": float(group["mfe_180s_bps"].median()),
            "median_mae_180s_bps": float(group["mae_180s_bps"].median()),
            "behavior_14bps": group["behavior_14bps"].value_counts().sort_index().astype(int).to_dict(),
        }
    correlations = {
        f"flow_{horizon}s_vs_return_180s": _safe_spearman(
            frame[f"directional_flow_{horizon}s"], frame["return_180s_bps"]
        )
        for horizon in (5, 10, 30, 60)
    }
    return {
        "schema_id": "aegis-w13p-descriptive-pulse-v1",
        "status": "DESCRIPTIVE_ONLY_NOT_TRAINING_NOT_EDGE_EVIDENCE",
        "audit": audit,
        "horizons": horizons,
        "first_barriers": barrier_counts,
        "behavior_14bps": behavior,
        "per_symbol": by_symbol,
        "spearman_descriptive": correlations,
        "limitations": [
            "All strict signals are SHORT.",
            "The sample spans only a few days and contains temporally correlated signals.",
            "GOOD_PROXY/BAD_PROXY are fixed 14-bps path descriptions, not training labels or validated rules.",
            "MFE is available movement, not realizable PnL without a frozen exit policy.",
            "No threshold, model, split, holdout or production decision was created.",
        ],
    }


def render_report(summary: dict[str, Any]) -> str:
    audit = summary["audit"]
    horizon_rows = []
    for horizon, values in summary.get("horizons", {}).items():
        horizon_rows.append(
            f"| {horizon}s | {values['median_return_bps']:.2f} | {values['favorable_fraction']:.1%} | "
            f"{values['median_mfe_bps']:.2f} | {values['median_mae_bps']:.2f} | "
            f"{values['mfe_at_least_14bps_fraction']:.1%} |"
        )
    behavior_rows = []
    for label, values in sorted(summary.get("behavior_14bps", {}).items()):
        behavior_rows.append(
            f"| {label} | {values['signals']} | {values['median_return_10s_bps']:.2f} | "
            f"{values['median_return_30s_bps']:.2f} | {values['median_mfe_180s_bps']:.2f} | "
            f"{values['median_mae_180s_bps']:.2f} |"
        )
    barrier = summary.get("first_barriers", {}).get("14", {})
    return f"""# W13-P Descriptive Micro-Path Pulse

Generated: {datetime.now(UTC).isoformat()}

## Scope

This is a read-only descriptive pulse. It is not TRAIN, model fitting, threshold
discovery, economic validation, Shadow or Live authorization. The existing W13 sample
minimums and sealed holdout are unchanged.

## Sample

- Strict quality-v2 signals: **{audit.get('strict_signals', 0)}**.
- Fifteen-minute same-symbol dependence clusters: **{audit.get('independence_clusters_15m', 0)}**.
- Directions: `{json.dumps(audit.get('directions', {}), sort_keys=True)}`.
- Symbols: {', '.join(audit.get('symbols', []))}.
- Coverage: {audit.get('start_utc')} to {audit.get('end_utc')}.

## Early Path

Directional values are oriented to the proposed side; positive is favorable.

| Horizon | Median return bps | Favorable | Median MFE | Median MAE | MFE >= 14 bps |
| ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(horizon_rows)}

At the pre-existing 14-bps cost barrier: favorable-first={barrier.get('FAVORABLE_FIRST', 0)},
adverse-first={barrier.get('ADVERSE_FIRST', 0)}, neither={barrier.get('NEITHER', 0)}.

## Descriptive Outcome Groups

`GOOD_PROXY` means +14 bps occurred before -14 bps; `BAD_PROXY` is symmetric;
`INDETERMINATE` reached neither. These are path descriptions, not learned labels.

| Group | N | Median return 10s | Median return 30s | Median MFE 180s | Median MAE 180s |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(behavior_rows)}

## Interpretation Limits

- Every strict signal is SHORT; nothing here transfers to LONG.
- Signals cover only a few days and repeated same-symbol signals are correlated.
- Available MFE is not realized profit and does not prove an executable policy.
- Subgroup differences are hypotheses for the future preregistration, not filters.
- No existing holdout was opened and no production behavior was changed.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe strict W13-P micro-path captures")
    parser.add_argument("--collection-root", type=Path, default=Path("data/w13p_prospective_collection"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reports/governance/aegis_prospective_validation/live/"
            "signal_conditioned_micro_path_w13/prospective_collection/descriptive_pulse"
        ),
    )
    args = parser.parse_args()
    frame, audit = build_episode_table(args.collection_root)
    summary = summarize(frame, audit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "w13p_descriptive_episodes.csv", index=False)
    (args.output_dir / "w13p_descriptive_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "w13p_descriptive_report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
