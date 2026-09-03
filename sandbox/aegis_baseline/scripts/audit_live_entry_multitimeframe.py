#!/usr/bin/env python3
"""Map good/bad Live entries against causal 1m/5m/15m/1h context."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from aegis.research.live_entry_multitimeframe import (
    add_directional_context,
    aggregate_klines,
    attach_features,
    feature_comparison,
    indicator_frame,
)
from aegis.research.live_entry_quality_audit import fit_bad_entry_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_live_entry_quality_audit_20260815.yaml"))
    parser.add_argument("--entry-csv", type=Path, default=Path("reports/governance/aegis_prospective_validation/live/live_entry_quality_audit_20260815/live_entry_classification.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/governance/aegis_prospective_validation/live/live_entry_quality_audit_20260815/multitimeframe"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mtf = config["multitimeframe"]
    entries = pd.read_csv(args.entry_csv)
    candles = {
        symbol: pd.read_parquet(Path(mtf["source_dir"]) / f"{symbol}_1m.parquet")
        for symbol in sorted(entries["symbol"].unique())
    }
    enriched = add_directional_context(
        attach_features(entries, candles, mtf["timeframes_minutes"]), mtf["timeframes_minutes"]
    )
    features = [column for column in enriched if column.startswith("tf")]
    comparison = feature_comparison(enriched, features)
    directional_features = [column for column in enriched if column.startswith("dir")]
    directional_discovery = feature_comparison(enriched, directional_features, "DISCOVERY")
    directional_validation = feature_comparison(enriched, directional_features, "VALIDATION")
    directional_stability = directional_discovery.merge(
        directional_validation, on="feature", suffixes=("_discovery", "_validation"), validate="one_to_one"
    )
    directional_stability["same_sign"] = (
        directional_stability["standardized_median_difference_bad_minus_good_discovery"]
        * directional_stability["standardized_median_difference_bad_minus_good_validation"] > 0
    )
    directional_stability["minimum_absolute_effect"] = directional_stability[[
        "standardized_median_difference_bad_minus_good_discovery",
        "standardized_median_difference_bad_minus_good_validation",
    ]].abs().min(axis=1)
    directional_stability = directional_stability.sort_values(
        ["same_sign", "minimum_absolute_effect"], ascending=[False, False]
    )
    model_config = {
        **mtf["model"], "features": features, "validation_gate": config["validation_gate"]
    }
    model = fit_bad_entry_model(enriched, model_config)
    verdict = {
        "schema_version": "aegis-live-entry-multitimeframe-audit-v1",
        "episodes": len(enriched), "feature_count": len(features),
        "timeframes": [f"{value}m" for value in mtf["timeframes_minutes"]],
        "model_evidence": asdict(model),
        "top_discovery_differences": comparison.head(20).to_dict(orient="records"),
        "stable_directional_differences": directional_stability.loc[
            directional_stability["same_sign"]
        ].head(20).to_dict(orient="records"),
        "MULTITIMEFRAME_BAD_ENTRY_FILTER_FOUND": bool(model.gate_passed),
        "PRODUCTION_CHANGE_JUSTIFIED": False,
        "limitations": [
            "Full-trade MFE/MAE do not preserve excursion ordering.",
            "Manual resizing makes monetary PnL secondary.",
            "A retrospective filter must pass frozen prospective observation before enforcement.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.out_dir / "live_entry_multitimeframe_features.csv", index=False)
    comparison.to_csv(args.out_dir / "good_vs_bad_feature_comparison.csv", index=False)
    directional_stability.to_csv(args.out_dir / "directional_feature_stability.csv", index=False)
    (args.out_dir / "multitimeframe_verdict.json").write_text(json.dumps(_safe(verdict), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _charts(enriched, candles, args.out_dir / "charts", mtf)
    (args.out_dir / "multitimeframe_report.md").write_text(_report(verdict), encoding="utf-8")
    print(json.dumps(_safe(verdict), indent=2, sort_keys=True, allow_nan=False))


def _safe(value):
    if isinstance(value, dict): return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def _charts(entries: pd.DataFrame, candles: dict[str, pd.DataFrame], out_dir: Path, config: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bad = entries.loc[entries["entry_class"].eq("BAD_ENTRY")].nlargest(int(config["chart_bad_entries"]), "mae_bps_underlying")
    good = entries.loc[entries["entry_class"].eq("GOOD_CLEAN_ENTRY")].nsmallest(int(config["chart_good_entries"]), "mae_bps_underlying")
    timeframes = [int(value) for value in config["timeframes_minutes"]]
    selected = pd.concat([bad, good])
    prepared = {
        symbol: {
            minutes: (aggregate_klines(candles[symbol], minutes), indicator_frame(candles[symbol], minutes))
            for minutes in timeframes
        }
        for symbol in selected["symbol"].unique()
    }
    for _, entry in selected.iterrows():
        symbol = str(entry["symbol"])
        timestamp = pd.Timestamp(entry["entry_timestamp"])
        start = timestamp - pd.Timedelta(minutes=int(config["lookback_chart_minutes"]))
        end = timestamp + pd.Timedelta(minutes=int(config["forward_chart_minutes"]))
        fig, axes = plt.subplots(len(timeframes), 1, figsize=(14, 10), sharex=True)
        for axis, minutes in zip(axes, timeframes):
            bars = prepared[symbol][minutes][0]
            bars = bars.loc[bars["open_time"].between(start, end)].copy()
            indicators = prepared[symbol][minutes][1]
            indicators = indicators.loc[indicators["close_time"].between(start, end)].copy()
            _draw_candles(axis, bars, minutes)
            ema25 = bars["close"].ewm(span=25, adjust=False).mean()
            axis.plot(bars["open_time"], ema25, color="#5e35b1", linewidth=0.9, label="EMA25")
            axis.axvline(timestamp, color="#d32f2f", linewidth=1.2)
            prefix = f"tf{minutes}m__"
            latest = indicators.loc[indicators["close_time"].le(timestamp)].tail(1)
            if not latest.empty:
                row = latest.iloc[0]
                summary = (
                    f"RSI12 {row[prefix + 'rsi12']:.1f} | "
                    f"ATR {row[prefix + 'atr_pct_bps']:.1f} bps | "
                    f"Vol {row[prefix + 'volume_ratio20']:.2f}x"
                )
                axis.text(0.995, 0.94, summary, transform=axis.transAxes, ha="right", va="top", fontsize=8,
                          bbox={"facecolor": "white", "edgecolor": "#b0bec5", "alpha": 0.82})
            axis.set_ylabel(f"{minutes}m")
            axis.grid(alpha=0.15)
        axes[0].set_title(
            f"{entry['entry_class']} | {entry['symbol']} {entry['side']} | "
            f"MAE {entry['mae_bps_underlying']:.1f} bps | MFE {entry['mfe_bps_underlying']:.1f} bps"
        )
        axes[-1].set_xlabel("UTC (red line = entry)")
        fig.tight_layout()
        name = f"{entry['opened_at'][:10]}_{entry['symbol']}_{entry['side']}_{entry['trade_id_hash'][:10]}.png"
        fig.savefig(out_dir / name, dpi=120); plt.close(fig)


def _draw_candles(axis, bars: pd.DataFrame, minutes: int) -> None:
    if bars.empty:
        return
    width_days = minutes / (24.0 * 60.0) * 0.72
    colors = bars["close"].ge(bars["open"]).map({True: "#00897b", False: "#e53935"})
    axis.vlines(bars["open_time"], bars["low"], bars["high"], color=colors, linewidth=0.7)
    bottoms = bars[["open", "close"]].min(axis=1)
    heights = (bars["close"] - bars["open"]).abs()
    minimum_body = max(float(bars["close"].median()) * 1e-7, 1e-12)
    axis.bar(
        bars["open_time"], heights.clip(lower=minimum_body), bottom=bottoms,
        width=width_days, color=colors, edgecolor=colors, linewidth=0.4,
    )


def _report(verdict: dict) -> str:
    validation = verdict["model_evidence"]["validation"]
    ci = verdict["model_evidence"]["bootstrap_bad_rate_reduction_ci95"]
    rows = "\n".join(
        f"| `{item['feature']}` | {item['good_median']:.4f} | {item['bad_median']:.4f} | {item['standardized_median_difference_bad_minus_good']:+.3f} |"
        for item in verdict["top_discovery_differences"][:15]
    )
    stable_rows = "\n".join(
        f"| `{item['feature']}` | {item['standardized_median_difference_bad_minus_good_discovery']:+.3f} | {item['standardized_median_difference_bad_minus_good_validation']:+.3f} |"
        for item in verdict["stable_directional_differences"][:15]
    )
    return f"""# Aegis Live Entry Multi-Timeframe Audit

## Verdict

- Episodes: {verdict['episodes']}; causal features: {verdict['feature_count']}.
- Context: `1m`, `5m`, `15m`, `1h`, closed bars only.
- Validation allowed/total: {validation['allowed']}/{validation['episodes']}.
- Validation bad-rate reduction: {validation['relative_bad_rate_reduction']:.1%}.
- Clean-good retention: {validation['good_retention_rate']:.1%}.
- Bootstrap 95% CI: [{ci[0]:.1%}, {ci[1]:.1%}].
- `MULTITIMEFRAME_BAD_ENTRY_FILTER_FOUND = {str(verdict['MULTITIMEFRAME_BAD_ENTRY_FILTER_FOUND']).upper()}`.
- `PRODUCTION_CHANGE_JUSTIFIED = FALSE`.

## Discovery differences

| Feature | Good median | Bad median | Standardized difference |
|---|---:|---:|---:|
{rows}

These are descriptive TRAIN differences. Only the frozen validation result determines whether they can support a filter.

## Direction-aware descriptive stability

| Feature | Discovery effect | Validation effect |
|---|---:|---:|
{stable_rows}

Positive values mean BAD entries had a larger value than clean GOOD entries after orienting the feature to the trade side. This table is descriptive and was produced after the primary model failed; it cannot promote a guard.
"""


if __name__ == "__main__":
    main()
