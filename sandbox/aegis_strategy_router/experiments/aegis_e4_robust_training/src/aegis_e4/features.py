"""Vectorized causal E4 feature construction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from aegis.research.live_entry_multitimeframe import aggregate_klines, indicator_frame


BASE_RAW = (
    "return_1_bps", "return_3_bps", "return_6_bps", "atr_pct_bps",
    "atr_percentile_96", "rsi6", "rsi12", "rsi24", "ema7_extension_atr",
    "ema25_extension_atr", "ema99_extension_atr", "ema7_slope_atr",
    "ema25_slope_atr", "trend_age", "prior_move_6_atr", "volume_ratio20",
    "volume_z50", "body_ratio", "clv", "distance_recent_high_atr",
    "distance_recent_low_atr", "range_48_atr", "path_efficiency_6",
    "breakout_up", "breakout_down",
)
SIGNED_RAW = {
    "return_1_bps", "return_3_bps", "return_6_bps", "ema7_extension_atr",
    "ema25_extension_atr", "ema99_extension_atr", "ema7_slope_atr",
    "ema25_slope_atr", "trend_age", "prior_move_6_atr",
}


def build_neutral_symbol_panel(
    one_minute: pd.DataFrame, anchors: pd.DatetimeIndex, timeframes: list[int]
) -> tuple[pd.DataFrame, dict[str, str]]:
    panel = pd.DataFrame({"decision_at": anchors}).sort_values("decision_at")
    panel["decision_at"] = pd.to_datetime(panel["decision_at"], utc=True).astype("datetime64[ns, UTC]")
    families: dict[str, str] = {}
    for minutes in timeframes:
        indicators = indicator_frame(one_minute, minutes).sort_values("close_time")
        indicators["close_time"] = pd.to_datetime(indicators["close_time"], utc=True).astype("datetime64[ns, UTC]")
        panel = pd.merge_asof(
            panel, indicators, left_on="decision_at", right_on="close_time",
            direction="backward", allow_exact_matches=True,
        )
        availability = f"available_at__tf{minutes}m"
        panel = panel.rename(columns={"close_time": availability})
        for raw in BASE_RAW:
            name = f"tf{minutes}m__{raw}"
            if name in panel:
                families[name] = "BASE"
    flow = flow_features(one_minute)
    flow["close_time"] = pd.to_datetime(flow["close_time"], utc=True).astype("datetime64[ns, UTC]")
    panel = pd.merge_asof(
        panel.sort_values("decision_at"), flow.sort_values("close_time"),
        left_on="decision_at", right_on="close_time", direction="backward",
        allow_exact_matches=True,
    ).rename(columns={"close_time": "available_at__flow"})
    for name in flow.columns:
        if name != "close_time":
            families[name] = "FLOW"
    return panel, families


def flow_features(one_minute: pd.DataFrame) -> pd.DataFrame:
    bars = aggregate_klines(one_minute, 5)
    volume = bars.volume.astype(float)
    buys = bars.taker_buy_volume.astype(float)
    sells = (volume - buys).clip(lower=0.0)
    imbalance = (buys - sells) / volume.replace(0.0, np.nan)
    ret = bars.close.pct_change() * 10_000.0
    signed_effect = np.sign(imbalance) * ret
    output = pd.DataFrame({"close_time": bars.close_time})
    output["flow__buy_fraction"] = buys / volume.replace(0.0, np.nan)
    output["flow__sell_fraction"] = sells / volume.replace(0.0, np.nan)
    output["flow__signed_imbalance"] = imbalance
    output["flow__imbalance_slope_3"] = imbalance - imbalance.shift(3)
    output["flow__imbalance_acceleration"] = imbalance.diff() - imbalance.diff().shift(1)
    output["flow__imbalance_persistence_6"] = np.sign(imbalance).rolling(6, min_periods=3).mean()
    output["flow__signed_price_effectiveness"] = signed_effect
    output["flow__price_bps_per_abs_imbalance"] = ret / (imbalance.abs() + 1.0e-3)
    output["flow__without_price_response"] = imbalance.abs() / (ret.abs() + 1.0)
    output["flow__impact_slope_3"] = signed_effect - signed_effect.shift(3)
    output["flow__impact_acceleration"] = signed_effect.diff() - signed_effect.diff().shift(1)
    output["flow__impact_decay_3"] = signed_effect - signed_effect.shift(1).rolling(3, min_periods=2).mean()
    output["flow__volume_log"] = np.log1p(volume)
    output["flow__available"] = 1.0
    output["flow__source_stale"] = 0.0
    output["flow__source_gap"] = 0.0
    return output


def add_cross_market(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    result = panel.copy().sort_values(["decision_at", "symbol"], kind="mergesort")
    source = [f"tf5m__return_{bars}_bps" for bars in (1, 3, 6)]
    refs = result.loc[result.symbol.isin(["BTCUSDT", "ETHUSDT"]), ["decision_at", "symbol", *source]]
    wide = refs.pivot(index="decision_at", columns="symbol", values=source)
    wide.columns = [f"cross__{symbol.lower()}__{name}" for name, symbol in wide.columns]
    result = result.merge(wide.reset_index(), on="decision_at", how="left", validate="many_to_one")
    market = result.groupby("decision_at", sort=False)[source[0]].agg(
        cross__market_mean_return="mean", cross__dispersion="std",
        cross__breadth_positive=lambda values: float(np.mean(values > 0)),
    ).reset_index()
    result = result.merge(market, on="decision_at", how="left", validate="many_to_one")
    result["cross__btc_eth_agreement"] = np.sign(result["cross__btcusdt__tf5m__return_1_bps"]) * np.sign(result["cross__ethusdt__tf5m__return_1_bps"])
    result["cross__relative_vs_btc"] = result[source[0]] - result["cross__btcusdt__tf5m__return_1_bps"]
    result["cross__propagation_lag_btc"] = result["cross__btcusdt__tf5m__return_3_bps"] - result[source[1]]
    result["cross__breadth_acceleration"] = result.groupby("symbol", sort=False).cross__breadth_positive.diff(3).fillna(0.0)
    result["cross__btc_impulse_acceleration"] = result.groupby("symbol", sort=False)["cross__btcusdt__tf5m__return_1_bps"].diff(2).fillna(0.0)
    families = {name: "CROSS_MARKET" for name in result.columns if name.startswith("cross__")}
    return result, families


def orient_sides(panel: pd.DataFrame, families: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
    pieces = []
    output_families: dict[str, str] = {}
    for side, direction in (("LONG", 1.0), ("SHORT", -1.0)):
        data: dict[str, Any] = {
            "symbol": panel.symbol.to_numpy(), "decision_at": panel.decision_at.to_numpy(),
            "side": np.repeat(side, len(panel)), "feature__quality__side_sign": np.repeat(direction, len(panel)),
        }
        for column in [name for name in panel.columns if name.startswith("available_at__")]:
            data[column] = panel[column].to_numpy()
        output_families["feature__quality__side_sign"] = "QUALITY"
        for minutes in (5, 15, 60, 240):
            prefix = f"tf{minutes}m__"
            for raw in BASE_RAW:
                source = prefix + raw
                if source not in panel:
                    continue
                if raw in SIGNED_RAW:
                    name = f"feature__base__tf{minutes}m__directional_{raw}"
                    data[name] = direction * panel[source].to_numpy(float)
                elif raw.startswith("rsi"):
                    name = f"feature__base__tf{minutes}m__directional_{raw}_extension"
                    data[name] = direction * (panel[source].to_numpy(float) - 50.0)
                    room = f"feature__remaining__tf{minutes}m__rsi_room"
                    data[room] = np.where(direction > 0, 100.0 - panel[source].to_numpy(float), panel[source].to_numpy(float))
                    output_families[room] = "REMAINING_MOVE"
                elif raw == "clv":
                    name = f"feature__base__tf{minutes}m__directional_clv"
                    raw_values = panel[source].to_numpy(float)
                    data[name] = direction * (2.0 * np.nan_to_num(raw_values, nan=0.5) - 1.0)
                elif raw == "distance_recent_high_atr":
                    name = f"feature__remaining__tf{minutes}m__favorable_space_atr"
                    other = prefix + "distance_recent_low_atr"
                    data[name] = panel[source].to_numpy(float) if direction > 0 else panel[other].to_numpy(float)
                    output_families[name] = "REMAINING_MOVE"
                    continue
                elif raw == "distance_recent_low_atr":
                    name = f"feature__remaining__tf{minutes}m__adverse_space_atr"
                    other = prefix + "distance_recent_high_atr"
                    data[name] = panel[source].to_numpy(float) if direction < 0 else panel[other].to_numpy(float)
                    output_families[name] = "REMAINING_MOVE"
                    continue
                elif raw in {"breakout_up", "breakout_down"}:
                    if raw == "breakout_down":
                        continue
                    name = f"feature__base__tf{minutes}m__aligned_breakout"
                    data[name] = panel[prefix + ("breakout_up" if direction > 0 else "breakout_down")].to_numpy(float)
                else:
                    name = f"feature__base__tf{minutes}m__{raw}"
                    raw_values = panel[source].to_numpy(float)
                    if raw in {"body_ratio", "path_efficiency_6"}:
                        raw_values = np.nan_to_num(raw_values, nan=0.0)
                    data[name] = raw_values
                output_families[name] = "BASE"
            undefined = panel[prefix + "body_ratio"].isna() | panel[prefix + "clv"].isna()
            quality_name = f"feature__quality__tf{minutes}m__zero_range_candle"
            data[quality_name] = undefined.astype(float).to_numpy()
            output_families[quality_name] = "QUALITY"
            path_name = f"feature__quality__tf{minutes}m__zero_path"
            data[path_name] = panel[prefix + "path_efficiency_6"].isna().astype(float).to_numpy()
            output_families[path_name] = "QUALITY"
        for source in [name for name in panel.columns if name.startswith("flow__")]:
            name = "feature__flow__" + source.removeprefix("flow__")
            value = panel[source].to_numpy(float)
            if source in {
                "flow__signed_imbalance", "flow__imbalance_slope_3",
                "flow__imbalance_acceleration", "flow__imbalance_persistence_6",
                "flow__price_bps_per_abs_imbalance",
            }:
                value = direction * value
            data[name] = value
            output_families[name] = "FLOW"
        for source in [name for name in panel.columns if name.startswith("cross__")]:
            name = "feature__cross__" + source.removeprefix("cross__")
            value = panel[source].to_numpy(float)
            if source not in {"cross__dispersion", "cross__btc_eth_agreement"}:
                if source == "cross__breadth_positive":
                    value = panel[source].to_numpy(float) if direction > 0 else 1.0 - panel[source].to_numpy(float)
                else:
                    value = direction * panel[source].to_numpy(float)
            data[name] = value
            output_families[name] = "CROSS_MARKET"
        prior_move = direction * panel["tf5m__prior_move_6_atr"].to_numpy(float)
        trend_age = direction * panel["tf5m__trend_age"].to_numpy(float)
        data["feature__remaining__consumed_move_atr"] = np.maximum(0.0, prior_move)
        data["feature__remaining__impulse_age_bars"] = np.maximum(0.0, trend_age)
        data["feature__remaining__extension_atr"] = np.maximum(0.0, direction * panel["tf5m__ema25_extension_atr"].to_numpy(float))
        data["feature__remaining__momentum_decay"] = direction * panel["tf5m__return_1_bps"].to_numpy(float) - direction * panel["tf5m__return_3_bps"].to_numpy(float) / 3.0
        data["feature__quality__warmup_complete"] = np.ones(len(panel))
        data["feature__quality__l2_available"] = np.zeros(len(panel))
        data["feature__quality__oi_available"] = np.zeros(len(panel))
        for name in (
            "feature__remaining__consumed_move_atr", "feature__remaining__impulse_age_bars",
            "feature__remaining__extension_atr", "feature__remaining__momentum_decay",
        ):
            output_families[name] = "REMAINING_MOVE"
        for name in ("feature__quality__warmup_complete", "feature__quality__l2_available", "feature__quality__oi_available"):
            output_families[name] = "QUALITY"
        pieces.append(pd.DataFrame(data))
    result = pd.concat(pieces, ignore_index=True).sort_values(["decision_at", "symbol", "side"], kind="mergesort")
    return result, output_families


def assert_causal_availability(frame: pd.DataFrame) -> None:
    decision = pd.to_datetime(frame.decision_at, utc=True)
    for name in [column for column in frame.columns if column.startswith("available_at__")]:
        available = pd.to_datetime(frame[name], utc=True)
        if available.isna().any() or (available > decision).any():
            raise ValueError(f"NON_CAUSAL_AVAILABILITY:{name}")
    features = frame.filter(like="feature__")
    required = [name for name in features if name.startswith("feature__base__")]
    if features[required].isna().any().any():
        missing = features[required].isna().sum()
        detail = missing.loc[missing.gt(0)].sort_values(ascending=False).head(10).to_dict()
        raise ValueError(f"REQUIRED_BASE_FEATURE_MISSING:{detail}")
    if not np.isfinite(features.to_numpy(float)).all():
        raise ValueError("NON_FINITE_FEATURE")
