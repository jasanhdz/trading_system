#!/usr/bin/env python3
"""Discover wave regimes on W1 TRAIN only and test once on internal validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from aegis.research.wave_regime_discovery_w5 import (
    benjamini_hochberg, correlation_cluster_id, economic_summary, stable_wave_id,
)


SYMBOLS = ("ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT", "SUIUSDT", "XRPUSDT")
BASE_FEATURES = (
    "volume_ratio_20", "volume_z_20", "body_ratio", "body_atr", "side_clv",
    "side_taker_imbalance", "velocity_atr_1", "velocity_atr_2", "acceleration_atr",
    "atr_fraction", "side_price_vs_ma25_atr", "side_ma25_slope_atr",
    "side_15m_return", "side_15m_ma25_slope_atr", "side_btc_15m_return_atr",
    "side_rsi_space", "side_extension_ma25_atr", "side_directional_persistence_3",
    "side_directional_persistence_6", "price_vs_ma_7_atr", "ma_7_slope_atr",
    "price_vs_ma_99_atr", "ma_99_slope_atr", "context_15m_atr_fraction",
    "context_15m_volume_ratio_20", "context_15m_taker_imbalance",
    "btc_5m_return_1", "btc_15m_return_1", "btc_return_aligned", "btc_correlation",
    "return_3", "higher_high", "higher_low", "lower_high", "lower_low",
)
DERIVED_FEATURES = (
    "directional_move_3_atr", "directional_rsi_extension", "trend_alignment_count",
    "trend_maturity_proxy", "moderate_volume", "extreme_volume",
    "structure_aligned", "structure_opposed", "btc_supports_position", "consolidation_proxy",
    "breakout_proxy", "hour_sin", "hour_cos", "day_sin", "day_cos", "weekend",
    "session_asia", "session_europe", "session_usa",
)
FEATURES = BASE_FEATURES + DERIVED_FEATURES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_population(root: Path, config: dict[str, Any]) -> pd.DataFrame:
    source = root / config["population"]["source"]
    discovery_start = pd.Timestamp(config["partitions"]["discovery"][0]).timestamp() * 1000
    validation_end = pd.Timestamp(config["partitions"]["development_validation"][1]).timestamp() * 1000
    future = [f"future_1m_{kind}_{minute}" for kind in ("high", "low", "close") for minute in range(1, 31)]
    columns = list(dict.fromkeys((
        "symbol", "side", "sample_source", "entry_variant", "event_timestamp_ms",
        "entry_price", "entry_atr", *BASE_FEATURES, *future,
    )))
    frames = []
    for symbol in SYMBOLS:
        frame = pd.read_parquet(
            source / f"{symbol}.parquet", columns=columns,
            filters=[("event_timestamp_ms", ">=", int(discovery_start)), ("event_timestamp_ms", "<", int(validation_end))],
        )
        frame = frame.loc[
            frame["sample_source"].eq(config["population"]["sample_source"])
            & frame["entry_variant"].eq(config["population"]["entry_variant"])
        ].copy()
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.drop_duplicates(["symbol", "side", "event_timestamp_ms"], inplace=True)
    return result


def add_outcomes(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    entry = result["entry_price"].to_numpy(float)
    atr = result["entry_atr"].to_numpy(float)
    direction = np.where(result["side"].eq("LONG"), 1.0, -1.0)
    highs = result[[f"future_1m_high_{i}" for i in range(1, 31)]].to_numpy(float)
    lows = result[[f"future_1m_low_{i}" for i in range(1, 31)]].to_numpy(float)
    closes = result[[f"future_1m_close_{i}" for i in range(1, 31)]].to_numpy(float)
    favorable = np.where(direction[:, None] > 0, highs / entry[:, None] - 1, 1 - lows / entry[:, None])
    adverse = np.where(direction[:, None] > 0, 1 - lows / entry[:, None], highs / entry[:, None] - 1)
    fav_level = float(config["outcome"]["favorable_barrier_atr"]) * atr / entry
    adv_level = float(config["outcome"]["adverse_barrier_atr"]) * atr / entry
    fav_hit, adv_hit = favorable >= fav_level[:, None], adverse >= adv_level[:, None]
    sentinel = favorable.shape[1]
    fav_first = np.where(fav_hit.any(1), fav_hit.argmax(1), sentinel)
    adv_first = np.where(adv_hit.any(1), adv_hit.argmax(1), sentinel)
    is_adv = (adv_first <= fav_first) & (adv_first < sentinel)
    is_fav = (fav_first < adv_first) & (fav_first < sentinel)
    exit_index = np.where(is_adv, adv_first, np.where(is_fav, fav_first, sentinel - 1))
    terminal = direction * (closes[:, -1] / entry - 1)
    gross = np.where(is_adv, -adv_level, np.where(is_fav, fav_level, terminal))
    active = np.arange(sentinel)[None, :] <= exit_index[:, None]
    mfe = np.where(active, favorable, -np.inf).max(1) * entry / atr
    mae = np.where(active, adverse, -np.inf).max(1) * entry / atr
    directional_close = direction[:, None] * (closes / entry[:, None] - 1)
    increments = np.diff(np.column_stack([np.zeros(len(result)), directional_close]), axis=1)
    path_length = np.where(active, np.abs(increments), 0).sum(1)
    endpoint = directional_close[np.arange(len(result)), exit_index]
    efficiency = np.abs(endpoint) / np.maximum(path_length, 1e-12)
    ratio = mfe / np.maximum(mae, 1e-9)
    net = gross * 10_000 - float(config["outcome"]["round_trip_cost_bps"])
    label = np.full(len(result), "NEUTRAL_WAVE", dtype=object)
    good = is_fav & (net > 0) & (mae <= 0.25) & (ratio >= 1.5) & (efficiency >= 0.25)
    bad = is_adv | (net <= -14)
    label[bad] = "BAD_WAVE"
    label[good] = "GOOD_WAVE"
    result["barrier_outcome"] = np.where(is_adv, "ADVERSE", np.where(is_fav, "FAVORABLE", "TIME"))
    result["exit_minute"] = exit_index + 1
    result["gross_return_bps"] = gross * 10_000
    result["net_return_bps"] = net
    result["mfe_atr"], result["mae_atr"] = mfe, mae
    result["mfe_mae_ratio"], result["path_efficiency"] = ratio, efficiency
    result["wave_label"] = label
    return result


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    direction = np.where(result["side"].eq("LONG"), 1.0, -1.0)
    atr_fraction = result["atr_fraction"].replace(0, np.nan)
    result["directional_move_3_atr"] = direction * result["return_3"] / atr_fraction
    rsi = 100 - result["side_rsi_space"]
    result["directional_rsi_extension"] = np.where(direction > 0, rsi, 100 - rsi)
    aligned = np.where(direction > 0, result["higher_high"] & result["higher_low"], result["lower_high"] & result["lower_low"])
    opposed = np.where(direction > 0, result["lower_high"] & result["lower_low"], result["higher_high"] & result["higher_low"])
    result["structure_aligned"], result["structure_opposed"] = aligned.astype(float), opposed.astype(float)
    result["trend_alignment_count"] = (
        (result["side_ma25_slope_atr"] > 0).astype(int)
        + (result["side_15m_ma25_slope_atr"] > 0).astype(int)
        + (result["side_directional_persistence_3"] > 0).astype(int)
        + (result["directional_move_3_atr"] > 0).astype(int)
    )
    result["trend_maturity_proxy"] = result["side_extension_ma25_atr"].abs() + result["directional_move_3_atr"].clip(lower=0)
    result["moderate_volume"] = result["volume_ratio_20"].between(1.5, 3.0).astype(float)
    result["extreme_volume"] = result["volume_ratio_20"].gt(4).astype(float)
    result["btc_supports_position"] = result["side_btc_15m_return_atr"].ge(0).astype(float)
    result["consolidation_proxy"] = (result["return_3"].abs() / atr_fraction).lt(0.5).astype(float)
    result["breakout_proxy"] = (result["structure_aligned"].gt(0) & result["body_atr"].gt(0.5)).astype(float)
    ts = pd.to_datetime(result["event_timestamp_ms"], unit="ms", utc=True)
    hour, weekday = ts.dt.hour.to_numpy(), ts.dt.weekday.to_numpy()
    result["hour_sin"], result["hour_cos"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    result["day_sin"], result["day_cos"] = np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7)
    result["weekend"] = (weekday >= 5).astype(float)
    result["session_asia"] = ((hour >= 0) & (hour < 8)).astype(float)
    result["session_europe"] = ((hour >= 8) & (hour < 16)).astype(float)
    result["session_usa"] = ((hour >= 16) & (hour < 24)).astype(float)
    result["correlation_cluster_id"] = result["event_timestamp_ms"].map(correlation_cluster_id)
    result["utc_day"] = ts.dt.strftime("%Y-%m-%d")
    result["wave_episode_id"] = [stable_wave_id(s, side, int(t)) for s, side, t in zip(result.symbol, result.side, result.event_timestamp_ms, strict=True)]
    return result


def pipeline(model: Any) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", model)])


def bootstrap_mean(frame: pd.DataFrame, repetitions: int, seed: int) -> tuple[list[float], float]:
    daily = frame.groupby("utc_day")["net_return_bps"].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(daily, size=(repetitions, len(daily)), replace=True).mean(1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])], float((samples <= 0).mean())


def candidate_metrics(frame: pd.DataFrame, config: dict[str, Any], seed: int) -> dict[str, Any]:
    summary = economic_summary(frame)
    ci, pvalue = bootstrap_mean(frame, int(config["validation"]["bootstrap_repetitions"]), seed)
    thirds = pd.qcut(frame["event_timestamp_ms"].rank(method="first"), 3, labels=False)
    summary.update({
        "expectancy_ci_95_bps": ci, "bootstrap_probability_expectancy_le_zero": pvalue,
        "stress_20bps_expectancy_bps": float((frame["net_return_bps"] - 6).mean()),
        "positive_symbols": int((frame.groupby("symbol")["net_return_bps"].mean() > 0).sum()),
        "positive_temporal_thirds": int((frame.assign(third=thirds).groupby("third")["net_return_bps"].mean() > 0).sum()),
    })
    return summary


def context_tables(discovery: pd.DataFrame, validation: pd.DataFrame) -> list[dict[str, Any]]:
    maturity_edges = [-np.inf, *discovery["trend_maturity_proxy"].quantile([0.25, 0.50, 0.75]).tolist(), np.inf]
    volatility_edges = [-np.inf, *discovery["atr_fraction"].quantile([0.25, 0.50, 0.75]).tolist(), np.inf]
    output: list[dict[str, Any]] = []
    for partition, source in (("DISCOVERY", discovery), ("VALIDATION", validation)):
        categories = {
            "trend_maturity_quartile": pd.cut(source["trend_maturity_proxy"], maturity_edges, labels=False, include_lowest=True),
            "volatility_quartile": pd.cut(source["atr_fraction"], volatility_edges, labels=False, include_lowest=True),
            "trend_alignment_count": source["trend_alignment_count"].astype(int),
            "btc_supports_position": source["btc_supports_position"].astype(int),
            "structure_aligned": source["structure_aligned"].astype(int),
            "breakout_proxy": source["breakout_proxy"].astype(int),
            "consolidation_proxy": source["consolidation_proxy"].astype(int),
            "weekend": source["weekend"].astype(int),
            "session": np.select(
                [source["session_asia"].eq(1), source["session_europe"].eq(1)],
                ["ASIA", "EUROPE"], default="USA",
            ),
            "extension_bucket": pd.cut(
                source["side_extension_ma25_atr"].abs(), [-np.inf, 1, 2, np.inf],
                labels=["LT_1_ATR", "1_TO_2_ATR", "GT_2_ATR"],
            ),
            "rsi_space_bucket": pd.cut(
                source["side_rsi_space"], [-np.inf, 20, 40, np.inf],
                labels=["EXTREME_LT20", "20_TO_40", "SPACE_GT40"],
            ),
        }
        for family, values in categories.items():
            for value, subset in source.assign(_category=values).groupby("_category", observed=True):
                output.append({"partition": partition, "family": family, "value": str(value), **economic_summary(subset)})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(); root = args.root.resolve()
    config_path = root / "config/experiments/aegis_wave_regime_discovery_w5.yaml"
    config = yaml.safe_load(config_path.read_text())
    frame = add_features(add_outcomes(load_population(root, config), config))
    discovery_end = int(pd.Timestamp(config["partitions"]["discovery"][1]).timestamp()*1000)
    purge = int(config["partitions"]["purge_minutes"])*60_000
    discovery = frame.loc[frame.event_timestamp_ms < discovery_end-purge].copy()
    validation = frame.loc[frame.event_timestamp_ms >= discovery_end+purge].copy()
    Xd, Xv = discovery[list(FEATURES)], validation[list(FEATURES)]
    y = discovery.wave_label.eq("GOOD_WAVE").astype(int)
    logistic = pipeline(LogisticRegression(C=0.25, max_iter=500, class_weight="balanced", random_state=519001)).fit(Xd, y)
    tree = pipeline(DecisionTreeClassifier(max_depth=3, min_samples_leaf=2000, class_weight="balanced", random_state=519001)).fit(Xd, y)
    cluster = pipeline(KMeans(n_clusters=6, n_init=20, random_state=519001)).fit(Xd)
    discovery["logistic_probability"] = logistic.predict_proba(Xd)[:,1]
    validation["logistic_probability"] = logistic.predict_proba(Xv)[:,1]
    discovery["tree_leaf"] = tree.named_steps["model"].apply(tree[:-1].transform(Xd))
    validation["tree_leaf"] = tree.named_steps["model"].apply(tree[:-1].transform(Xv))
    discovery["cluster"] = cluster.predict(Xd); validation["cluster"] = cluster.predict(Xv)
    candidates: list[dict[str, Any]] = []
    for quantile, name in ((0.90,"LOGISTIC_TOP_DECILE"),(0.80,"LOGISTIC_TOP_QUINTILE")):
        threshold = float(discovery.logistic_probability.quantile(quantile))
        subset = discovery.loc[discovery.logistic_probability >= threshold]
        candidates.append({"name":name,"kind":"probability","value":threshold,"discovery":economic_summary(subset)})
    for kind, column in (("tree_leaf","tree_leaf"),("cluster","cluster")):
        for value, subset in discovery.groupby(column):
            metrics = economic_summary(subset)
            if metrics["episodes"] >= int(config["discovery"]["minimum_discovery_episodes"]) and metrics["net_expectancy_bps"] > 0:
                candidates.append({"name":f"{kind.upper()}_{value}","kind":kind,"value":int(value),"discovery":metrics})
    candidates.sort(key=lambda x:(x["discovery"]["net_expectancy_bps"],x["discovery"]["episodes"]), reverse=True)
    candidates = candidates[:int(config["discovery"]["maximum_candidates_forwarded"])]
    pvalues: dict[str,float] = {}
    for index, candidate in enumerate(candidates):
        if candidate["kind"] == "probability": selected = validation.loc[validation.logistic_probability >= candidate["value"]]
        else: selected = validation.loc[validation[candidate["kind"]] == candidate["value"]]
        candidate["validation"] = candidate_metrics(selected, config, 519100+index)
        pvalues[candidate["name"]] = candidate["validation"]["bootstrap_probability_expectancy_le_zero"]
    fdr = benjamini_hochberg(pvalues) if pvalues else {}
    gate = config["gate"]
    passes=[]
    for candidate in candidates:
        m=candidate["validation"]; blockers=[]
        if m["episodes"] < gate["minimum_validation_episodes"]: blockers.append("EPISODES_LT_2000")
        if m["independent_clusters"] < gate["minimum_independent_clusters"]: blockers.append("CLUSTERS_LT_500")
        if m["net_expectancy_bps"] < gate["minimum_net_expectancy_bps"]: blockers.append("NET_EXPECTANCY_LT_2BPS")
        if m["expectancy_ci_95_bps"][0] <= 0: blockers.append("BOOTSTRAP_CI_CROSSES_ZERO")
        if m["stress_20bps_expectancy_bps"] <= 0: blockers.append("FAILS_20BPS_COST")
        if m["positive_symbols"] < gate["minimum_positive_symbols"]: blockers.append("POSITIVE_SYMBOLS_LT_7")
        if m["positive_temporal_thirds"] < gate["minimum_positive_temporal_thirds"]: blockers.append("TEMPORAL_THIRDS_LT_2")
        if m["maximum_symbol_share"] > gate["maximum_symbol_share"]: blockers.append("SYMBOL_SHARE_GT_20PCT")
        if not fdr.get(candidate["name"],False): blockers.append("FDR_NOT_SIGNIFICANT")
        candidate["gate_blockers"], candidate["gate_pass"] = blockers, not blockers
        if not blockers: passes.append(candidate["name"])
    mi = mutual_info_classif(SimpleImputer(strategy="median").fit_transform(Xd), y, random_state=519001)
    coefficients = logistic.named_steps["model"].coef_[0]
    feature_analysis = sorted([{"feature":f,"mutual_information":float(v),"logistic_coefficient":float(c)} for f,v,c in zip(FEATURES,mi,coefficients,strict=True)], key=lambda x:x["mutual_information"], reverse=True)
    label_profiles = {}
    discovery_scale = discovery[list(FEATURES)].std().replace(0, np.nan)
    for feature in FEATURES:
        medians = discovery.groupby("wave_label")[feature].median().to_dict()
        means = discovery.groupby("wave_label")[feature].mean().to_dict()
        label_profiles[feature] = {
            "median": {str(k): float(v) for k, v in medians.items()},
            "standardized_good_minus_bad_mean": float(
                (means.get("GOOD_WAVE", 0.0) - means.get("BAD_WAVE", 0.0))
                / discovery_scale[feature]
            ) if pd.notna(discovery_scale[feature]) else 0.0,
        }
    cluster_analysis=[]
    cluster_profile_features = (
        "atr_fraction", "volume_ratio_20", "volume_z_20", "body_atr",
        "side_taker_imbalance", "directional_move_3_atr", "side_extension_ma25_atr",
        "side_rsi_space", "trend_alignment_count", "trend_maturity_proxy",
        "btc_supports_position", "consolidation_proxy", "breakout_proxy",
    )
    for cid in range(6):
        discovery_cluster = discovery.loc[discovery.cluster==cid]
        cluster_analysis.append({
            "cluster":cid,
            "profile_discovery_median": {
                feature: float(discovery_cluster[feature].median())
                for feature in cluster_profile_features
            },
            "discovery":economic_summary(discovery_cluster),
            "validation":economic_summary(validation.loc[validation.cluster==cid]),
        })
    volume_bins=pd.cut(frame.volume_ratio_20,[1.25,1.5,2,3,4,np.inf],right=False)
    volume_analysis=[]
    for (part,side,bin_),sub in frame.assign(volume_bin=volume_bins).groupby([np.where(frame.event_timestamp_ms<discovery_end,"DISCOVERY","VALIDATION"),"side","volume_bin"],observed=True):
        volume_analysis.append({"partition":part,"side":side,"bin":str(bin_),**economic_summary(sub)})
    report={
        "schema_version":"aegis-wave-regime-discovery-w5-evaluation-v1",
        "status":"AEGIS_WAVE_REGIME_W5_EDGE_FOUND" if passes else "AEGIS_WAVE_REGIME_W5_NO_ROBUST_ECONOMIC_REGIME",
        "config_sha256":sha256(config_path), "source_manifest_sha256":sha256(Path(config["population"]["source"])/"manifest.json" if Path(config["population"]["source"]).is_absolute() else root/Path(config["population"]["source"])/"manifest.json"),
        "population":{"total":len(frame),"discovery":economic_summary(discovery),"validation":economic_summary(validation),"labels_discovery":discovery.wave_label.value_counts().to_dict(),"labels_validation":validation.wave_label.value_counts().to_dict()},
        "feature_analysis":feature_analysis, "label_profiles":label_profiles,
        "model_diagnostics": {
            "logistic_validation_roc_auc": float(roc_auc_score(validation.wave_label.eq("GOOD_WAVE"), validation.logistic_probability)),
            "logistic_validation_average_precision": float(average_precision_score(validation.wave_label.eq("GOOD_WAVE"), validation.logistic_probability)),
            "validation_good_base_rate": float(validation.wave_label.eq("GOOD_WAVE").mean()),
        },
        "tree_rules_scaled_space":export_text(tree.named_steps["model"],feature_names=list(FEATURES)),
        "clusters":cluster_analysis,"volume_bins":volume_analysis,
        "context_tables":context_tables(discovery, validation),
        "per_symbol_side": [
            {"partition": partition, "symbol": symbol, "side": side, **economic_summary(subset)}
            for partition, source in (("DISCOVERY", discovery), ("VALIDATION", validation))
            for (symbol, side), subset in source.groupby(["symbol", "side"])
        ],
        "candidates":candidates,"fdr":fdr,"passing_regimes":passes,
        "holdouts":{"w1_validation":"UNREAD","w1_holdout":"SEALED_UNREAD","w2_holdout":"SEALED_UNREAD","w3_holdout":"SEALED_UNREAD"},
        "flags":{"W5_REGIME_DIFFERENTIATION_FOUND":bool(feature_analysis and feature_analysis[0]["mutual_information"]>0.001),"W5_ECONOMIC_REGIME_EDGE_FOUND":bool(passes),"W5_READY_FOR_FUTURE_CONFIRMATION":bool(passes),"W5_READY_FOR_SHADOW":False,"W5_READY_FOR_LIVE":False},
        "safety":{"authenticated_requests":0,"exchange_mutations":0,"production_changes":0},
    }
    out=root/"data/wave_regime_discovery_w5"; out.mkdir(parents=True,exist_ok=True)
    path=out/"evaluation_01.json"; path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); os.chmod(path,0o600)
    frame[["wave_episode_id","symbol","side","event_timestamp_ms","wave_label","net_return_bps","mfe_atr","mae_atr","path_efficiency","correlation_cluster_id"]].to_parquet(out/"episodes_01.parquet",index=False)
    print(json.dumps({"status":report["status"],"output":str(path),"passing_regimes":passes,"population":report["population"]},indent=2))


if __name__ == "__main__":
    main()
