"""End-to-end cached, chronological W12 experiment orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score

from .data import SANDBOX, add_cross_market_features, build_panels, canonical_hash, feature_columns, load_config, sha256_file, verify_source
from .modeling import FrozenCandidate, fit_direct_candidates, ranked_metrics, select_candidate, temporal_negative_mask


MODEL_CONTEXT: dict[str, Any] = {}


@dataclass
class ExperimentResult:
    summary: dict[str, Any]
    feature_analysis: pd.DataFrame
    label_analysis: pd.DataFrame
    candidate_metrics: pd.DataFrame
    prospective_predictions: pd.DataFrame
    prospective_trades: pd.DataFrame
    economic_metrics: pd.DataFrame
    negative_controls: pd.DataFrame
    stability: pd.DataFrame
    selected_candidate: FrozenCandidate
    cache_manifest: dict[str, Any]


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="ascii")


def _checkpoint(stage: str, payload: Mapping[str, Any]) -> None:
    _json(SANDBOX / "artifacts" / "checkpoints" / f"{stage.lower()}.json", {
        "stage": stage, "status": "COMPLETE", **dict(payload),
    })


def _cache_key(config: Mapping[str, Any], authority: Mapping[str, Any]) -> str:
    return canonical_hash({
        "config": config,
        "manifest": authority["manifest_sha256"],
        "sources": {symbol: item["sha256"] for symbol, item in authority["symbols"].items()},
        "feature_version": config["features"]["version"],
    })


def load_or_build_cache(config: Mapping[str, Any], repository: Path, workers: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    authority = verify_source(config, repository)
    key = _cache_key(config, authority)
    cache = SANDBOX / "artifacts" / "cache"
    manifest_path = cache / "cache_manifest.json"
    features_path = cache / "base_features.parquet"
    labels_path = cache / "teacher_labels.parquet"
    if manifest_path.is_file() and features_path.is_file() and labels_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        if (
            manifest.get("cache_key") == key
            and manifest.get("features_sha256") == sha256_file(features_path)
            and manifest.get("labels_sha256") == sha256_file(labels_path)
        ):
            features = pd.read_parquet(features_path)
            labels = pd.read_parquet(labels_path)
            _checkpoint("FEATURE_GENERATION", {"cache": "HIT", "rows": len(features), "sha256": manifest["features_sha256"]})
            _checkpoint("LABEL_GENERATION", {"cache": "HIT", "rows": len(labels), "sha256": manifest["labels_sha256"]})
            return features, labels, manifest
    features, labels, schema = build_panels(config, repository, workers)
    cache.mkdir(parents=True, exist_ok=True)
    features.to_parquet(features_path, index=False, compression="zstd")
    labels.to_parquet(labels_path, index=False, compression="zstd")
    manifest = {
        "schema_version": "aegis-w12-cache-v1", "cache_key": key,
        "config_sha256": sha256_file(SANDBOX / "config" / "w12_frozen.json"),
        "input_manifest_sha256": authority["manifest_sha256"],
        "feature_version": config["features"]["version"],
        "feature_schema_sha256": schema["feature_schema_sha256"],
        "feature_names": schema["feature_names"],
        "features_rows": len(features), "labels_rows": len(labels),
        "features_sha256": sha256_file(features_path), "labels_sha256": sha256_file(labels_path),
        "source_sha256": {symbol: item["sha256"] for symbol, item in authority["symbols"].items()},
    }
    _json(manifest_path, manifest)
    _checkpoint("FEATURE_GENERATION", {"cache": "MISS", "rows": len(features), "sha256": manifest["features_sha256"]})
    _checkpoint("LABEL_GENERATION", {"cache": "MISS", "rows": len(labels), "sha256": manifest["labels_sha256"]})
    return features, labels, manifest


def attach_features(labels: pd.DataFrame, features: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    metadata = ["decision_at", "symbol", "feature_available_at", *names]
    merged = labels.merge(features[metadata], on=["decision_at", "symbol"], how="inner", validate="many_to_one")
    if not merged["feature_available_at"].le(merged["decision_at"]).all():
        raise AssertionError("feature availability exceeds decision time")
    if merged[names].columns.str.contains("future|label|teacher|mfe|mae|barrier|quality|ideal|gross|net|outcome", case=False, regex=True).any():
        raise AssertionError("label-derived column entered model matrix")
    return merged.sort_values(["decision_at", "symbol", "horizon_minutes", "side"], kind="mergesort", ignore_index=True)


def partition_frame(frame: pd.DataFrame, config: Mapping[str, Any], name: str) -> pd.DataFrame:
    start, end = (pd.Timestamp(value) for value in config["partitions"][name])
    purge = pd.Timedelta(minutes=int(config["partitions"]["purge_minutes"]))
    return frame[frame["decision_at"].ge(start + purge) & frame["outcome_available_at"].lt(end - purge)].copy()


def label_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    teachers = [f"teacher_{key}_good" for key in "abcde"]
    for (partition, side, horizon), group in frame.groupby(["partition", "side", "horizon_minutes"], sort=True):
        record = {
            "partition": partition, "side": side, "horizon_minutes": horizon,
            "rows": len(group), "zones": int(group["zone_best"].sum()),
            "majority_rate": float(group["majority_ideal"].mean()),
            "strict_rate": float(group["strict_ideal"].mean()),
            "weighted_rate": float(group["weighted_ideal"].mean()),
            "quality_mean": float(group["entry_quality_score"].mean()),
        }
        for teacher in teachers:
            record[f"{teacher}_rate"] = float(group[teacher].mean())
        teacher_values = group[teachers].to_numpy(bool)
        agreement = [
            float((teacher_values[:, left] == teacher_values[:, right]).mean())
            for left in range(len(teachers)) for right in range(left + 1, len(teachers))
        ]
        record["mean_pairwise_teacher_agreement"] = float(np.mean(agreement))
        rows.append(record)
    return pd.DataFrame(rows)


def discovery_feature_analysis(discovery: pd.DataFrame, names: list[str], seed: int) -> pd.DataFrame:
    base = discovery.drop_duplicates(["decision_at", "symbol", "side", "horizon_minutes"]).copy()
    interactions = {
        "interaction_momentum_x_flow": base["return_15m_bps"] * base["taker_imbalance_15m"],
        "interaction_compression_x_breakout": base["compression_ratio"] * (base["breakout_up_60m"] - base["breakout_down_60m"]),
        "interaction_relative_strength_x_breadth": base["relative_to_btc_15m_bps"] * (base["market_breadth_15m"] - 0.5),
    }
    for name, values in interactions.items():
        base[name] = values
    analysis_names = [*names, *interactions]
    target = base["majority_ideal"].astype(int).to_numpy()
    if len(base) > 150_000:
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(len(base), 150_000, replace=False))
        sample = base.iloc[positions]
        sample_target = target[positions]
    else:
        sample, sample_target = base, target
    matrix = SimpleImputer(strategy="median").fit_transform(sample[analysis_names])
    mi = mutual_info_classif(matrix, sample_target, random_state=seed, discrete_features=False)
    rows = []
    for index, name in enumerate(analysis_names):
        ideal = pd.to_numeric(base.loc[base["majority_ideal"], name], errors="coerce").dropna()
        normal = pd.to_numeric(base.loc[~base["majority_ideal"], name], errors="coerce").dropna()
        pooled = pd.concat((ideal, normal))
        scale = float(pooled.std(ddof=0))
        standardized = (float(ideal.median()) - float(normal.median())) / scale if scale > 0 and len(ideal) and len(normal) else None
        diagnostic = pd.DataFrame({"value": matrix[:, index], "ideal": sample_target})
        ranks = diagnostic["value"].rank(method="average").to_numpy(float)
        positive = diagnostic["ideal"].to_numpy(bool)
        n_positive, n_negative = int(positive.sum()), int((~positive).sum())
        rank_biserial = None
        if n_positive and n_negative:
            u_statistic = float(ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
            rank_biserial = 2.0 * u_statistic / (n_positive * n_negative) - 1.0
        diagnostic["decile"] = pd.qcut(diagnostic["value"], 10, labels=False, duplicates="drop")
        decile_rates = diagnostic.groupby("decile", sort=True)["ideal"].mean()
        decile_monotonicity = (
            float(np.corrcoef(np.arange(len(decile_rates)), decile_rates.to_numpy(float))[0, 1])
            if len(decile_rates) >= 2 and decile_rates.nunique() > 1 else None
        )
        rows.append({
            "feature": name, "analysis_type": "INTERACTION" if name in interactions else "FEATURE",
            "analysis_sample_n": len(sample), "ideal_n": len(ideal), "normal_n": len(normal),
            "ideal_median": float(ideal.median()) if len(ideal) else None,
            "normal_median": float(normal.median()) if len(normal) else None,
            "standardized_median_difference": standardized,
            "rank_biserial": rank_biserial,
            "decile_monotonicity": decile_monotonicity,
            "decile_ideal_rates": decile_rates.to_list(),
            "mutual_information": float(mi[index]),
        })
    return pd.DataFrame(rows).sort_values(["mutual_information", "feature"], ascending=[False, True], kind="mergesort", ignore_index=True)


def _model_task(task: tuple[int, str]) -> tuple[list[FrozenCandidate], pd.DataFrame]:
    horizon, side = task
    config = json.loads(json.dumps(MODEL_CONTEXT["config"]))
    config["teachers"]["horizons_minutes"] = [horizon]
    config["teachers"]["sides"] = [side]
    return fit_direct_candidates(
        MODEL_CONTEXT["discovery"], MODEL_CONTEXT["validation"],
        MODEL_CONTEXT["features"], config, include_two_stage=side == "LONG",
    )


def fit_candidates_parallel(discovery: pd.DataFrame, validation: pd.DataFrame, names: list[str], config: Mapping[str, Any], workers: int) -> tuple[list[FrozenCandidate], pd.DataFrame]:
    MODEL_CONTEXT.clear()
    MODEL_CONTEXT.update({"discovery": discovery, "validation": validation, "features": names, "config": dict(config)})
    tasks = [(int(horizon), side) for horizon in config["teachers"]["horizons_minutes"] for side in config["teachers"]["sides"]]
    # Linux fork shares immutable parent pages; each estimator remains single-threaded.
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        results = list(executor.map(_model_task, tasks))
    candidates = [candidate for group, _ in results for candidate in group]
    reports = pd.concat([report for _, report in results], ignore_index=True)
    return candidates, reports.drop_duplicates("candidate").sort_values("candidate", kind="mergesort", ignore_index=True)


def candidate_population(candidate: FrozenCandidate, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    horizon = candidate.horizon_minutes
    subset = frame[frame["horizon_minutes"].eq(horizon)]
    if candidate.formulation == "OPPORTUNITY_THEN_SIDE":
        base = subset.drop_duplicates(["decision_at", "symbol"]).sort_values(["decision_at", "symbol"], kind="mergesort").copy()
        scores, sides = candidate.score(base)
        lookup = subset.set_index(["decision_at", "symbol", "side"])
        labels, gross, quality, mfe, mae = [], [], [], [], []
        for row, side in zip(base.itertuples(), sides, strict=True):
            outcome = lookup.loc[(row.decision_at, row.symbol, side)]
            labels.append(bool(outcome["zone_best"]))
            gross.append(float(outcome["policy_gross_bps"]))
            quality.append(float(outcome["entry_quality_score"]))
            mfe.append(float(outcome["mfe_bps"]))
            mae.append(float(outcome["mae_bps"]))
        base["predicted_side"] = sides
        base["entry_quality_score"] = quality
        base["mfe_bps"] = mfe
        base["mae_bps"] = mae
        return base, scores, np.asarray(labels), np.asarray(gross)
    base = subset[subset["side"].eq(candidate.side)].sort_values(["decision_at", "symbol"], kind="mergesort").copy()
    scores, sides = candidate.score(base)
    base["predicted_side"] = sides
    return base, scores, base["zone_best"].to_numpy(bool), base["policy_gross_bps"].to_numpy(float)


def evaluate_candidate(candidate: FrozenCandidate, frame: pd.DataFrame, config: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    base, scores, labels, gross = candidate_population(candidate, frame)
    metrics, _, selected = ranked_metrics(
        labels, scores, gross, base["decision_at"], base["symbol"],
        [int(value) for value in config["selection"]["top_percentiles"]], candidate.thresholds,
    )
    base = base.reset_index(drop=True)
    base["score"] = scores
    base["actual_ideal"] = labels
    base["policy_gross_bps_selected_side"] = gross
    base["selected_top1"] = scores >= candidate.thresholds[1]
    base["selected_top2"] = scores >= candidate.thresholds[2]
    base["selected_top5"] = scores >= candidate.thresholds[5]
    base["selected_top10"] = scores >= candidate.thresholds[10]
    base["decision_model_family"] = "W12_IDEAL_ENTRY"
    base["decision_model_instance_id"] = f"W12_{candidate.name}_DISCOVERY_FROZEN"
    base["entry_quality_score_predicted"] = np.clip(scores * 100.0, 0.0, 100.0)
    return metrics, base


def day_block_bootstrap(trades: pd.DataFrame, draws: int, seed: int) -> dict[str, Any]:
    if trades.empty:
        return {"draws": draws, "mean": None, "ci_lower": None, "ci_upper": None, "probability_positive": None}
    days = [group["net14_bps"].to_numpy(float) for _, group in trades.assign(day=pd.to_datetime(trades["decision_at"], utc=True).dt.floor("D")).groupby("day", sort=True)]
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for index in range(draws):
        selected = rng.integers(0, len(days), len(days))
        values[index] = np.concatenate([days[item] for item in selected]).mean()
    return {
        "draws": draws, "mean": float(trades["net14_bps"].mean()),
        "ci_lower": float(np.quantile(values, 0.025)), "ci_upper": float(np.quantile(values, 0.975)),
        "probability_positive": float((values > 0).mean()),
    }


def _negative_model_control(
    candidate: FrozenCandidate,
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    prospective: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    random_features: bool,
) -> np.ndarray:
    """Refit the frozen selected formulation after destroying feature/label relation."""
    rng = np.random.default_rng(int(config["seed"]) + (101 if random_features else 37))
    names = (
        [f"random_feature_{index}" for index in range(int(config["negative_controls"]["random_feature_count"]))]
        if random_features else list(candidate.feature_names)
    )

    def control_matrix(frame: pd.DataFrame, stage: int) -> pd.DataFrame:
        if not random_features:
            return frame[list(candidate.feature_names)]
        stage_rng = np.random.default_rng(int(config["seed"]) + 1000 + stage)
        return pd.DataFrame(stage_rng.normal(size=(len(frame), len(names))), columns=names, index=frame.index)

    if candidate.formulation == "OPPORTUNITY_THEN_SIDE":
        horizon = candidate.horizon_minutes

        def populations(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            labels = frame[frame["horizon_minutes"].eq(horizon)]
            base = labels.drop_duplicates(["decision_at", "symbol"]).sort_values(["decision_at", "symbol"], kind="mergesort").copy()
            return base, labels

        train_base, train_labels = populations(discovery)
        validation_base, validation_labels = populations(validation)
        prospective_base, prospective_labels = populations(prospective)
        keys = ["decision_at", "symbol"]
        positives = train_labels[train_labels["zone_best"]].sort_values(
            [*keys, "entry_quality_score", "side"], ascending=[True, True, False, True], kind="mergesort"
        ).drop_duplicates(keys)
        positive_index = pd.MultiIndex.from_frame(positives[keys])
        train_index = pd.MultiIndex.from_frame(train_base[keys])
        y_opportunity = train_index.isin(positive_index).astype(int)
        any_ideal = train_labels.groupby(keys, sort=True)["majority_ideal"].any().reindex(train_index).to_numpy(bool)
        zone_proxy = train_base[keys].copy()
        zone_proxy["majority_ideal"] = any_ideal
        keep = y_opportunity.astype(bool) | (
            ~any_ideal & temporal_negative_mask(zone_proxy, int(config["zones"]["negative_exclusion_minutes"]))
        )
        train_base = train_base.loc[keep].reset_index(drop=True)
        y_opportunity = y_opportunity[keep]
        y_direction = positives["side"].eq("LONG").astype(int).to_numpy()
        if not random_features:
            rng.shuffle(y_opportunity)
            rng.shuffle(y_direction)
        opportunity = clone(candidate.model[0]).fit(control_matrix(train_base, 0), y_opportunity)
        direction = clone(candidate.model[1]).fit(control_matrix(positives, 1), y_direction)

        def score(base: pd.DataFrame, labels: pd.DataFrame, stage: int) -> tuple[np.ndarray, np.ndarray]:
            values = control_matrix(base, stage)
            p_opportunity = opportunity.predict_proba(values)[:, 1]
            p_long = direction.predict_proba(values)[:, 1]
            sides = np.where(p_long >= 0.5, "LONG", "SHORT")
            scores = p_opportunity * np.maximum(p_long, 1.0 - p_long)
            lookup = labels.set_index(["decision_at", "symbol", "side"])
            gross = np.array([
                float(lookup.loc[(row.decision_at, row.symbol, side)]["policy_gross_bps"])
                for row, side in zip(base.itertuples(), sides, strict=True)
            ])
            return scores, gross

        validation_scores, _ = score(validation_base, validation_labels, 2)
        threshold = float(np.quantile(validation_scores, 0.98))
        prospective_scores, prospective_gross = score(prospective_base, prospective_labels, 3)
        return prospective_gross[prospective_scores >= threshold]

    train = discovery[
        discovery["horizon_minutes"].eq(candidate.horizon_minutes)
        & discovery["side"].eq(candidate.side)
    ].copy()
    validation_set = validation[
        validation["horizon_minutes"].eq(candidate.horizon_minutes)
        & validation["side"].eq(candidate.side)
    ].copy()
    prospective_set = prospective[
        prospective["horizon_minutes"].eq(candidate.horizon_minutes)
        & prospective["side"].eq(candidate.side)
    ].copy()
    keep = train["zone_best"].to_numpy(bool) | (
        ~train["majority_ideal"].to_numpy(bool)
        & temporal_negative_mask(train, int(config["zones"]["negative_exclusion_minutes"]))
    )
    train = train.loc[keep].copy()
    model = clone(candidate.model)
    if candidate.formulation == "QUALITY_REGRESSION":
        target = train["entry_quality_score"].to_numpy(float).copy()
        if not random_features:
            rng.shuffle(target)
        model.fit(control_matrix(train, 0), target)
        validation_scores = np.clip(model.predict(control_matrix(validation_set, 1)) / 100.0, 0, 1)
        prospective_scores = np.clip(model.predict(control_matrix(prospective_set, 2)) / 100.0, 0, 1)
    else:
        target = train["zone_best"].to_numpy(int).copy()
        if not random_features:
            rng.shuffle(target)
        model.fit(control_matrix(train, 0), target)
        validation_scores = model.predict_proba(control_matrix(validation_set, 1))[:, 1]
        prospective_scores = model.predict_proba(control_matrix(prospective_set, 2))[:, 1]
    threshold = float(np.quantile(validation_scores, 0.98))
    return prospective_set.loc[prospective_scores >= threshold, "policy_gross_bps"].to_numpy(float)


def controls_and_baselines(candidate: FrozenCandidate, discovery: pd.DataFrame, validation: pd.DataFrame, prospective: pd.DataFrame, predictions: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    selected = predictions[predictions["selected_top2"]].copy()
    n = len(selected)
    rows = []

    def record(name: str, gross: np.ndarray, kind: str) -> None:
        rows.append({
            "name": name, "kind": kind, "trades": len(gross),
            "gross_mean_bps": float(np.mean(gross)) if len(gross) else None,
            "net14_mean_bps": float(np.mean(gross) - 14.0) if len(gross) else None,
            "net20_mean_bps": float(np.mean(gross) - 20.0) if len(gross) else None,
        })

    record("W12_PRIMARY", selected["policy_gross_bps_selected_side"].to_numpy(float), "MODEL")
    record("ALWAYS_SKIP", np.array([], dtype=float), "BASELINE")
    base = prospective[prospective["horizon_minutes"].eq(candidate.horizon_minutes)].drop_duplicates(["decision_at", "symbol"])
    lookup = prospective[prospective["horizon_minutes"].eq(candidate.horizon_minutes)].set_index(["decision_at", "symbol", "side"])
    long_gross, short_gross, momentum_gross, reversion_gross = [], [], [], []
    for row in base.itertuples():
        long_row = lookup.loc[(row.decision_at, row.symbol, "LONG")]
        short_row = lookup.loc[(row.decision_at, row.symbol, "SHORT")]
        long_gross.append(float(long_row["policy_gross_bps"]))
        short_gross.append(float(short_row["policy_gross_bps"]))
        momentum_long = float(row.return_15m_bps) >= 0
        momentum_gross.append(float(long_row["policy_gross_bps"] if momentum_long else short_row["policy_gross_bps"]))
        reversion_gross.append(float(short_row["policy_gross_bps"] if momentum_long else long_row["policy_gross_bps"]))
    record("ALWAYS_LONG", np.asarray(long_gross), "BASELINE")
    record("ALWAYS_SHORT", np.asarray(short_gross), "BASELINE")
    record("15M_MOMENTUM", np.asarray(momentum_gross), "BASELINE")
    record("15M_MEAN_REVERSION", np.asarray(reversion_gross), "BASELINE")
    rng = np.random.default_rng(int(config["seed"]))
    pool = base.index.to_numpy()
    random_means = []
    for _ in range(int(config["negative_controls"]["random_entry_repetitions"])):
        chosen = rng.choice(pool, size=min(n, len(pool)), replace=False)
        sides = rng.choice(["LONG", "SHORT"], size=len(chosen))
        random_means.append(np.mean([
            float(lookup.loc[(base.loc[index, "decision_at"], base.loc[index, "symbol"], side)]["policy_gross_bps"])
            for index, side in zip(chosen, sides, strict=True)
        ]))
    record("RANDOM_ENTRIES_MEAN", np.asarray(random_means), "NEGATIVE_CONTROL_DISTRIBUTION")
    shifted = predictions.copy()
    shift_rows = int(config["negative_controls"]["time_shift_hours"]) * 60 // int(config["source"]["decision_cadence_minutes"])
    shifted[["score", "predicted_side"]] = shifted.groupby("symbol", sort=False)[["score", "predicted_side"]].shift(shift_rows)
    shifted["policy_gross_bps_selected_side"] = [
        float(lookup.loc[(row.decision_at, row.symbol, row.predicted_side)]["policy_gross_bps"])
        if pd.notna(row.predicted_side) else np.nan
        for row in shifted.itertuples()
    ]
    shifted_selected = shifted[shifted["score"].ge(candidate.thresholds[2])]
    record("TIME_SHIFT_24H", shifted_selected["policy_gross_bps_selected_side"].to_numpy(float), "NEGATIVE_CONTROL")
    shuffled_gross = _negative_model_control(candidate, discovery, validation, prospective, config, random_features=False)
    record("LABEL_SHUFFLE_MODEL", shuffled_gross, "NEGATIVE_CONTROL")
    random_feature_gross = _negative_model_control(candidate, discovery, validation, prospective, config, random_features=True)
    record("RANDOM_FEATURE_MODEL", random_feature_gross, "NEGATIVE_CONTROL")
    return pd.DataFrame(rows)


def stability_metrics(trades: pd.DataFrame, discovery: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    work = trades.copy()
    work["month"] = pd.to_datetime(work["decision_at"], utc=True).dt.strftime("%Y-%m")
    volatility = pd.to_numeric(discovery["realized_vol_60m_bps"], errors="coerce").dropna()
    cuts = volatility.quantile([1 / 3, 2 / 3]).to_numpy(float)
    work["volatility_tercile"] = np.asarray(["LOW", "MID", "HIGH"])[np.searchsorted(cuts, work["realized_vol_60m_bps"].to_numpy(float), side="right")]
    for field in ("symbol", "predicted_side", "month", "horizon_minutes", "volatility_tercile"):
        for value, group in work.groupby(field, sort=True):
            rows.append({"dimension": field, "value": value, "trades": len(group), "gross_mean_bps": float(group["gross_bps"].mean()), "net14_mean_bps": float(group["net14_bps"].mean()), "ideal_precision": float(group["actual_ideal"].mean())})
    return pd.DataFrame(rows)


def run_experiment(config: Mapping[str, Any], repository: Path, workers: int = 4) -> ExperimentResult:
    features, labels, cache_manifest = load_or_build_cache(config, repository, workers)
    names = list(cache_manifest["feature_names"])
    merged = attach_features(labels, features, names)
    partitions = []
    for name in ("discovery", "validation", "prospective"):
        part = partition_frame(merged, config, name)
        part["partition"] = name.upper()
        partitions.append(part)
    full = pd.concat(partitions, ignore_index=True)
    discovery = full[full["partition"].eq("DISCOVERY")]
    validation = full[full["partition"].eq("VALIDATION")]
    prospective = full[full["partition"].eq("PROSPECTIVE")]
    labels_report = label_analysis(full)
    features_report = discovery_feature_analysis(discovery, names, int(config["seed"]))
    _checkpoint("DISCOVERY", {"rows": len(discovery), "ideal_zones": int(discovery["zone_best"].sum())})
    candidates, candidate_report = fit_candidates_parallel(discovery, validation, names, config, workers)
    selected = select_candidate(candidates)
    _checkpoint("TRAINING", {"candidates": len(candidates), "selected": selected.name})
    _checkpoint("VALIDATION", {"selected": selected.name, "net14_top2_bps": selected.validation_primary_net14_bps})
    prospective_metrics, predictions = evaluate_candidate(selected, prospective, config)
    trades = predictions[predictions["selected_top2"]].copy()
    trades["gross_bps"] = trades["policy_gross_bps_selected_side"]
    trades["fees_bps"] = 10.0
    trades["slippage_bps"] = 4.0
    trades["net14_bps"] = trades["gross_bps"] - 14.0
    trades["net20_bps"] = trades["gross_bps"] - 20.0
    trades["net30_bps"] = trades["gross_bps"] - 30.0
    trades["holding_minutes"] = selected.horizon_minutes
    trades["decision"] = "ENTER"
    _checkpoint("PROSPECTIVE", {"trades": len(trades), "net14_bps": float(trades["net14_bps"].mean()) if len(trades) else None})
    controls = controls_and_baselines(selected, discovery, validation, prospective, predictions, config)
    _checkpoint("NEGATIVE_CONTROLS", {"rows": len(controls)})
    bootstrap = day_block_bootstrap(trades, int(config["selection"]["bootstrap_draws"]), int(config["seed"]))
    _checkpoint("BOOTSTRAP", bootstrap)
    stability = stability_metrics(trades, discovery)
    symbol_counts = trades["symbol"].value_counts()
    month_metrics = stability[stability["dimension"].eq("month")]
    primary_top = prospective_metrics["top"]["2"]
    negative_controls = controls[controls["kind"].str.startswith("NEGATIVE_CONTROL") & controls["net14_mean_bps"].notna()]
    baselines = controls[controls["kind"].eq("BASELINE") & controls["net14_mean_bps"].notna()]
    controls_fail = bool(len(negative_controls) and primary_top["net14_mean_bps"] > negative_controls["net14_mean_bps"].max())
    baseline_superiority = bool(len(baselines) and primary_top["net14_mean_bps"] > baselines["net14_mean_bps"].max())
    gates = {
        "minimum_trades": len(trades) >= int(config["selection"]["minimum_prospective_trades"]),
        "minimum_symbols": trades["symbol"].nunique() >= int(config["selection"]["minimum_symbols"]) if len(trades) else False,
        "symbol_concentration": bool(len(trades) and symbol_counts.max() / len(trades) <= float(config["selection"]["maximum_symbol_fraction"])),
        "net14_positive": bool(len(trades) and trades["net14_bps"].mean() > 0),
        "net20_positive": bool(len(trades) and trades["net20_bps"].mean() > 0),
        "bootstrap_lower_positive": bool(bootstrap["ci_lower"] is not None and bootstrap["ci_lower"] > 0),
        "precision_lift": bool(primary_top["precision_lift"] is not None and primary_top["precision_lift"] >= float(config["success"]["require_precision_lift_top2"])),
        "multiple_periods": bool(len(month_metrics) >= 2 and (month_metrics["net14_mean_bps"] > 0).sum() >= 2),
        "negative_controls_fail": controls_fail,
        "baseline_superiority": baseline_superiority,
        "leakage_audit": True,
    }
    predictive = bool(primary_top["precision_lift"] is not None and primary_top["precision_lift"] > 1.0 and prospective_metrics["pr_auc"] is not None and prospective_metrics["pr_auc"] > prospective_metrics["prevalence"])
    if all(gates.values()):
        grade, verdict = "A", "IDEAL_ENTRY_ALPHA_CONFIRMED"
    elif predictive:
        grade, verdict = "B", "IDEAL_ENTRY_SIGNAL_DETECTED_NOT_YET_ECONOMIC"
    else:
        grade, verdict = "C", "NO_IDEAL_ENTRY_PREDICTIVE_EDGE"
    summary = {
        "grade": grade, "verdict": verdict, "gates": gates,
        "selected_candidate": {"name": selected.name, "formulation": selected.formulation, "side": selected.side, "horizon_minutes": selected.horizon_minutes},
        "counts": {
            "discovery_rows": len(discovery), "validation_rows": len(validation), "prospective_rows": len(prospective),
            "discovery_ideal_zones": int(discovery["zone_best"].sum()),
            "validation_ideal_zones": int(validation["zone_best"].sum()),
            "prospective_ideal_zones": int(prospective["zone_best"].sum()),
            "prospective_opportunities": len(trades), "prospective_symbols": trades["symbol"].nunique(),
        },
        "prospective_metrics": prospective_metrics,
        "prospective_economics": {
            "gross_mean_bps": float(trades["gross_bps"].mean()) if len(trades) else None,
            "net14_mean_bps": float(trades["net14_bps"].mean()) if len(trades) else None,
            "net20_mean_bps": float(trades["net20_bps"].mean()) if len(trades) else None,
            "net30_mean_bps": float(trades["net30_bps"].mean()) if len(trades) else None,
            "median_net14_bps": float(trades["net14_bps"].median()) if len(trades) else None,
            "win_rate_net14": float(trades["net14_bps"].gt(0).mean()) if len(trades) else None,
            "skip_fraction": 1.0 - len(trades) / len(predictions) if len(predictions) else None,
        },
        "bootstrap": bootstrap,
        "negative_controls_passed": controls_fail,
        "leakage_audit": "PASS",
        "external_holdouts_accessed": False,
        "merits_w12_1": grade == "A",
        "merits_shadow": False,
    }
    economics = pd.DataFrame([
        {"scenario": "BASELINE_14", "cost_bps": 14.0, "trades": len(trades), "mean_net_bps": summary["prospective_economics"]["net14_mean_bps"]},
        {"scenario": "STRESS_20", "cost_bps": 20.0, "trades": len(trades), "mean_net_bps": summary["prospective_economics"]["net20_mean_bps"]},
        {"scenario": "SEVERE_30", "cost_bps": 30.0, "trades": len(trades), "mean_net_bps": summary["prospective_economics"]["net30_mean_bps"]},
    ])
    return ExperimentResult(summary, features_report, labels_report, candidate_report, predictions, trades, economics, controls, stability, selected, cache_manifest)
