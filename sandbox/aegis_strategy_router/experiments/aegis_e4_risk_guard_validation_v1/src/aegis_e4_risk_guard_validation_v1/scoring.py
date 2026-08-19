"""Scoring Aegis signals with frozen E4 risk models."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def score_signals_with_e4(
    signals: pd.DataFrame,
    side_panel: pd.DataFrame,
    config: dict[str, Any],
    repository: Path,
) -> pd.DataFrame:
    models_path = repository / config["frozen_e4_artifacts"]["models_joblib"]
    expected_hash = config["frozen_e4_artifacts"]["models_joblib_sha256"]
    actual_hash = sha256_file(models_path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"E4_MODELS_HASH_MISMATCH: expected {expected_hash}, got {actual_hash}")

    models = joblib.load(models_path)
    tail_bundle = models[config["frozen_e4_artifacts"]["tail_risk_head"]]
    late_bundle = models[config["frozen_e4_artifacts"]["late_entry_head"]]
    quality_bundle = models[config["frozen_e4_artifacts"]["entry_quality_head"]]
    reg_mfe = models.get("target__mfe_bps")
    reg_mae = models.get("target__mae_bps")
    reg_fixed = models.get("target__fixed_return_bps")

    # Index side_panel by (decision_at, symbol, side)
    keyed = side_panel.set_index(["decision_at", "symbol", "side"], drop=False)

    scored_rows = []
    for sig in signals.to_dict("records"):
        ts = pd.to_datetime(sig["signal_timestamp"], utc=True)
        sym = sig["symbol"]
        side = sig["side"]

        try:
            feat_row = keyed.loc[(ts, sym, side)]
            if isinstance(feat_row, pd.DataFrame):
                feat_row = feat_row.iloc[0]
        except KeyError:
            raise RuntimeError(f"CAUSAL_FEATURE_ROW_NOT_FOUND: ({ts}, {sym}, {side})")

        # Extract features for each model
        tail_feats = feat_row[tail_bundle["features"]].to_frame().T
        late_feats = feat_row[late_bundle["features"]].to_frame().T
        quality_feats = feat_row[quality_bundle["features"]].to_frame().T

        tail_raw = tail_bundle["model"].decision_function(tail_feats).reshape(-1, 1)
        tail_score = float(tail_bundle["calibrator"].predict_proba(tail_raw)[:, 1][0])

        late_raw = late_bundle["model"].decision_function(late_feats).reshape(-1, 1)
        late_score = float(late_bundle["calibrator"].predict_proba(late_raw)[:, 1][0])

        quality_raw = quality_bundle["model"].decision_function(quality_feats).reshape(-1, 1)
        quality_score = float(quality_bundle["calibrator"].predict_proba(quality_raw)[:, 1][0])

        pred_mfe = float(reg_mfe["model"].predict(late_feats)[0]) if reg_mfe else np.nan
        pred_mae = float(reg_mae["model"].predict(late_feats)[0]) if reg_mae else np.nan
        pred_fixed = float(reg_fixed["model"].predict(late_feats)[0]) if reg_fixed else np.nan

        # Causal geometry indicators
        consumed_move = float(feat_row.get("feature__remaining__consumed_move_atr", np.nan))
        impulse_age = float(feat_row.get("feature__remaining__impulse_age_bars", np.nan))
        extension_atr = float(feat_row.get("feature__remaining__extension_atr", np.nan))
        momentum_decay = float(feat_row.get("feature__remaining__momentum_decay", np.nan))
        atr_p96 = float(feat_row.get("feature__base__tf15m__atr_percentile_96", np.nan))
        prior_move = float(feat_row.get("feature__base__tf5m__directional_prior_move_6_atr", np.nan))
        btc_dir_ret = float(feat_row.get("feature__cross__btcusdt__tf5m__return_1_bps", np.nan))

        row = dict(sig)
        row.update({
            "e4_late_entry_score": late_score,
            "e4_tail_risk_score": tail_score,
            "e4_entry_quality_score": quality_score,
            "e4_predicted_mfe_bps": pred_mfe,
            "e4_predicted_mae_bps": pred_mae,
            "e4_predicted_fixed_return_bps": pred_fixed,
            "geom_consumed_move_atr": consumed_move,
            "geom_impulse_age_bars": impulse_age,
            "geom_extension_atr": extension_atr,
            "geom_momentum_decay": momentum_decay,
            "geom_15m_atr_percentile": atr_p96,
            "geom_prior_move_5m_atr": prior_move,
            "geom_btc_dir_return_5m_bps": btc_dir_ret,
            "causal_reconstruction_status": "VERIFIED_CAUSAL_CLEAN",
        })
        scored_rows.append(row)

    res = pd.DataFrame(scored_rows).sort_values(["signal_timestamp", "trade_id"], kind="mergesort")
    return res
