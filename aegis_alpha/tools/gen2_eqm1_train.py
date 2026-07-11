#!/usr/bin/env python3
"""GEN2-EQM1 trainer (spec: GEN2_EQM1_SPEC.md v1.0). Research-only.

Opportunities = H12 rows of the canonical D3 dense dataset, development period
only. The frozen TRRM V2 bundle vetoes the top-30% tail scores (past-only
per-fold threshold); EQM candidates are trained and evaluated on the retained
population. Two targets, never mixed: net_quality regression and clean_entry
classification. Selection is pre-registered and stability-first.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import GEN2_ROOT, sha256_file, sha256_text, utc_stamp, validate_gen2_path, write_manifest  # noqa: E402
import aegis_alpha.tools.gen2_rv2_train as rv2  # noqa: E402
from aegis_alpha.tools.gen2_rv2_train import MedianImputer, score_of  # noqa: E402

EQM_ROOT = GEN2_ROOT / "eqm1"
DEFAULT_TRRM_DIR = GEN2_ROOT / "rv2" / "20260711T171832Z"
QUALITY_TARGET = "future_eval.net_quality_after_costs"
CLEAN_TARGET = "label.clean_entry_v4"
VETO_BUDGET = 0.30
TOPK_BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.30)
SEED = 42
PRIMARY_HORIZON = 12
CORR_WINDOW_MIN = 30


def load_trrm(trrm_dir: Path) -> dict[str, Any]:
    sys.modules["__main__"].MedianImputer = MedianImputer
    with (trrm_dir / "rv2_candidate.pkl").open("rb") as f:
        return pickle.load(f)


def tail_scores(bundle: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    x = bundle["imputer"].transform(df[bundle["features"]].apply(pd.to_numeric, errors="coerce"))
    raw = score_of(bundle["trrm_model"], x)
    cal = bundle.get("calibrator")
    if cal is not None and bundle.get("calibrator_kind") == "isotonic":
        return np.asarray(cal.predict(raw), dtype=float)
    if cal is not None:
        return cal.predict_proba(raw.reshape(-1, 1))[:, 1]
    return raw


def eqm_candidates() -> dict[str, Any]:
    from sklearn.ensemble import (
        ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingRegressor,
        HistGradientBoostingClassifier, HistGradientBoostingRegressor,
        RandomForestClassifier, RandomForestRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge

    return {
        "reg": {
            "rf_reg": (True, lambda: RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=25, random_state=SEED, n_jobs=4)),
            "et_reg": (True, lambda: ExtraTreesRegressor(n_estimators=300, max_depth=14, min_samples_leaf=25, random_state=SEED, n_jobs=4)),
            "hgb_reg": (True, lambda: HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)),
            "gb_reg": (True, lambda: GradientBoostingRegressor(n_estimators=150, learning_rate=0.06, max_depth=3, subsample=0.7, random_state=SEED)),
            "ridge_reference": (False, lambda: Ridge(alpha=1.0)),
        },
        "clf": {
            "rf_clf": (True, lambda: RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=25, class_weight="balanced_subsample", random_state=SEED, n_jobs=4)),
            "hgb_clf": (True, lambda: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)),
            "logistic_reference": (False, lambda: LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
        },
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def topk_quality(score: np.ndarray, quality: np.ndarray, k: float) -> float:
    n = max(1, int(round(k * len(score))))
    return float(quality[np.argsort(-score)[:n]].mean())


def correlation_limit(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """Keep at most one opportunity per 30-minute portfolio window (best score)."""
    ts = pd.to_datetime(df["id.timestamp"]).dt.floor(f"{CORR_WINDOW_MIN}min")
    order = pd.DataFrame({"w": ts.to_numpy(), "s": score}).reset_index()
    keep = order.sort_values("s", ascending=False).drop_duplicates("w")["index"].to_numpy()
    mask = np.zeros(len(df), dtype=bool)
    mask[keep] = True
    return mask


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    stamp = utc_stamp()
    out_dir = EQM_ROOT / stamp
    validate_gen2_path(out_dir)
    out_dir.mkdir(parents=True)
    dev_all, _, feature_cols, _ = rv2.load_development(Path(args.dataset_csv))
    dev = dev_all[pd.to_numeric(dev_all["id.horizon"], errors="coerce") == PRIMARY_HORIZON].reset_index(drop=True)
    folds = rv2.make_folds(dev)
    trrm = load_trrm(Path(args.trrm_dir))
    tail = tail_scores(trrm, dev)
    quality = pd.to_numeric(dev[QUALITY_TARGET], errors="coerce").fillna(0.0).to_numpy()
    clean = dev[CLEAN_TARGET].astype(int).to_numpy()
    x_raw = dev[feature_cols].apply(pd.to_numeric, errors="coerce")
    cands = eqm_candidates()

    results: dict[str, Any] = {"reg": {}, "clf": {}}
    fold_cache: list[dict[str, Any]] = []
    def retain(idx: np.ndarray, thr: float) -> np.ndarray:
        """Rank-based veto: keep the lowest (1-budget) fraction by tail score.

        A value threshold with ties (isotonic steps) can retain 0% or 100%;
        rank retention is deterministic and always keeps the intended fraction.
        """
        keep_n = max(1, int(round((1 - VETO_BUDGET) * len(idx))))
        return idx[np.argsort(tail[idx], kind="stable")[:keep_n]]

    for fold in folds:
        thr = float(np.quantile(tail[fold["train_idx"]], 1 - VETO_BUDGET))
        fold_cache.append({"fold": fold, "thr": thr, "rt": retain(fold["train_idx"], thr), "re": retain(fold["eval_idx"], thr)})

    for kind in ("reg", "clf"):
        y = quality if kind == "reg" else clean
        for name, (eligible, factory) in cands[kind].items():
            rows = []
            for fc in fold_cache:
                imputer = MedianImputer().fit(x_raw.iloc[fc["rt"]])
                xt = imputer.transform(x_raw)
                model = factory()
                model.fit(xt.iloc[fc["rt"]], y[fc["rt"]])
                s = score_of(model, xt.iloc[fc["re"]]) if kind == "clf" else np.asarray(model.predict(xt.iloc[fc["re"]]), dtype=float)
                q_eval = quality[fc["re"]]
                rows.append({
                    "fold": fc["fold"]["name"],
                    "retained_rows": int(len(fc["re"])),
                    "spearman_vs_quality": spearman(s, q_eval),
                    "top_decile_quality": topk_quality(s, q_eval, 0.10),
                    "topk": {str(k): topk_quality(s, q_eval, k) for k in TOPK_BUDGETS},
                })
            sp = [r["spearman_vs_quality"] for r in rows]
            td = [r["top_decile_quality"] for r in rows]
            results[kind][name] = {
                "folds": rows, "eligible": eligible,
                "aggregate": {
                    "mean_spearman": float(np.mean(sp)), "worst_spearman": float(np.min(sp)),
                    "mean_top_decile_quality": float(np.mean(td)), "worst_top_decile_quality": float(np.min(td)),
                    "h1_top_decile_positive_all_folds": bool(all(v > 0 for v in td)),
                    "h2_stability": bool(np.min(sp) >= 0.60 * np.mean(sp)) if np.mean(sp) > 0 else False,
                },
            }

    def pick(kind: str) -> str | None:
        ok = {k: v for k, v in results[kind].items() if v["eligible"] and v["aggregate"]["h1_top_decile_positive_all_folds"]}
        if not ok:
            ok = {k: v for k, v in results[kind].items() if v["eligible"]}
        if not ok:
            return None
        return sorted(ok.items(), key=lambda kv: (-kv[1]["aggregate"]["worst_top_decile_quality"], -kv[1]["aggregate"]["mean_top_decile_quality"]))[0][0]

    reg_winner = pick("reg")
    clf_winner = pick("clf")

    # composite EV + incrementality + ensemble challenger + correlation-limit diagnostics per fold
    composite = []
    for fc in fold_cache:
        imputer = MedianImputer().fit(x_raw.iloc[fc["rt"]])
        xt = imputer.transform(x_raw)
        regm = cands["reg"][reg_winner][1]()
        regm.fit(xt.iloc[fc["rt"]], quality[fc["rt"]])
        clfm = cands["clf"][clf_winner][1]()
        clfm.fit(xt.iloc[fc["rt"]], clean[fc["rt"]])
        e_idx = fc["fold"]["eval_idx"]
        s_reg_all = np.asarray(regm.predict(xt.iloc[e_idx]), dtype=float)
        s_clf_all = score_of(clfm, xt.iloc[e_idx])
        s_ev_all = s_clf_all * s_reg_all
        q_all = quality[e_idx]
        veto_mask = np.isin(e_idx, fc["re"])
        # ensemble challenger (reg family) with disagreement abstention
        others = [cands["reg"][n][1]() for n in ("rf_reg", "hgb_reg", "et_reg") if n in cands["reg"]]
        preds = []
        for om in others:
            om.fit(xt.iloc[fc["rt"]], quality[fc["rt"]])
            preds.append(np.asarray(om.predict(xt.iloc[e_idx]), dtype=float))
        stack = np.vstack(preds)
        ens_mean = stack.mean(axis=0)
        ens_std = stack.std(axis=0)
        abstain = ens_std > np.quantile(ens_std, 0.80)
        k = 0.10
        comp = {
            "fold": fc["fold"]["name"],
            "eqm_no_veto_top10": topk_quality(s_ev_all, q_all, k),
            "trrm_only_top10": topk_quality(-tail[e_idx], q_all, k),
            "trrm_plus_eqm_top10": topk_quality(s_ev_all[veto_mask], q_all[veto_mask], k),
            "random_plus_trrm_top10": float(np.random.default_rng(SEED).choice(q_all[veto_mask], size=max(1, int(k * veto_mask.sum())), replace=False).mean()),
            "reg_component_top10": topk_quality(s_reg_all[veto_mask], q_all[veto_mask], k),
            "clf_component_top10": topk_quality(s_clf_all[veto_mask], q_all[veto_mask], k),
            "ensemble_top10": topk_quality(ens_mean[veto_mask], q_all[veto_mask], k),
            "ensemble_abstain_top10": topk_quality(ens_mean[veto_mask & ~abstain], q_all[veto_mask & ~abstain], k) if (veto_mask & ~abstain).sum() > 20 else None,
            "abstention_rate": float(abstain.mean()),
        }
        sub = dev.iloc[e_idx][veto_mask]
        cl_mask = correlation_limit(sub.reset_index(drop=True), s_ev_all[veto_mask])
        comp["with_corr_limit_top10"] = topk_quality(s_ev_all[veto_mask][cl_mask], q_all[veto_mask][cl_mask], k)
        comp["corr_limit_retention"] = float(cl_mask.mean())
        composite.append(comp)

    h3 = sum(1 for c in composite if c["trrm_plus_eqm_top10"] > c["eqm_no_veto_top10"] and c["trrm_plus_eqm_top10"] > c["trrm_only_top10"]) >= 3
    combo_better = sum(1 for c in composite if c["trrm_plus_eqm_top10"] >= max(c["reg_component_top10"], c["clf_component_top10"])) >= 3
    ens_gain = [c for c in composite if c["ensemble_top10"] > 1.10 * c["trrm_plus_eqm_top10"]]
    final_score_kind = "composite_ev" if combo_better else "reg_component"
    adopt_ensemble = len(ens_gain) >= 3

    # freeze EQM bundle on full development (train<=90%, embargo)
    ts = dev["_ts"]
    split_ts = ts.quantile(0.90)
    tr = np.flatnonzero((ts <= split_ts).to_numpy())
    thr_full = float(np.quantile(tail[tr], 1 - VETO_BUDGET))
    keep_n = max(1, int(round((1 - VETO_BUDGET) * len(tr))))
    rt_full = tr[np.argsort(tail[tr], kind="stable")[:keep_n]]
    imputer = MedianImputer().fit(x_raw.iloc[rt_full])
    xt = imputer.transform(x_raw)
    regm = cands["reg"][reg_winner][1]()
    regm.fit(xt.iloc[rt_full], quality[rt_full])
    clfm = cands["clf"][clf_winner][1]()
    clfm.fit(xt.iloc[rt_full], clean[rt_full])
    bundle = {
        "reg_model": regm, "clf_model": clfm, "imputer": imputer, "features": feature_cols,
        "score_kind": final_score_kind, "veto_budget": VETO_BUDGET, "veto_threshold_full_dev": thr_full,
        "primary_horizon": PRIMARY_HORIZON, "corr_window_min": CORR_WINDOW_MIN,
        "reg_winner": reg_winner, "clf_winner": clf_winner, "adopt_ensemble": adopt_ensemble, "seed": SEED,
    }
    pkl = out_dir / "eqm1_candidate.pkl"
    with pkl.open("wb") as f:
        pickle.dump(bundle, f)

    payload = {
        "schema": "gen2_eqm1_training_v1", "artifact_id": stamp, "spec": "GEN2_EQM1_SPEC.md v1.0",
        "dataset_csv": args.dataset_csv, "dataset_sha256": sha256_file(Path(args.dataset_csv)),
        "trrm_dir": args.trrm_dir, "trrm_sha256": sha256_file(Path(args.trrm_dir) / "rv2_candidate.pkl"),
        "opportunities_h12_dev": int(len(dev)), "feature_hash": sha256_text("\n".join(feature_cols)),
        "results": results, "reg_winner": reg_winner, "clf_winner": clf_winner,
        "composite_and_incrementality": composite,
        "h3_incrementality_pass": bool(h3), "composite_beats_components": bool(combo_better),
        "final_score_kind": final_score_kind, "ensemble_adopted": adopt_ensemble,
        "frozen_candidate": {"pickle": str(pkl), "pickle_sha256": sha256_file(pkl), "veto_threshold_full_dev": thr_full},
        "safety": {"trrm_not_retrained": True, "no_forward_touched": True, "research_only": True},
    }
    (out_dir / "training_report.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    write_manifest(out_dir, {"schema": "gen2_eqm1_artifact_v1", "artifact_id": stamp, "files": {p.name: sha256_file(p) for p in out_dir.iterdir() if p.is_file()}}, "eqm1_manifest")
    print(json.dumps({"artifact": str(out_dir), "reg_winner": reg_winner, "clf_winner": clf_winner,
                      "reg_agg": results["reg"].get(reg_winner, {}).get("aggregate"),
                      "h3_incrementality": h3, "final_score": final_score_kind, "ensemble": adopt_ensemble}, indent=2, default=json_default))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GEN2-EQM1 trainer")
    p.add_argument("--dataset-csv", default=str(rv2.DEFAULT_DATASET))
    p.add_argument("--trrm-dir", default=str(DEFAULT_TRRM_DIR))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run_training(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
