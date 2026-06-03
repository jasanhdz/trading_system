#!/usr/bin/env python3
import json
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from aegis_alpha.tools.walk_forward_long_alpha_family_d import (
    ALLOWED_FAMILY,
    WalkConfig,
    build_expanding_folds,
    classify_d1_symbol,
    configs_from_args,
    score_summary,
    write_reports,
)
from aegis_alpha.tools.train_long_alpha_candidates_c import assert_research_only_path


def summary(**kw):
    base = {
        'symbol': 'ETHUSDT', 'family': ALLOWED_FAMILY, 'target': 'long_hit3_before_minus2', 'horizon': 24,
        'valid_folds': 4, 'negative_folds': 0,
        'mean_hit_lift': 0.02, 'latest_fold_hit_lift': 0.01,
        'mean_net_quality_lift_after_costs': 0.001, 'latest_fold_net_quality_lift_after_costs': 0.0005,
        'mean_p90_mae_delta': 0.0, 'mean_stop_rate_delta': 0.0,
        'mean_selected_fraction': 0.12, 'mean_hit_auc': 0.55, 'mean_hit_top_decile_hit_lift': 0.03,
    }
    base.update(kw)
    return base


def main():
    folds = build_expanding_folds(1000, 4, 300, 100)
    assert len(folds) == 4
    prev_train = 0
    for train, test in folds:
        assert train[0] == 0
        assert train[-1] < test[0]
        assert len(train) >= prev_train
        prev_train = len(train)
    assert classify_d1_symbol(summary()) == 'LONG_D1_CONFIRMED'
    assert classify_d1_symbol(summary(mean_hit_lift=-0.001, latest_fold_hit_lift=0.02, mean_net_quality_lift_after_costs=0.001)) == 'LONG_D1_FAILED'
    assert classify_d1_symbol(summary(mean_hit_lift=0.001, mean_net_quality_lift_after_costs=0.0, latest_fold_net_quality_lift_after_costs=0.001, mean_hit_auc=0.51, mean_hit_top_decile_hit_lift=0.0)) == 'LONG_D1_MIXED'
    assert classify_d1_symbol(summary(mean_net_quality_lift_after_costs=-0.01)) == 'LONG_D1_FAILED'
    better = summary(symbol='AVAXUSDT', mean_hit_lift=0.03)
    worse = summary(symbol='ETHUSDT', mean_hit_lift=0.01)
    assert score_summary(better) > score_summary(worse)
    try:
        assert_research_only_path(ROOT/'aegis_alpha/models/turbo/ETHUSDT/active/x.joblib')
        raise AssertionError('expected active path rejection')
    except ValueError:
        pass
    Args = type('Args', (), {})
    args = Args()
    args.family = ALLOWED_FAMILY
    args.symbols = 'ETHUSDT'
    args.target = 'long_hit3_before_minus2'
    args.horizon = 24
    args.include_secondary = False
    cfgs = configs_from_args(args)
    assert cfgs == [WalkConfig('ETHUSDT', ALLOWED_FAMILY, 'long_hit3_before_minus2', 24)]
    args.family = 'slow_trend_long'
    try:
        configs_from_args(args)
        raise AssertionError('expected family restriction')
    except SystemExit:
        pass
    out = Path(tempfile.mkdtemp())
    args = type('Args', (), {'out_dir': str(out), 'symbols': 'ETHUSDT', 'family': ALLOWED_FAMILY, 'target': 'long_hit3_before_minus2', 'horizon': 24, 'fold_count': 4, 'lookback_days': 1, 'model_dir': str(out/'models/research'), 'db_path': 'x', 'feature_mode': 'selected_family', 'fast': True, 'save_models': False, 'no_save_models': True, 'min_train_samples': 1, 'min_test_samples': 1, 'include_secondary': False})()
    s = summary(d1_status='LONG_D1_CONFIRMED', score=score_summary(summary()), recommendation='pass_to_frozen_confirmation')
    fold = {'symbol':'ETHUSDT','family':ALLOWED_FAMILY,'target':'long_hit3_before_minus2','horizon':24,'fold_index':1,'fold_status':'POSITIVE','hit_lift':0.1,'net_quality_lift_after_costs':0.1,'p90_mae_delta':0,'stop_rate_delta':0,'selected_fraction':0.1}
    paths = write_reports([s], [fold], [], args)
    assert Path(paths['json']).exists()
    assert json.loads(Path(paths['json']).read_text())['schema_version']
    assert Path(paths['folds']).exists()
    print('PASS test_walk_forward_long_alpha_family_d')


if __name__ == '__main__':
    main()
