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
    classify_d_symbol,
    configs_from_args,
    phase_slug,
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
    assert classify_d_symbol(summary(family='slow_trend_long'), 'slow_trend_long') == 'LONG_D2_CONFIRMED'
    micro = summary(family='micro_roe_momentum_long', horizon=6, mean_time_to_target=3.0, fast_hit_lift=0.02, fast_hit_rate=0.08)
    assert classify_d_symbol(micro, 'micro_roe_momentum_long') == 'LONG_D3_CONFIRMED'
    slow_micro = dict(micro, mean_time_to_target=5.0)
    assert classify_d_symbol(slow_micro, 'micro_roe_momentum_long') == 'LONG_D3_FAILED'
    risky_micro = dict(micro, mean_stop_rate_delta=0.09)
    assert classify_d_symbol(risky_micro, 'micro_roe_momentum_long') == 'LONG_D3_FAILED'
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
    args.use_cache = True
    args.cache_max_items = 8
    cfgs = configs_from_args(args)
    assert cfgs == [WalkConfig('ETHUSDT', ALLOWED_FAMILY, 'long_hit3_before_minus2', 24)]
    args.family = 'slow_trend_long'
    cfgs = configs_from_args(args)
    assert cfgs == [WalkConfig('ETHUSDT', 'slow_trend_long', 'long_hit3_before_minus2', 24)]
    assert phase_slug('slow_trend_long') == 'd2_slowtrend'
    args.family = 'micro_roe_momentum_long'
    cfgs = configs_from_args(args)
    assert cfgs == [WalkConfig('ETHUSDT', 'micro_roe_momentum_long', 'long_hit3_before_minus2', 24)]
    assert phase_slug('micro_roe_momentum_long') == 'd3_micro_roe'
    args.family = 'breakout_momentum_long'
    try:
        configs_from_args(args)
        raise AssertionError('expected family restriction')
    except SystemExit:
        pass
    out = Path(tempfile.mkdtemp())
    args = type('Args', (), {'out_dir': str(out), 'symbols': 'ETHUSDT', 'family': ALLOWED_FAMILY, 'target': 'long_hit3_before_minus2', 'horizon': 24, 'fold_count': 4, 'lookback_days': 1, 'model_dir': str(out/'models/research'), 'db_path': 'x', 'feature_mode': 'selected_family', 'fast': True, 'save_models': False, 'no_save_models': True, 'min_train_samples': 1, 'min_test_samples': 1, 'include_secondary': False, 'use_cache': True, 'cache_max_items': 8})()
    s = summary(d1_status='LONG_D1_CONFIRMED', score=score_summary(summary()), recommendation='pass_to_frozen_confirmation')
    fold = {'symbol':'ETHUSDT','family':ALLOWED_FAMILY,'target':'long_hit3_before_minus2','horizon':24,'fold_index':1,'fold_status':'POSITIVE','hit_lift':0.1,'net_quality_lift_after_costs':0.1,'p90_mae_delta':0,'stop_rate_delta':0,'selected_fraction':0.1}
    paths = write_reports([s], [fold], [], args)
    assert Path(paths['json']).exists()
    assert json.loads(Path(paths['json']).read_text())['schema_version']
    assert Path(paths['folds']).exists()
    args.family = 'slow_trend_long'
    s2 = dict(s, family='slow_trend_long', d_status='LONG_D2_CONFIRMED', d1_status=None, d2_status='LONG_D2_CONFIRMED')
    paths2 = write_reports([s2], [dict(fold, family='slow_trend_long')], [], args)
    assert 'd2_slowtrend' in paths2['json']
    args.family = 'micro_roe_momentum_long'
    s3 = dict(s, family='micro_roe_momentum_long', d_status='LONG_D3_CONFIRMED', d1_status=None, d2_status=None, d3_status='LONG_D3_CONFIRMED', mean_time_to_target=3.0, fast_hit_rate=0.08, fast_hit_lift=0.02, late_entry_rate_selected=0.1)
    paths3 = write_reports([s3], [dict(fold, family='micro_roe_momentum_long')], [], args)
    assert 'd3_micro_roe' in paths3['json']
    print('PASS test_walk_forward_long_alpha_family_d')


if __name__ == '__main__':
    main()
