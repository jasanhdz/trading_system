#!/usr/bin/env python3
import json,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from aegis_alpha.tools.profile_long_alpha_families_a import (
    compute_long_target_arrays, compute_micro_roe_long_targets,
    classify_long_family_candidate, select_best_by_symbol, write_csv, write_reports,
)

def row(**kw):
    base={
        'symbol':'ETHUSDT','side':'LONG','alpha_family':'momentum_burst_long','target_name':'long_hit3_before_minus2','horizon_candles':6,
        'sample_count':1000,'selected_fraction':0.10,'selected_count':100,
        'hit_lift':0.08,'quality_lift':0.02,'net_quality_lift_after_costs':0.02,
        'p90_mae_delta':0.0,'stop_delta':0.0,'time_to_target_avg':3.0,
    }
    base.update(kw); return base

def main():
    close=np.array([100,100,100,100,100,100,100],dtype=float)
    high=np.array([100,103.2,101,100,100,100,100],dtype=float)
    low=np.array([100,99,99,99,99,99,99],dtype=float)
    t=compute_long_target_arrays(close,high,low,0.03,0.02,3)
    assert t.hit[0]==1 and t.stop[0]==0 and t.time_to_target[0]==1
    high=np.array([100,103.5,100,100,100,100,100],dtype=float)
    low=np.array([100,97.5,100,100,100,100,100],dtype=float)
    t=compute_long_target_arrays(close,high,low,0.03,0.02,3)
    assert t.hit[0]==0 and t.stop[0]==1 and t.ambiguous[0]==1
    market={'close':close,'high':np.array([100,100.6,100,100,100,100,100],float),'low':np.array([100,99.8,100,100,100,100,100],float)}
    micro=compute_micro_roe_long_targets(market,20,0.10,0.06,3)
    assert micro.hit[0]==1  # +10% ROE at 20x == +0.5% price move
    assert classify_long_family_candidate(row())=='LONG_FAMILY_PROMISING'
    assert classify_long_family_candidate(row(time_to_target_avg=5.0))!='LONG_FAMILY_PROMISING'
    assert classify_long_family_candidate(row(net_quality_lift_after_costs=-0.01))=='LONG_FAMILY_FAILED'
    avoid=row(alpha_family='avoid_only_bad_long_filter',selected_fraction=0.15,avoid_selected_fraction=0.15,avoid_quality_delta=-0.01,avoid_stop_rate_delta=0.05,avoid_p90_mae_delta=0.10,avoid_hit_rate_delta=-0.03,avoid_usefulness_score=0.10)
    assert classify_long_family_candidate(avoid)=='LONG_FAMILY_AVOID_ONLY'
    failed=row(symbol='BTCUSDT',family_status='LONG_FAMILY_FAILED',net_quality_lift_after_costs=-1)
    prom=row(symbol='BTCUSDT',family_status='LONG_FAMILY_PROMISING',net_quality_lift_after_costs=.1,hit_lift=.1,p90_mae_delta=0)
    assert select_best_by_symbol([failed,prom])[0]['family_status']=='LONG_FAMILY_PROMISING'
    link_avoid=dict(avoid,symbol='LINKUSDT',family_status='LONG_FAMILY_AVOID_ONLY')
    assert select_best_by_symbol([link_avoid])[0]['symbol']=='LINKUSDT'
    out=Path(tempfile.mkdtemp())
    write_csv(out/'x.csv',[prom])
    args=type('Args',(),{'out_dir':str(out),'symbols':('BTCUSDT',),'lookback_days':1,'family_group':'all'})()
    paths=write_reports([prom,link_avoid],args)
    assert Path(paths['json']).exists() and json.loads(Path(paths['json']).read_text())['schema_version']
    assert Path(paths['all_configs']).exists()
    print('PASS test_profile_long_alpha_families_a')
if __name__=='__main__': main()
