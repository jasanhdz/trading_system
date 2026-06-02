#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_phase_o5_fraction_alignment import alignment_rows

def test_alignment_rows_inherit_ml_buckets_without_phase_duplication():
 cfg={'aegis':{'phase_o_short_live':{'symbols':{'LTCUSDT':{'enabled':True}}},'turbo':{'position_fraction_cap':.20}}}
 row={r['symbol']:r for r in alignment_rows(cfg,20)}['LTCUSDT']
 assert row['phase_o_yaml_fraction'] is None and row['fraction_source']=='raw_ml_turbo_bucket'
 assert row['wallet_20_margin_conservative']==1.6 and row['wallet_20_margin_normal']==2.4
 assert round(row['notional_25x_premium'],8)==90

def test_alignment_rows_reports_legacy_duplicate_when_present():
 cfg={'aegis':{'phase_o_short_live':{'max_position_fraction_default':.01,'symbols':{}},'turbo':{'position_fraction_cap':.01}}}
 assert alignment_rows(cfg,20)[0]['phase_o_yaml_fraction']==.01

def main():
 test_alignment_rows_inherit_ml_buckets_without_phase_duplication(); test_alignment_rows_reports_legacy_duplicate_when_present(); print('PASS test_audit_phase_o5_fraction_alignment')
if __name__=='__main__': main()
