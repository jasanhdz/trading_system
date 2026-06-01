from __future__ import annotations
import json, sys, tempfile
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from aegis_alpha.tools import preflight_phase_o_short_live_o2 as pf

def setup(tmp):
    for s in pf.ALL:
        d=tmp/'models'/s; d.mkdir(parents=True); mp={'long_30d':'long'}
        if s in pf.ENTRY:
            k=f'short_{pf.LOOK[s]}d'; m=d/f'phase_o_{s}.joblib'; m.write_text('x'); mp[k]=str(m)
            data={'model_paths':mp,'pre_phase_o_live_model_paths':{'long_30d':'long',k:'old'},'phase_o_live_enabled':True}
        else:
            a=d/'avoid.joblib'; a.write_text('x'); data={'model_paths':{'short_30d':'old'},'phase_o_link_entry_enabled':False,'phase_o_avoid_only':True,'phase_o_avoid_artifacts':[str(a)]}
        (d/'active_manifest.json').write_text(json.dumps(data))
    y=tmp/'cfg.yaml'; y.write_text('aegis:\n  turbo:\n    enabled: true\n    live_enabled: true\n    allow_short: true\n    position_fraction_cap: 0.01\n    max_trades_per_day: 3\n    require_brackets: true\n  phase_o_short_live:\n    allow_link_entry: false\n    link_avoid_only: true\n')
    return y

def test_preflight_pass_and_fail_cases():
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td); orig=pf.turbo_symbol_model_dir; pf.turbo_symbol_model_dir=lambda s: tmp/'models'/s
        try:
            y=setup(tmp); r=pf.preflight(tmp,y); assert r['status']=='PASSED', r['errors']
            bad=y.read_text().replace('require_brackets: true','require_brackets: false'); y.write_text(bad); r=pf.preflight(tmp,y); assert r['status']=='FAILED' and any('require_brackets' in e for e in r['errors'])
            y.write_text(bad.replace('require_brackets: false','require_brackets: true').replace('position_fraction_cap: 0.01','position_fraction_cap: 0.02')); r=pf.preflight(tmp,y); assert any('position_fraction_cap' in e for e in r['errors'])
            link=json.load(open(tmp/'models/LINKUSDT/active_manifest.json')); link['phase_o_link_entry_enabled']=True; (tmp/'models/LINKUSDT/active_manifest.json').write_text(json.dumps(link)); r=pf.preflight(tmp,y); assert any('LINKUSDT:entry' in e for e in r['errors'])
        finally: pf.turbo_symbol_model_dir=orig
if __name__=='__main__': test_preflight_pass_and_fail_cases(); print('manual_preflight_phase_o_short_live_o2_tests_passed')
