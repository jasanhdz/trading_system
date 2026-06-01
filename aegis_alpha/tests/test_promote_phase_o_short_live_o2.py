from __future__ import annotations
import json, sys, tempfile
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from aegis_alpha.tools import promote_phase_o_short_live_o2 as pro

def make_symbol(tmp,s='LTCUSDT',look=14):
    d=tmp/'models'/s; act=d/'active'; art=d/'backups'/'b'/f'phase_o_20260601T070114Z'; art.mkdir(parents=True)
    (act).mkdir(parents=True)
    active={'model_paths':{'long_14d':'long_old','short_14d':'short_old','short_30d':'short30_old'},'symbol':s}
    (d/'active_manifest.json').write_text(json.dumps(active))
    (art/'symbol_shadow_manifest.json').write_text(json.dumps({'symbol':s,'entry_enabled':s!='LINKUSDT','avoid_only':s=='LINKUSDT','shadow_type':'avoid_only_filter' if s=='LINKUSDT' else 'entry_model'}))
    if s=='LINKUSDT':
        for n in ['micro_hit_classifier.joblib','micro_quality_regressor.joblib','micro_danger_classifier.joblib']:(art/n).write_text('x')
    else:(art/f'turbo_short_edge_{look}d_phase_o_20260601T070114Z.joblib').write_text('x')
    return d

def test_dry_run_no_modify_and_apply_short_only():
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td); orig=pro.turbo_symbol_model_dir; pro.turbo_symbol_model_dir=lambda s: tmp/'models'/s
        try:
            make_symbol(tmp)
            before=(tmp/'models/LTCUSDT/active_manifest.json').read_text()
            pro.promote(['LTCUSDT'],apply=False,out_dir=tmp,backup_dir=tmp/'bkp',phase_o_stamp='20260601T070114Z',require_model_files=True)
            assert (tmp/'models/LTCUSDT/active_manifest.json').read_text()==before
            pro.promote(['LTCUSDT'],apply=True,out_dir=tmp,backup_dir=tmp/'bkp',phase_o_stamp='20260601T070114Z',require_model_files=True)
            d=json.load(open(tmp/'models/LTCUSDT/active_manifest.json'))
            assert d['pre_phase_o_live_model_paths']['long_14d']=='long_old'
            assert d['model_paths']['long_14d']=='long_old'
            assert '/phase_o_' in d['model_paths']['short_14d']
        finally: pro.turbo_symbol_model_dir=orig

def test_link_no_entry_and_missing_file_fails():
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td); orig=pro.turbo_symbol_model_dir; pro.turbo_symbol_model_dir=lambda s: tmp/'models'/s
        try:
            make_symbol(tmp,'LINKUSDT',30)
            pro.promote(['LINKUSDT'],apply=True,out_dir=tmp,backup_dir=tmp/'bkp',phase_o_stamp='20260601T070114Z',require_model_files=True)
            d=json.load(open(tmp/'models/LINKUSDT/active_manifest.json'))
            assert d['phase_o_link_entry_enabled'] is False
            assert not any('/phase_o_' in str(v) for k,v in d['model_paths'].items() if k.startswith('short_'))
            (tmp/'models/LINKUSDT/backups/b/phase_o_20260601T070114Z/micro_hit_classifier.joblib').unlink()
            try: pro.load_phase_o_artifacts(['LINKUSDT'],'20260601T070114Z',True)
            except SystemExit: pass
            else: raise AssertionError('missing file accepted')
        finally: pro.turbo_symbol_model_dir=orig

def test_unknown_symbol_fails():
    try: pro.select_symbols('FAKEUSDT')
    except SystemExit: pass
    else: raise AssertionError('unknown symbol accepted')
if __name__=='__main__':
    test_dry_run_no_modify_and_apply_short_only(); test_link_no_entry_and_missing_file_fails(); test_unknown_symbol_fails(); print('manual_promote_phase_o_short_live_o2_tests_passed')
