#!/usr/bin/env python3
import json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(Path(__file__).parent))
from phase_o_overlay_test_fixture import fixture
from aegis_alpha.turbo.phase_o_overlay import apply_phase_o_overlay_to_active_manifest,preserve_phase_o_fields,resolve_phase_o_symbol_overlay,validate_phase_o_overlay
def main():
 root=Path(tempfile.mkdtemp()); base=fixture(root)
 raw={'symbol':'ETHUSDT','model_paths':{'long_30d':'keep-long','short_30d':'fresh-base'}}; out=apply_phase_o_overlay_to_active_manifest('ETHUSDT',raw,base)
 assert out['model_paths']['long_30d']=='keep-long'; assert '/phase_o_' in out['model_paths']['short_30d']; assert out['pre_phase_o_live_model_paths']['short_30d']=='fresh-base'; assert not validate_phase_o_overlay('ETHUSDT',out,base)
 link=apply_phase_o_overlay_to_active_manifest('LINKUSDT',{'model_paths':{'long_30d':'keep','short_30d':'base'}},base); assert link['phase_o_avoid_only'] and link['phase_o_link_entry_enabled'] is False; assert not any('/phase_o_' in str(v) for v in link['model_paths'].values()); assert not validate_phase_o_overlay('LINKUSDT',link,base)
 untouched={'model_paths':{'long_30d':'x'}}; assert apply_phase_o_overlay_to_active_manifest('OTHERUSDT',untouched,base)==untouched
 preserved=preserve_phase_o_fields({'phase_o_live_enabled':True,'pre_phase_o_live_model_paths':{'short_30d':'old'}},{'model_paths':{'short_30d':'new'}}); assert preserved['phase_o_live_enabled']; assert preserved['pre_phase_o_live_model_paths']['short_30d']=='old'
 overlay=resolve_phase_o_symbol_overlay('ETHUSDT',base); Path(overlay['phase_o_model_path']).unlink()
 try: apply_phase_o_overlay_to_active_manifest('ETHUSDT',raw,base); raise AssertionError('missing joblib should fail')
 except FileNotFoundError: pass
 retrain=(ROOT/'aegis_alpha/tools/run_turbo_scheduled_retrain.py').read_text(); assert 'apply_phase_o_overlay_to_active_manifest' in retrain and '--disable-phase-o-overlay' in retrain
 print('PASS test_phase_o_overlay')
if __name__=='__main__': main()
