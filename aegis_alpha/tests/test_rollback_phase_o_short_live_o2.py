from __future__ import annotations
import json, sys, tempfile
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from aegis_alpha.tools import rollback_phase_o_short_live_o2 as rb

def test_rollback_restores_model_paths():
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td); orig=rb.turbo_symbol_model_dir; rb.turbo_symbol_model_dir=lambda s: tmp/s
        try:
            d=tmp/'LTCUSDT'; d.mkdir(); (d/'active_manifest.json').write_text(json.dumps({'model_paths':{'short_14d':'phase'},'pre_phase_o_live_model_paths':{'short_14d':'old'},'phase_o_live_enabled':True}))
            rb.rollback(backup_stamp='s',apply=True,out_dir=tmp)
            out=json.load(open(d/'active_manifest.json'))
            assert out['model_paths']['short_14d']=='old'
            assert out['phase_o_live_enabled'] is False
            assert out['phase_o_live_mode']=='rolled_back'
        finally: rb.turbo_symbol_model_dir=orig
if __name__=='__main__': test_rollback_restores_model_paths(); print('manual_rollback_phase_o_short_live_o2_tests_passed')
