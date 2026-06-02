#!/usr/bin/env python3
import json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(Path(__file__).parent)); sys.path.insert(0,str(ROOT/'aegis_alpha/tools'))
from phase_o_overlay_test_fixture import fixture
from aegis_alpha.turbo.phase_o_overlay import apply_phase_o_overlay_to_active_manifest
from audit_phase_o_retrain_manifest_drift_o71 import audit
def dump(path,payload): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload))
def main():
 root=Path(tempfile.mkdtemp()); base=fixture(root); tools=root/'aegis_alpha/tools'; tools.mkdir(exist_ok=True); (tools/'run_turbo_scheduled_retrain.py').write_text("manifest = {}\natomic_write_json(manifest_path, manifest)\napply_phase_o_overlay_to_active_manifest\n--disable-phase-o-overlay")
 dump(root/'aegis_alpha/logs/turbo_retrain/turbo_retrain_20260602T002001Z.json',{'started_at':'x','promoted_symbols':['ETHUSDT']})
 first=audit(root,include_pm2=False); assert 'ETHUSDT' in first['drifted_symbols']; assert first['artifact_files_all_exist']; assert first['writer']['overlay_integrated']
 path=base/'ETHUSDT/active_manifest.json'; fixed=apply_phase_o_overlay_to_active_manifest('ETHUSDT',json.loads(path.read_text()),base); dump(path,fixed); second=audit(root,include_pm2=False); assert 'ETHUSDT' not in second['drifted_symbols']
 print('PASS test_audit_phase_o_retrain_manifest_drift_o71')
if __name__=='__main__': main()
