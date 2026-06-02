#!/usr/bin/env python3
import json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(Path(__file__).parent)); sys.path.insert(0,str(ROOT/'aegis_alpha/tools'))
from phase_o_overlay_test_fixture import ALL,fixture
from reapply_phase_o_overlay_o71 import apply_repair
def main():
 root=Path(tempfile.mkdtemp()); base=fixture(root); backup=root/'backups'
 before=(base/'ETHUSDT/active_manifest.json').read_text(); dry=apply_repair(list(ALL),base,backup,apply=False,strict=True); assert 'ETHUSDT' in dry['repaired_symbols']; assert (base/'ETHUSDT/active_manifest.json').read_text()==before; assert not backup.exists()
 applied=apply_repair(list(ALL),base,backup,apply=True,strict=True); assert len(applied['repaired_symbols'])==11; assert applied['long_paths_intact'] and applied['link_no_entry']; assert Path(applied['backup_dir'],'backup_index.json').is_file(); eth=json.loads((base/'ETHUSDT/active_manifest.json').read_text()); assert '/phase_o_' in eth['model_paths']['short_30d']; assert eth['model_paths']['long_30d']=='ETHUSDT-long'
 second=apply_repair(list(ALL),base,backup,apply=True,strict=True); assert not second['repaired_symbols']; assert len(second['already_correct_symbols'])==11
 print('PASS test_reapply_phase_o_overlay_o71')
if __name__=='__main__': main()
