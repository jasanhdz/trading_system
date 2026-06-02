#!/usr/bin/env python3
"""Audit Phase O SHORT manifest drift caused by scheduled retrain promotions."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from aegis_alpha.turbo.phase_o_overlay import ENTRY_SYMBOLS,AVOID_ONLY_SYMBOLS,PHASE_O_SYMBOLS,resolve_phase_o_symbol_overlay,validate_phase_o_overlay
MODEL_ROOT=ROOT/'aegis_alpha/models/turbo'
RETRAIN=ROOT/'aegis_alpha/tools/run_turbo_scheduled_retrain.py'
LOG_DIR=ROOT/'aegis_alpha/logs/turbo_retrain'
def nowstamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def load(p): return json.loads(Path(p).read_text())
def latest_retrain(root=ROOT):
 files=sorted((Path(root)/'aegis_alpha/logs/turbo_retrain').glob('turbo_retrain_*.json'))
 return (files[-1],load(files[-1])) if files else (None,{})
def pm2_retrain():
 try: return subprocess.check_output(['pm2','describe','07-Aegis-Turbo-Retrain'],text=True,stderr=subprocess.STDOUT,timeout=10)
 except Exception as e: return f'unavailable: {e!r}'
def audit(root=ROOT,include_pm2=True):
 root=Path(root); model_root=root/'aegis_alpha/models/turbo'; retrain=root/'aegis_alpha/tools/run_turbo_scheduled_retrain.py'; src=retrain.read_text(); latest_path,latest=latest_retrain(root); rows=[]
 for symbol in PHASE_O_SYMBOLS:
  path=model_root/symbol/'active_manifest.json'; row={'symbol':symbol,'active_manifest_path':str(path),'exists':path.is_file(),'entry_symbol':symbol in ENTRY_SYMBOLS,'avoid_only_symbol':symbol in AVOID_ONLY_SYMBOLS,'phase_o_live_enabled':False,'phase_o_avoid_only':False,'phase_o_link_entry_enabled':None,'phase_o_overlay_persistence_enabled':False,'phase_o_paths':[],'expected_phase_o_model_path':'','expected_short_key':'','artifact_files_exist':False,'missing_phase_o_fields':[],'validation_errors':[],'drifted':True}
  try:
   overlay=resolve_phase_o_symbol_overlay(symbol,model_root); row['expected_short_key']=overlay['short_window_key']; row['expected_phase_o_model_path']=overlay.get('phase_o_model_path',''); row['artifact_files_exist']=all(Path(x).is_file() for x in overlay['model_files'])
   m=load(path); paths=m.get('model_paths') or {}; row.update({'phase_o_live_enabled':bool(m.get('phase_o_live_enabled')),'phase_o_avoid_only':bool(m.get('phase_o_avoid_only')),'phase_o_link_entry_enabled':m.get('phase_o_link_entry_enabled'),'phase_o_overlay_persistence_enabled':bool(m.get('phase_o_overlay_persistence_enabled')),'phase_o_paths':[str(v) for k,v in paths.items() if k.startswith('short_') and '/phase_o_' in str(v)]})
   required=['phase_o_overlay_persistence_enabled','phase_o_live_artifact_stamp']
   row['missing_phase_o_fields']=[k for k in required if not m.get(k)]
   row['validation_errors']=validate_phase_o_overlay(symbol,m,model_root)
   row['drifted']=bool(row['validation_errors'])
  except Exception as e: row['validation_errors']=[repr(e)]
  rows.append(row)
 drifted=[r['symbol'] for r in rows if r['drifted']]
 promoted=latest.get('promoted_symbols') or []
 writer={'writer_script':str(retrain),'reconstructs_manifest_literal':'manifest = {' in src and 'atomic_write_json(manifest_path, manifest)' in src,'overlay_integrated':'apply_phase_o_overlay_to_active_manifest' in src,'disable_overlay_cli':'--disable-phase-o-overlay' in src,'latest_retrain_path':str(latest_path) if latest_path else None,'latest_retrain_started_at':latest.get('started_at'),'latest_promoted_symbols':promoted,'pm2_process':'07-Aegis-Turbo-Retrain','pm2_describe':pm2_retrain() if include_pm2 else 'skipped'}
 return {'schema_version':'aegis_phase_o71_retrain_drift_audit_v1','created_at':datetime.now(timezone.utc).isoformat(),'writer':writer,'rows':rows,'drifted_symbols':drifted,'drift_count':len(drifted),'artifact_files_all_exist':all(r['artifact_files_exist'] for r in rows),'link_avoid_only_safe':not next(r for r in rows if r['symbol']=='LINKUSDT')['validation_errors'],'root_cause':'run_turbo_scheduled_retrain.promote_candidate reconstructed active_manifest.json from base candidate paths without reapplying Phase O overlay; scheduled promotion overwrote Phase O metadata and SHORT paths.'}
def reports(payload,out):
 out=Path(out); out.mkdir(parents=True,exist_ok=True); stamp=nowstamp(); base=out/f'aegis_phase_o71_retrain_drift_audit_{stamp}'; jp=base.with_suffix('.json'); mp=base.with_suffix('.md'); cp=out/f'aegis_phase_o71_retrain_drift_symbols_{stamp}.csv'; jp.write_text(json.dumps(payload,indent=2)+'\n')
 fields=['symbol','entry_symbol','avoid_only_symbol','exists','phase_o_live_enabled','phase_o_avoid_only','phase_o_link_entry_enabled','phase_o_overlay_persistence_enabled','expected_short_key','expected_phase_o_model_path','artifact_files_exist','drifted','missing_phase_o_fields','validation_errors','phase_o_paths']
 with cp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
  for r in payload['rows']: w.writerow({k:(json.dumps(r[k]) if isinstance(r[k],list) else r[k]) for k in fields})
 lines=['# Phase O.7.1 Retrain Manifest Drift Audit','',f"Drifted symbols: **{len(payload['drifted_symbols'])}**",f"Artifact files exist: **{payload['artifact_files_all_exist']}**",f"Overlay integrated in retrain: **{payload['writer']['overlay_integrated']}**",'', '## Root Cause',payload['root_cause'],'','## Symbols','| Symbol | Entry | Avoid only | Phase O live | Overlay persistence | Drifted | Errors |','|---|---:|---:|---:|---:|---:|---|']
 for r in payload['rows']: lines.append(f"| {r['symbol']} | {r['entry_symbol']} | {r['avoid_only_symbol']} | {r['phase_o_live_enabled']} | {r['phase_o_overlay_persistence_enabled']} | {r['drifted']} | {', '.join(r['validation_errors']) or 'none'} |")
 mp.write_text('\n'.join(lines)+'\n'); return [str(mp),str(jp),str(cp)]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out-dir',default='/home/jasan/Develop'); a=ap.parse_args(); payload=audit(); paths=reports(payload,a.out_dir); print(json.dumps({'reports':paths,'drifted_symbols':payload['drifted_symbols'],'artifact_files_all_exist':payload['artifact_files_all_exist'],'overlay_integrated':payload['writer']['overlay_integrated']},indent=2))
if __name__=='__main__': main()
