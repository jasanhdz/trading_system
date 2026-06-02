#!/usr/bin/env python3
"""Reapply persistent Phase O SHORT overlay to current manifests with backups."""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,os,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from aegis_alpha.turbo.phase_o_overlay import PHASE_O_SYMBOLS,apply_phase_o_overlay_to_active_manifest,validate_phase_o_overlay
MODEL_ROOT=ROOT/'aegis_alpha/models/turbo'
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def load(p): return json.loads(Path(p).read_text())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,payload):
 path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)
def parse_symbols(raw):
 if raw.upper()=='ALL': return list(PHASE_O_SYMBOLS)
 values=[x.strip().upper() for x in raw.split(',') if x.strip()]; bad=sorted(set(values)-set(PHASE_O_SYMBOLS))
 if bad: raise ValueError(f'unknown Phase O symbols: {bad}')
 return list(dict.fromkeys(values))
def apply_repair(symbols,base_model_dir=MODEL_ROOT,backup_dir='/home/jasan/Develop/phase_o71_overlay_backups',apply=False,strict=False):
 base=Path(base_model_dir); started=stamp(); planned=[]; errors=[]
 for symbol in symbols:
  path=base/symbol/'active_manifest.json'
  try:
   before=load(path); before_errors=validate_phase_o_overlay(symbol,before,base)
   if before_errors:
    after=apply_phase_o_overlay_to_active_manifest(symbol,before,base)
   else:
    after=copy.deepcopy(before)
   after_errors=validate_phase_o_overlay(symbol,after,base)
   long_changed=[k for k,v in (before.get('model_paths') or {}).items() if str(k).startswith('long_') and (after.get('model_paths') or {}).get(k)!=v]
   if long_changed: after_errors.append(f'long_paths_changed:{long_changed}')
   row={'symbol':symbol,'active_manifest_path':str(path),'before_sha256':digest(path),'before_drifted':bool(before_errors),'before_errors':before_errors,'after_errors':after_errors,'changed':before!=after,'status':'planned_repair' if before_errors else 'already_correct','long_paths_intact':not long_changed,'phase_o_paths_after':[str(v) for k,v in (after.get('model_paths') or {}).items() if str(k).startswith('short_') and '/phase_o_' in str(v)],'_after':after}
   if after_errors: errors.append(f'{symbol}: {after_errors}')
   planned.append(row)
  except Exception as e: errors.append(f'{symbol}: {e!r}')
 if errors and strict: raise RuntimeError('; '.join(errors))
 backup_stamp=None; backup_index=[]
 if apply and not errors:
  backup_stamp=started; root=Path(backup_dir)/started
  for row in planned:
   src=Path(row['active_manifest_path']); dst=root/src.parent.name/'active_manifest.json'; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst); backup_index.append({'original_path':str(src),'backup_path':str(dst),'sha256_before':row['before_sha256'],'symbol':row['symbol'],'reason':'phase_o71_overlay_reapply'})
  atomic(root/'backup_index.json',{'schema_version':'aegis_phase_o71_overlay_backup_v1','created_at':datetime.now(timezone.utc).isoformat(),'items':backup_index})
  for row in planned:
   if row['changed']: atomic(Path(row['active_manifest_path']),row['_after']); row['status']='repaired'
   else: row['status']='already_correct'
 for row in planned: row.pop('_after',None)
 return {'schema_version':'aegis_phase_o71_overlay_reapply_v1','created_at':datetime.now(timezone.utc).isoformat(),'apply':apply,'dry_run':not apply,'backup_stamp':backup_stamp,'backup_dir':str(Path(backup_dir)/started) if apply else None,'errors':errors,'rows':planned,'repaired_symbols':[r['symbol'] for r in planned if r['status']=='repaired' or (not apply and r['status']=='planned_repair')],'already_correct_symbols':[r['symbol'] for r in planned if r['status']=='already_correct'],'long_paths_intact':all(r['long_paths_intact'] for r in planned),'link_no_entry':not any('/phase_o_' in x for r in planned if r['symbol']=='LINKUSDT' for x in r['phase_o_paths_after'])}
def reports(payload,out):
 out=Path(out); out.mkdir(parents=True,exist_ok=True); st=stamp(); base=out/f'aegis_phase_o71_retrain_overlay_fix_{st}'; jp=base.with_suffix('.json'); mp=base.with_suffix('.md'); cp=out/f'aegis_phase_o71_reapply_summary_{st}.csv'; jp.write_text(json.dumps(payload,indent=2)+'\n')
 fields=['symbol','status','before_drifted','before_errors','after_errors','changed','long_paths_intact','phase_o_paths_after','active_manifest_path']
 with cp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
  for r in payload['rows']: w.writerow({k:(json.dumps(r[k]) if isinstance(r[k],list) else r[k]) for k in fields})
 lines=['# Phase O.7.1 Overlay Reapply','',f"Apply: **{payload['apply']}**",f"Errors: **{len(payload['errors'])}**",f"LONG paths intact: **{payload['long_paths_intact']}**",f"LINK no entry: **{payload['link_no_entry']}**",f"Backup: `{payload.get('backup_dir')}`",'', '## Symbols','| Symbol | Status | Drifted before | Changed | LONG intact |','|---|---|---:|---:|---:|']
 for r in payload['rows']: lines.append(f"| {r['symbol']} | {r['status']} | {r['before_drifted']} | {r['changed']} | {r['long_paths_intact']} |")
 mp.write_text('\n'.join(lines)+'\n'); return [str(mp),str(jp),str(cp)]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--symbols',default='ALL'); ap.add_argument('--base-model-dir',default=str(MODEL_ROOT)); ap.add_argument('--backup-dir',default='/home/jasan/Develop/phase_o71_overlay_backups'); ap.add_argument('--out-dir',default='/home/jasan/Develop'); g=ap.add_mutually_exclusive_group(); g.add_argument('--apply',action='store_true'); g.add_argument('--dry-run',action='store_true'); ap.add_argument('--strict',action='store_true'); a=ap.parse_args(); payload=apply_repair(parse_symbols(a.symbols),a.base_model_dir,a.backup_dir,apply=a.apply,strict=a.strict); paths=reports(payload,a.out_dir); print(json.dumps({'reports':paths,'apply':payload['apply'],'backup_dir':payload['backup_dir'],'repaired_symbols':payload['repaired_symbols'],'already_correct_symbols':payload['already_correct_symbols'],'errors':payload['errors'],'long_paths_intact':payload['long_paths_intact'],'link_no_entry':payload['link_no_entry']},indent=2)); raise SystemExit(0 if not payload['errors'] else 2)
if __name__=='__main__': main()
