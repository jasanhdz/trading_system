#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,csv
from datetime import datetime,timezone
from pathlib import Path
from typing import Any,Optional,Sequence
if __package__ is None or __package__=='': sys.path.append(str(Path(__file__).resolve().parents[2]))
from aegis_alpha.turbo.snapshot_utils import turbo_symbol_model_dir # noqa:E402
SYMBOLS=["LTCUSDT","AVAXUSDT","ETHUSDT","SUIUSDT","ADAUSDT","DOGEUSDT","BTCUSDT","BNBUSDT","XRPUSDT","SOLUSDT","LINKUSDT"]
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def read_json(p:Path)->dict[str,Any]: return json.loads(p.read_text()) if p.exists() else {}
def write_json(p:Path,d:dict[str,Any]): p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def rollback(*,backup_stamp:str|None,apply:bool,out_dir:Path)->dict[str,Any]:
    rs=stamp(); rows=[]
    for s in SYMBOLS:
        p=turbo_symbol_model_dir(s)/'active_manifest.json'
        if not p.exists():
            rows.append({'symbol':s,'active_manifest':str(p),'restored_from_pre_phase_o':False,'before_phase_o_paths':0,'apply':apply,'status':'missing_active_manifest'})
            continue
        d=read_json(p); before=dict(d.get('model_paths') or {}); prev=d.get('pre_phase_o_live_model_paths')
        restored=False
        if isinstance(prev,dict):
            d['model_paths']=prev; restored=True
        d['phase_o_live_enabled']=False; d['phase_o_live_mode']='rolled_back'; d['phase_o_live_rolled_back_at']=now(); d['phase_o_live_rollback_backup_stamp']=backup_stamp or d.get('phase_o_live_backup_stamp')
        d['phase_o_link_entry_enabled']=False if s=='LINKUSDT' else d.get('phase_o_link_entry_enabled', False)
        if apply: write_json(p,d)
        rows.append({'symbol':s,'active_manifest':str(p),'restored_from_pre_phase_o':restored,'before_phase_o_paths':sum('/phase_o_' in str(v) for v in before.values()),'apply':apply,'status':'ok'})
    report={'schema_version':'aegis_phase_o2_live_rollback_v1','created_at':now(),'run_stamp':rs,'apply':apply,'backup_stamp':backup_stamp,'rows':rows,'manual_orders_sent':False}
    out_dir.mkdir(parents=True,exist_ok=True); write_json(out_dir/f'aegis_phase_o2_live_rollback_{rs}.json',report); (out_dir/f'aegis_phase_o2_live_rollback_plan_{rs}.md').write_text(f"# Phase O2 Rollback {rs}\n\n- apply: {apply}\n- backup_stamp: {backup_stamp}\n",encoding='utf-8')
    return report
def main(argv:Optional[Sequence[str]]=None)->int:
    p=argparse.ArgumentParser(); p.add_argument('--backup-stamp'); p.add_argument('--apply',action='store_true'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--out-dir',default='/home/jasan/Develop'); a=p.parse_args(argv); r=rollback(backup_stamp=a.backup_stamp,apply=bool(a.apply and not a.dry_run),out_dir=Path(a.out_dir)); print(json.dumps({'apply':r['apply'],'run_stamp':r['run_stamp']},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
