#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, hashlib, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
from aegis_alpha.turbo.snapshot_utils import turbo_symbol_model_dir, normalize_turbo_symbol  # noqa:E402

ENTRY_SYMBOLS=["LTCUSDT","AVAXUSDT","ETHUSDT","SUIUSDT","ADAUSDT","DOGEUSDT","BTCUSDT","BNBUSDT","XRPUSDT","SOLUSDT"]
AVOID_ONLY_SYMBOLS=["LINKUSDT"]
ALL_SYMBOLS=ENTRY_SYMBOLS+AVOID_ONLY_SYMBOLS
EXPECTED_LOOKBACK={"LTCUSDT":14,"AVAXUSDT":14,"ETHUSDT":30,"SUIUSDT":7,"ADAUSDT":30,"DOGEUSDT":30,"BTCUSDT":30,"BNBUSDT":30,"XRPUSDT":30,"SOLUSDT":30}
POSITION_FRACTIONS={"LTCUSDT":0.01,"AVAXUSDT":0.01,"ETHUSDT":0.0075,"SUIUSDT":0.0075,"ADAUSDT":0.01,"DOGEUSDT":0.0075,"BTCUSDT":0.005,"BNBUSDT":0.01,"XRPUSDT":0.0075,"SOLUSDT":0.0075}
CAUTION={"BTCUSDT":"very_cautious","ETHUSDT":"cautious","DOGEUSDT":"cautious","XRPUSDT":"cautious","SOLUSDT":"final_repair_candidate"}

def repo_root()->Path: return Path(__file__).resolve().parents[2]
def stamp()->str: return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def now()->str: return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path:Path)->str|None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
def read_json(path:Path)->dict[str,Any]:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
def write_json(path:Path,payload:Mapping[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def select_symbols(raw:str)->list[str]:
    syms=ALL_SYMBOLS if raw.upper()=="ALL" else [s.strip().upper() for s in raw.split(',') if s.strip()]
    unknown=sorted(set(syms)-set(ALL_SYMBOLS))
    if unknown: raise SystemExit(f"unknown symbols: {unknown}")
    return syms

def find_symbol_artifact(symbol:str, phase_o_stamp:str)->Path|None:
    base=turbo_symbol_model_dir(symbol)
    candidates=list((base/'active').glob(f'phase_o_{phase_o_stamp}/symbol_shadow_manifest.json'))
    candidates+=list((base/'backups').glob(f'*/phase_o_{phase_o_stamp}/symbol_shadow_manifest.json'))
    return sorted(candidates)[-1] if candidates else None

def model_files_for(symbol:str, artifact_manifest:Path)->list[Path]:
    d=artifact_manifest.parent
    if symbol=='LINKUSDT':
        return [d/'micro_hit_classifier.joblib', d/'micro_quality_regressor.joblib', d/'micro_danger_classifier.joblib']
    return sorted(d.glob('turbo_short_edge_*phase_o_*.joblib'))

def load_phase_o_artifacts(symbols:Sequence[str], phase_o_stamp:str, require_model_files:bool=True)->dict[str,dict[str,Any]]:
    found={}
    for s in symbols:
        p=find_symbol_artifact(s, phase_o_stamp)
        if not p or not p.exists(): raise SystemExit(f'missing Phase O symbol artifact for {s} stamp={phase_o_stamp}')
        m=read_json(p); files=model_files_for(s,p)
        if require_model_files and (not files or any(not x.exists() for x in files)): raise SystemExit(f'missing Phase O model files for {s}: {[str(x) for x in files]}')
        if s=='LINKUSDT' and (m.get('entry_enabled') is not False or m.get('avoid_only') is not True): raise SystemExit('LINK artifact is not avoid-only safe')
        if s!='LINKUSDT' and m.get('entry_enabled') is not True: raise SystemExit(f'{s} artifact is not entry-enabled')
        found[s]={'manifest_path':p,'manifest':m,'model_files':files}
    return found

def backup_files(paths:Sequence[Path], backup_dir:Path, backup_stamp:str)->Path:
    root=repo_root(); out=backup_dir/backup_stamp; out.mkdir(parents=True,exist_ok=True); items=[]
    for src in paths:
        rel=src.relative_to(root) if src.is_relative_to(root) else Path(src.name)
        dst=out/rel; exists=src.exists(); checksum=sha(src) if exists else None
        if exists: dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
        items.append({'original_path':str(src),'backup_path':str(dst),'sha256_before':checksum,'file_exists':exists,'symbol':src.parent.name if src.name=='active_manifest.json' else None,'reason':'phase_o2_live'})
    idx=out/'backup_index.json'; write_json(idx, {'backup_stamp':backup_stamp,'created_at':now(),'items':items})
    return idx

def promote(symbols:Sequence[str], *, apply:bool, out_dir:Path, backup_dir:Path, phase_o_stamp:str, require_model_files:bool=True)->dict[str,Any]:
    artifacts=load_phase_o_artifacts(symbols, phase_o_stamp, require_model_files)
    run_stamp=stamp(); changed=[]; rows=[]; backup_index=''
    paths=[turbo_symbol_model_dir(s)/'active_manifest.json' for s in symbols]
    paths.append(repo_root()/'aegis_alpha/models/turbo/phase_o_short_manifest.json')
    gp=repo_root()/f'aegis_alpha/models/turbo/phase_o_global_short_manifest_{phase_o_stamp}.json'
    if gp.exists(): paths.append(gp)
    if apply: backup_index=str(backup_files(paths,backup_dir,run_stamp))
    for s in symbols:
        active_path=turbo_symbol_model_dir(s)/'active_manifest.json'; active=read_json(active_path)
        before=dict(active.get('model_paths') or {})
        after=dict(before)
        artifact=artifacts[s]
        if s in ENTRY_SYMBOLS:
            key=f"short_{EXPECTED_LOOKBACK[s]}d"; model=str(artifact['model_files'][0].resolve()); after[key]=model
            active.update({'phase_o_live_enabled':True,'phase_o_live_mode':'experimental_short_only','phase_o_live_artifact_stamp':f'phase_o_{phase_o_stamp}','phase_o_live_entry_symbols':ENTRY_SYMBOLS,'phase_o_live_avoid_only_symbols':AVOID_ONLY_SYMBOLS,'phase_o_live_requires_experimental_yaml':True,'phase_o_live_capital_profile':'test_capital','phase_o_live_max_position_fraction_default':0.01,'phase_o_live_kill_switch_supported':True})
            active.setdefault('phase_o_symbols',{})[s]={'entry_enabled':True,'avoid_only':False,'caution_level':CAUTION.get(s,'normal'),'symbol_manifest':str(artifact['manifest_path'].resolve()),'max_position_fraction':POSITION_FRACTIONS[s]}
        else:
            active.update({'phase_o_link_avoid_only_enabled':True,'phase_o_avoid_only':True,'phase_o_avoid_artifacts':[str(x.resolve()) for x in artifact['model_files']],'phase_o_link_entry_enabled':False,'phase_o_live_entry_symbols':ENTRY_SYMBOLS,'phase_o_live_avoid_only_symbols':AVOID_ONLY_SYMBOLS,'phase_o_live_requires_experimental_yaml':True,'phase_o_live_capital_profile':'test_capital','phase_o_live_kill_switch_supported':True})
            active.setdefault('phase_o_symbols',{})[s]={'entry_enabled':False,'avoid_only':True,'caution_level':'avoid_only','symbol_manifest':str(artifact['manifest_path'].resolve())}
        if 'pre_phase_o_live_model_paths' not in active: active['pre_phase_o_live_model_paths']=before
        active['pre_phase_o_live_updated_at']=now(); active['phase_o_live_backup_stamp']=run_stamp; active['model_paths']=after
        if apply: write_json(active_path, active); changed.append(str(active_path))
        rows.append({'symbol':s,'active_manifest':str(active_path),'short_key':f"short_{EXPECTED_LOOKBACK.get(s,'')}d" if s in ENTRY_SYMBOLS else '', 'old_path':before.get(f"short_{EXPECTED_LOOKBACK.get(s,'')}d",''),'new_path':after.get(f"short_{EXPECTED_LOOKBACK.get(s,'')}d",''),'link_avoid_only':s=='LINKUSDT','changed':s in ENTRY_SYMBOLS and before!=after})
    report={'schema_version':'aegis_phase_o2_live_promotion_v1','created_at':now(),'run_stamp':run_stamp,'apply':apply,'phase_o_stamp':phase_o_stamp,'backup_index':backup_index,'changed_manifests':changed,'rows':rows,'manual_orders_sent':False,'pm2_restarted':False}
    out_dir.mkdir(parents=True,exist_ok=True)
    write_json(out_dir/f'aegis_phase_o2_live_promotion_{run_stamp}.json', report)
    with (out_dir/f'aegis_phase_o2_live_promotion_symbols_{run_stamp}.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['symbol','active_manifest','short_key','old_path','new_path','link_avoid_only','changed']); w.writeheader(); w.writerows(rows)
    (out_dir/f'aegis_phase_o2_live_promotion_{run_stamp}.md').write_text(f"# Phase O2 Live Promotion {run_stamp}\n\n- apply: {apply}\n- backup_index: {backup_index}\n- changed_manifests: {len(changed)}\n- manual_orders_sent: false\n",encoding='utf-8')
    return report

def build_parser():
    p=argparse.ArgumentParser(); p.add_argument('--symbols',default='ALL'); p.add_argument('--apply',action='store_true'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--out-dir',default='/home/jasan/Develop'); p.add_argument('--backup-dir',default='/home/jasan/Develop/phase_o2_live_backups'); p.add_argument('--require-model-files',action='store_true'); p.add_argument('--phase-o-stamp',default='20260601T070114Z'); p.add_argument('--strict',action='store_true'); return p

def main(argv:Optional[Sequence[str]]=None)->int:
    a=build_parser().parse_args(argv); syms=select_symbols(a.symbols); result=promote(syms,apply=bool(a.apply and not a.dry_run),out_dir=Path(a.out_dir),backup_dir=Path(a.backup_dir),phase_o_stamp=a.phase_o_stamp,require_model_files=a.require_model_files or a.strict); print(json.dumps({'apply':result['apply'],'run_stamp':result['run_stamp'],'changed_manifests':len(result['changed_manifests']),'backup_index':result['backup_index']},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
