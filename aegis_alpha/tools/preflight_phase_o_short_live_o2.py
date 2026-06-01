#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,sys
from datetime import datetime,timezone
from pathlib import Path
from typing import Any,Optional,Sequence
if __package__ is None or __package__=='': sys.path.append(str(Path(__file__).resolve().parents[2]))
from aegis_alpha.turbo.snapshot_utils import turbo_symbol_model_dir # noqa:E402
ENTRY=["LTCUSDT","AVAXUSDT","ETHUSDT","SUIUSDT","ADAUSDT","DOGEUSDT","BTCUSDT","BNBUSDT","XRPUSDT","SOLUSDT"]
ALL=ENTRY+["LINKUSDT"]
LOOK={"LTCUSDT":14,"AVAXUSDT":14,"ETHUSDT":30,"SUIUSDT":7,"ADAUSDT":30,"DOGEUSDT":30,"BTCUSDT":30,"BNBUSDT":30,"XRPUSDT":30,"SOLUSDT":30}
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def read_json(p:Path)->dict[str,Any]: return json.loads(p.read_text()) if p.exists() else {}
def write_json(p:Path,d:dict[str,Any]): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def load_yaml(path:Path)->dict[str,Any]:
    try:
        import yaml
        return yaml.safe_load(path.read_text()) or {}
    except Exception as e: return {'__error__':repr(e)}
def get_nested(d:dict[str,Any],keys:list[str])->Any:
    cur=d
    for k in keys:
        if not isinstance(cur,dict): return None
        cur=cur.get(k)
    return cur
def preflight(out_dir:Path, yaml_path:Path|None=None, runtime_check:bool=False)->dict[str,Any]:
    rs=stamp(); rows=[]; errors=[]
    for s in ALL:
        p=turbo_symbol_model_dir(s)/'active_manifest.json'; d=read_json(p); mp=d.get('model_paths') if isinstance(d.get('model_paths'),dict) else {}; pre=d.get('pre_phase_o_live_model_paths')
        if s in ENTRY:
            key=f"short_{LOOK[s]}d"; path=str(mp.get(key,'')); ok=('/phase_o_' in path and Path(path).exists())
            if d.get('phase_o_live_enabled') is not True: errors.append(f'{s}:phase_o_live_enabled_not_true')
            if not ok: errors.append(f'{s}:{key}_not_phase_o_or_missing')
            if not isinstance(pre,dict): errors.append(f'{s}:missing_pre_phase_o_live_model_paths')
            for k,v in mp.items():
                if k.startswith('long_') and isinstance(pre,dict) and pre.get(k)!=v: errors.append(f'{s}:long_path_changed:{k}')
            rows.append({'symbol':s,'entry':True,'short_key':key,'short_path':path,'phase_o_path_ok':ok,'link_entry_disabled':''})
        else:
            if d.get('phase_o_link_entry_enabled') is not False: errors.append('LINKUSDT:entry_not_disabled')
            if d.get('phase_o_avoid_only') is not True: errors.append('LINKUSDT:avoid_only_not_true')
            if any('/phase_o_' in str(v) for k,v in mp.items() if k.startswith('short_')): errors.append('LINKUSDT:short_model_path_points_to_phase_o')
            artifacts=d.get('phase_o_avoid_artifacts') or []
            if not artifacts or any(not Path(str(x)).exists() for x in artifacts): errors.append('LINKUSDT:avoid_artifacts_missing')
            rows.append({'symbol':s,'entry':False,'short_key':'','short_path':'','phase_o_path_ok':False,'link_entry_disabled':d.get('phase_o_link_entry_enabled') is False})
    runtime_rows=[]
    if runtime_check:
        try:
            from aegis_alpha.turbo.turbo_signal import evaluate_turbo_shadow, phase_o_runtime_metadata
            for s in ALL:
                meta=phase_o_runtime_metadata(s)
                if s in ENTRY and meta.get('phase_o_live_enabled') is not True:
                    errors.append(f'{s}:runtime_phase_o_metadata_missing')
                if s=='LINKUSDT' and not meta.get('phase_o_link_avoid_only'):
                    errors.append('LINKUSDT:runtime_avoid_only_metadata_missing')
                try:
                    signal=evaluate_turbo_shadow(s)
                    runtime_rows.append({'symbol':s,'runtime_ok':True,'action':signal.get('action'),'reason':signal.get('reason'),'phase_o_metadata':bool(signal.get('phase_o') or meta)})
                    if s=='LINKUSDT' and signal.get('action')!='HOLD' and meta.get('phase_o_link_avoid_only'):
                        errors.append('LINKUSDT:runtime_returned_entry_action')
                except Exception as exc:
                    runtime_rows.append({'symbol':s,'runtime_ok':False,'action':'','reason':repr(exc),'phase_o_metadata':bool(meta)})
                    errors.append(f'{s}:runtime_evaluate_crashed')
        except Exception as exc:
            runtime_rows=[]; errors.append(f'runtime_import_failed:{exc!r}')
    yaml_path=yaml_path or Path('binance-futures-bot-ts/regime_config.live.yaml')
    y=load_yaml(yaml_path); turbo=get_nested(y,['aegis','turbo']) or {}
    if turbo.get('enabled') is not True: errors.append('yaml:aegis.turbo.enabled_not_true')
    if turbo.get('live_enabled') is not True: errors.append('yaml:aegis.turbo.live_enabled_not_true')
    if turbo.get('allow_short') is not True: errors.append('yaml:aegis.turbo.allow_short_not_true')
    if float(turbo.get('position_fraction_cap',999))>0.01: errors.append('yaml:position_fraction_cap_gt_0.01')
    for idx, rule in enumerate(turbo.get('position_fraction_overrides') or []):
        if not isinstance(rule, dict):
            errors.append(f'yaml:position_fraction_override_{idx}_invalid')
            continue
        for side_key in ('long', 'short'):
            if side_key in rule and float(rule.get(side_key, 999)) > 0.01:
                errors.append(f"yaml:position_fraction_override_{rule.get('name', idx)}_{side_key}_gt_0.01")
    if int(turbo.get('max_trades_per_day',999))>3: errors.append('yaml:max_trades_per_day_gt_3')
    if turbo.get('require_brackets') is not True: errors.append('yaml:require_brackets_not_true')
    phase=get_nested(y,['aegis','phase_o_short_live']) or {}
    if phase.get('allow_link_entry') is not False: errors.append('yaml:allow_link_entry_not_false')
    if phase.get('link_avoid_only') is not True: errors.append('yaml:link_avoid_only_not_true')
    status='PASSED' if not errors else 'FAILED'
    report={'schema_version':'aegis_phase_o2_live_preflight_v1','created_at':now(),'run_stamp':rs,'status':status,'block_pm2_restart':bool(errors),'errors':errors,'rows':rows,'runtime_rows':runtime_rows,'yaml_path':str(yaml_path),'manual_orders_sent':False}
    out_dir.mkdir(parents=True,exist_ok=True); write_json(out_dir/f'aegis_phase_o2_live_preflight_{rs}.json',report); (out_dir/f'aegis_phase_o2_live_preflight_{rs}.md').write_text(f"# Phase O2 Preflight {rs}\n\n- status: {status}\n- block_pm2_restart: {bool(errors)}\n\n"+'\n'.join(f'- {e}' for e in errors)+'\n',encoding='utf-8')
    with (out_dir/f'aegis_phase_o2_live_preflight_symbols_{rs}.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['symbol','entry','short_key','short_path','phase_o_path_ok','link_entry_disabled']); w.writeheader(); w.writerows(rows)
    return report
def main(argv:Optional[Sequence[str]]=None)->int:
    p=argparse.ArgumentParser(); p.add_argument('--out-dir',default='/home/jasan/Develop'); p.add_argument('--yaml-path'); a=p.parse_args(argv); r=preflight(Path(a.out_dir),Path(a.yaml_path) if a.yaml_path else None, runtime_check=True); print(json.dumps({'status':r['status'],'block_pm2_restart':r['block_pm2_restart'],'errors':r['errors'][:20]},indent=2)); return 0 if r['status']=='PASSED' else 2
if __name__=='__main__': raise SystemExit(main())
