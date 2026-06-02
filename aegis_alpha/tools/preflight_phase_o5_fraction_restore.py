#!/usr/bin/env python3
"""Phase O.5 live fraction restore preflight. Read-only validation."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import yaml
TOOLS=Path(__file__).resolve().parent
if str(TOOLS) not in sys.path: sys.path.insert(0,str(TOOLS))
from audit_phase_o5_fraction_alignment import BUCKETS,ENTRY_SYMBOLS,nested
ROOT=Path(__file__).resolve().parents[2]; TS=ROOT/'binance-futures-bot-ts'
def load(path): return yaml.safe_load(Path(path).read_text()) or {}
def env_dirty(root=ROOT):
 try:
  lines=subprocess.check_output(['git','status','--short'],cwd=root,text=True,stderr=subprocess.DEVNULL).splitlines()
  lines += subprocess.check_output(['git','status','--short'],cwd=root/'binance-futures-bot-ts',text=True,stderr=subprocess.DEVNULL).splitlines()
  return [x for x in lines if '.env' in x]
 except Exception: return []
def check_eq(errors,label,got,want):
 if got != want: errors.append(f'{label}: expected {want}, got {got}')
def sizing_rows(wallet):
 return [{'symbol':s,'fraction_source':'raw_ml_turbo_bucket','wallet_usdt':wallet,'conservative_fraction':.08,'normal_fraction':.12,'premium_fraction':.18,'margin_conservative':wallet*.08,'margin_normal':wallet*.12,'margin_premium':wallet*.18,'notional_15x_conservative':wallet*.08*15,'notional_20x_normal':wallet*.12*20,'notional_25x_premium':wallet*.18*25,'min_notional_warning_conservative':wallet*.08*15<5} for s in ENTRY_SYMBOLS]

def validate(root=ROOT,wallet=20.0):
 root=Path(root); ts=root/'binance-futures-bot-ts'; turbo=load(root/'aegis_alpha/configs/turbo.yaml'); cfg=load(ts/'regime_config.live.yaml'); errors=[]
 phase=nested(cfg,'aegis','phase_o_short_live',default={}) or {}; at=nested(cfg,'aegis','turbo',default={}) or {}; safety=phase.get('hard_safety') or {}
 for key in ['enabled','allow_orders','require_brackets']:
  if phase.get(key) is not True: errors.append(f'phase_o_short_live.{key} must be true')
 if phase.get('allow_link_entry') is not False or phase.get('link_avoid_only') is not True: errors.append('LINK entry must stay disabled and avoid-only enabled')
 if phase.get('max_open_phase_o_positions') != 1: errors.append('max_open_phase_o_positions must equal 1')
 if phase.get('max_phase_o_trades_per_day') != 3 or at.get('max_trades_per_day') != 3: errors.append('max trades/day must equal 3')
 if at.get('max_consecutive_losses') != 2: errors.append('max_consecutive_losses must equal 2')
 if not (float(at.get('daily_loss_stop_pct',0) or 0)>0): errors.append('daily_loss_stop_pct must be active')
 for key in ['brackets','max_open_positions','max_trades_per_day','daily_loss_stop','exchange_min_notional','exchange_order_errors','link_no_entry']:
  if safety.get(key) != 'ENFORCE': errors.append(f'hard_safety.{key} must be ENFORCE')
 for bucket,want_frac,want_lev in [('conservative',.08,15.0),('normal',.12,20.0),('premium',.18,25.0)]:
  check_eq(errors,f'turbo sizing {bucket} fraction',float(nested(turbo,'sizing',bucket,'position_fraction',default=-1)),want_frac)
  check_eq(errors,f'turbo sizing {bucket} leverage',float(nested(turbo,'sizing',bucket,'leverage',default=-1)),want_lev)
 check_eq(errors,'turbo max fraction',float(nested(turbo,'sizing','max_allowed_position_fraction',default=-1)),.20)
 if float(nested(turbo,'sizing','max_allowed_leverage',default=999))>30: errors.append('turbo max leverage exceeds 30')
 for bucket,want in [('conservative',15),('normal',20),('premium',25)]: check_eq(errors,f'phase leverage {bucket}',nested(phase,'leverage',bucket),want)
 if float(nested(phase,'leverage','max_allowed_leverage',default=999))>30: errors.append('phase max leverage exceeds 30')
 if 'max_position_fraction_default' in phase: errors.append('phase_o_short_live must inherit ML fraction; remove duplicated max_position_fraction_default')
 cap=float(at.get('position_fraction_cap',0) or 0)
 if cap < .20: errors.append(f'aegis.turbo.position_fraction_cap must be >= 0.20, got {cap}')
 for symbol in ENTRY_SYMBOLS:
  sc=(phase.get('symbols') or {}).get(symbol) or {}
  if sc.get('enabled') is not True: errors.append(f'{symbol} phase symbol must be enabled')
  if 'max_position_fraction' in sc: errors.append(f'{symbol}: remove duplicated Phase O max_position_fraction; inherit ML raw fraction')
 link=(phase.get('symbols') or {}).get('LINKUSDT') or {}
 if link.get('entry_enabled') is not False or link.get('avoid_only') is not True: errors.append('LINKUSDT symbol config must remain avoid-only without entry')
 model_root=root/'aegis_alpha/models/turbo'
 for symbol in ENTRY_SYMBOLS:
  p=model_root/symbol/'active_manifest.json'
  if not p.exists(): errors.append(f'{symbol}: missing active manifest'); continue
  m=json.loads(p.read_text()); paths=m.get('model_paths') or {}; before=m.get('pre_phase_o_live_model_paths') or {}
  if not m.get('phase_o_live_enabled'): errors.append(f'{symbol}: phase_o_live_enabled missing')
  phase_paths=[str(v) for k,v in paths.items() if k.startswith('short_') and 'phase_o_' in str(v)]
  if not phase_paths: errors.append(f'{symbol}: missing Phase O short path')
  for model_path in phase_paths:
   if not Path(model_path).exists(): errors.append(f'{symbol}: missing Phase O model file: {model_path}')
  for k,v in paths.items():
   if k.startswith('long_') and before.get(k)!=v: errors.append(f'{symbol}: LONG model path changed: {k}')
 lp=model_root/'LINKUSDT'/'active_manifest.json'
 if lp.exists():
  linkm=json.loads(lp.read_text()); paths=linkm.get('model_paths') or {}
  if linkm.get('phase_o_link_entry_enabled') is not False or not linkm.get('phase_o_avoid_only'): errors.append('LINK manifest must remain avoid-only')
  if any('phase_o_' in str(v) for k,v in paths.items() if k.startswith('short_')): errors.append('LINK must not receive Phase O entry short path')
 dirty=env_dirty(root)
 if dirty: errors.append(f'.env modified: {dirty}')
 rows=sizing_rows(wallet)
 return {'status':'PASSED' if not errors else 'FAILED','block_pm2_restart':bool(errors),'errors':errors,'wallet_usdt':wallet,'sizing':rows,'hard_safety':safety,'long_model_paths_intact':not any('LONG model path changed' in e for e in errors),'link_no_entry':not any('LINK' in e for e in errors)}
def reports(payload,out):
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); out=Path(out); out.mkdir(parents=True,exist_ok=True); base=out/f'aegis_phase_o5_fraction_restore_preflight_{stamp}'
 base.with_suffix('.json').write_text(json.dumps(payload,indent=2)+"\n"); csvp=out/f'aegis_phase_o5_effective_sizing_20usdt_{stamp}.csv'
 with csvp.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(payload['sizing'][0])); w.writeheader(); w.writerows(payload['sizing'])
 md=['# Phase O.5 Fraction Restore Preflight','',f"Status: **{payload['status']}**",f"BLOCK_PM2_RESTART: **{str(payload['block_pm2_restart']).lower()}**",'', '## Errors']+[f'- {e}' for e in payload['errors']] + ['', '## Safety',f"LONG model paths intact: {payload['long_model_paths_intact']}",f"LINK no entry: {payload['link_no_entry']}"]
 base.with_suffix('.md').write_text('\n'.join(md)+'\n'); return [str(base.with_suffix('.md')),str(base.with_suffix('.json')),str(csvp)]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--wallet-usdt',type=float,default=20); ap.add_argument('--out-dir',default='/home/jasan/Develop'); a=ap.parse_args(); p=validate(wallet=a.wallet_usdt); print(json.dumps({'reports':reports(p,a.out_dir),'status':p['status'],'block_pm2_restart':p['block_pm2_restart'],'errors':p['errors']},indent=2)); raise SystemExit(0 if p['status']=='PASSED' else 2)
if __name__=='__main__': main()
