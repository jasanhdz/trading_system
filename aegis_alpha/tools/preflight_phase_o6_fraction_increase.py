#!/usr/bin/env python3
"""Phase O.6 live fraction increase preflight. Read-only validation."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import yaml
TOOLS=Path(__file__).resolve().parent
if str(TOOLS) not in sys.path: sys.path.insert(0,str(TOOLS))
from audit_phase_o5_fraction_alignment import ENTRY_SYMBOLS,nested
ROOT=Path(__file__).resolve().parents[2]
BUCKETS={'conservative':{'fraction':.20,'leverage':15.0},'normal':{'fraction':.35,'leverage':20.0},'premium':{'fraction':.50,'leverage':25.0}}
def load(path): return yaml.safe_load(Path(path).read_text()) or {}
def eq(errors,label,got,want):
 if got != want: errors.append(f'{label}: expected {want}, got {got}')
def env_dirty(root):
 try:
  lines=subprocess.check_output(['git','status','--short'],cwd=root,text=True,stderr=subprocess.DEVNULL).splitlines()
  lines += subprocess.check_output(['git','status','--short'],cwd=root/'binance-futures-bot-ts',text=True,stderr=subprocess.DEVNULL).splitlines()
  return [line for line in lines if '.env' in line]
 except Exception: return []
def sizing_rows(wallet):
 rows=[]
 for bucket,cfg in BUCKETS.items():
  margin=wallet*cfg['fraction']; notional=margin*cfg['leverage']
  rows.append({'bucket':bucket,'fraction':cfg['fraction'],'wallet_usdt':wallet,'margin_usdt':margin,'leverage':cfg['leverage'],'notional_usdt':notional,'margin_wallet_pct':cfg['fraction']*100,'risk_warning':'HIGH_RISK_TEST_CAPITAL' if cfg['fraction']>=.50 else ''})
 return rows
def validate(root=ROOT,wallet=20.0):
 root=Path(root); cfg=load(root/'binance-futures-bot-ts/regime_config.live.yaml'); turbo=load(root/'aegis_alpha/configs/turbo.yaml'); errors=[]; warnings=[]
 phase=nested(cfg,'aegis','phase_o_short_live',default={}) or {}; ts_turbo=nested(cfg,'aegis','turbo',default={}) or {}; safety=phase.get('hard_safety') or {}
 for bucket,want in BUCKETS.items():
  eq(errors,f'turbo.{bucket}.position_fraction',float(nested(turbo,'sizing',bucket,'position_fraction',default=-1)),want['fraction'])
  eq(errors,f'turbo.{bucket}.leverage',float(nested(turbo,'sizing',bucket,'leverage',default=-1)),want['leverage'])
 eq(errors,'turbo.max_allowed_position_fraction',float(nested(turbo,'sizing','max_allowed_position_fraction',default=-1)),.50)
 if float(nested(turbo,'sizing','max_allowed_leverage',default=999))>30: errors.append('turbo max leverage exceeds 30')
 for key in ['enabled','allow_orders','require_brackets']:
  if phase.get(key) is not True: errors.append(f'phase_o_short_live.{key} must be true')
 if phase.get('max_open_phase_o_positions') != 1: errors.append('max_open_phase_o_positions must equal 1')
 if phase.get('max_phase_o_trades_per_day') != 3 or ts_turbo.get('max_trades_per_day') != 3: errors.append('max trades/day must equal 3')
 if ts_turbo.get('max_consecutive_losses') != 2: errors.append('max_consecutive_losses must equal 2')
 if not (float(ts_turbo.get('daily_loss_stop_pct',0) or 0)>0): errors.append('daily_loss_stop_pct must remain active')
 if phase.get('allow_link_entry') is not False or phase.get('link_avoid_only') is not True: errors.append('LINK entry must stay disabled and avoid-only enabled')
 for bucket,want in [('conservative',15),('normal',20),('premium',25)]: eq(errors,f'phase leverage {bucket}',nested(phase,'leverage',bucket),want)
 if float(nested(phase,'leverage','max_allowed_leverage',default=999))>30: errors.append('phase max leverage exceeds 30')
 cap=float(ts_turbo.get('position_fraction_cap',-1))
 if cap != .50: errors.append(f'aegis.turbo.position_fraction_cap must equal 0.50, got {cap}')
 if cap > .50: errors.append('position fraction cap exceeds authorized 0.50')
 for key in ['brackets','max_open_positions','max_trades_per_day','daily_loss_stop','exchange_min_notional','exchange_order_errors','link_no_entry']:
  if safety.get(key)!='ENFORCE': errors.append(f'hard_safety.{key} must be ENFORCE')
 if 'max_position_fraction_default' in phase: errors.append('remove duplicated phase_o_short_live.max_position_fraction_default')
 for symbol in ENTRY_SYMBOLS:
  sc=(phase.get('symbols') or {}).get(symbol) or {}
  if sc.get('enabled') is not True: errors.append(f'{symbol} phase symbol must be enabled')
  if 'max_position_fraction' in sc: errors.append(f'{symbol}: duplicated Phase O fraction must be removed')
 link=(phase.get('symbols') or {}).get('LINKUSDT') or {}
 if link.get('entry_enabled') is not False or link.get('avoid_only') is not True: errors.append('LINKUSDT symbol config must remain avoid-only without entry')
 model_root=root/'aegis_alpha/models/turbo'
 for symbol in ENTRY_SYMBOLS:
  p=model_root/symbol/'active_manifest.json'
  if not p.exists(): errors.append(f'{symbol}: missing active manifest'); continue
  m=json.loads(p.read_text()); paths=m.get('model_paths') or {}; before=m.get('pre_phase_o_live_model_paths') or {}
  if not m.get('phase_o_live_enabled'): errors.append(f'{symbol}: phase_o_live_enabled missing')
  phase_paths=[str(v) for k,v in paths.items() if k.startswith('short_') and 'phase_o_' in str(v)]
  if not phase_paths: errors.append(f'{symbol}: missing Phase O short model path')
  for mp in phase_paths:
   if not Path(mp).exists(): errors.append(f'{symbol}: missing model file: {mp}')
  for k,v in paths.items():
   if k.startswith('long_') and before.get(k)!=v: errors.append(f'{symbol}: LONG model path changed: {k}')
 lp=model_root/'LINKUSDT'/'active_manifest.json'
 if lp.exists():
  m=json.loads(lp.read_text()); paths=m.get('model_paths') or {}
  if m.get('phase_o_link_entry_enabled') is not False or not m.get('phase_o_avoid_only'): errors.append('LINK manifest must remain avoid-only')
  if any('phase_o_' in str(v) for k,v in paths.items() if k.startswith('short_')): errors.append('LINK must not receive Phase O entry model')
 dirty=env_dirty(root)
 if dirty: errors.append(f'.env modified: {dirty}')
 rows=sizing_rows(wallet)
 if any(r['fraction']>=.50 for r in rows): warnings.append('HIGH_RISK_TEST_CAPITAL: premium margin uses 50% of wallet; explicitly authorized for O.6')
 return {'status':'PASSED' if not errors else 'FAILED','block_pm2_restart':bool(errors),'errors':errors,'warnings':warnings,'wallet_usdt':wallet,'sizing':rows,'hard_safety':safety,'long_model_paths_intact':not any('LONG model path changed' in e for e in errors),'link_no_entry':not any('LINK' in e for e in errors)}
def reports(payload,out):
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); out=Path(out); out.mkdir(parents=True,exist_ok=True); base=out/f'aegis_phase_o6_fraction_increase_preflight_{stamp}'; base.with_suffix('.json').write_text(json.dumps(payload,indent=2)+'\n'); csvp=out/f'aegis_phase_o6_effective_sizing_20usdt_{stamp}.csv'
 with csvp.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(payload['sizing'][0])); w.writeheader(); w.writerows(payload['sizing'])
 md=['# Phase O.6 Fraction Increase Preflight','',f"Status: **{payload['status']}**",f"BLOCK_PM2_RESTART: **{str(payload['block_pm2_restart']).lower()}**",'','## Warnings']+[f'- {x}' for x in payload['warnings']]+['','## Errors']+[f'- {x}' for x in payload['errors']]+['','## Safety',f"LONG model paths intact: {payload['long_model_paths_intact']}",f"LINK no entry: {payload['link_no_entry']}"]
 base.with_suffix('.md').write_text('\n'.join(md)+'\n'); return [str(base.with_suffix('.md')),str(base.with_suffix('.json')),str(csvp)]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--wallet-usdt',type=float,default=20); ap.add_argument('--out-dir',default='/home/jasan/Develop'); a=ap.parse_args(); payload=validate(wallet=a.wallet_usdt); print(json.dumps({'reports':reports(payload,a.out_dir),'status':payload['status'],'block_pm2_restart':payload['block_pm2_restart'],'warnings':payload['warnings'],'errors':payload['errors']},indent=2)); raise SystemExit(0 if payload['status']=='PASSED' else 2)
if __name__=='__main__': main()
