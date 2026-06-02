#!/usr/bin/env python3
"""Phase O.7 aggressive-mode live preflight. Read-only validation."""
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
  lines=subprocess.check_output(['git','status','--short'],cwd=root,text=True,stderr=subprocess.DEVNULL).splitlines(); lines += subprocess.check_output(['git','status','--short'],cwd=root/'binance-futures-bot-ts',text=True,stderr=subprocess.DEVNULL).splitlines(); return [x for x in lines if '.env' in x]
 except Exception: return []
def exposure_rows(wallet,max_open=9):
 rows=[]
 for bucket,cfg in BUCKETS.items():
  margin=wallet*cfg['fraction']; notional=margin*cfg['leverage']
  rows.append({'bucket':bucket,'fraction':cfg['fraction'],'wallet_usdt':wallet,'margin_per_trade_usdt':margin,'leverage':cfg['leverage'],'notional_per_trade_usdt':notional,'max_open_positions':max_open,'worst_case_margin_usdt':margin*max_open,'worst_case_notional_usdt':notional*max_open,'margin_overcommit':margin*max_open>wallet})
 return rows
def validate(root=ROOT,wallet=20.0):
 root=Path(root); cfg=load(root/'binance-futures-bot-ts/regime_config.live.yaml'); turbo=load(root/'aegis_alpha/configs/turbo.yaml'); errors=[]; warnings=['HIGH_RISK_AGGRESSIVE_MODE','MARGIN_OVERCOMMIT_POSSIBLE','PREMIUM_USES_50_PERCENT_WALLET','MAX_OPEN_9_EXCEEDS_WALLET_IF_MULTIPLE_PREMIUM']
 phase=nested(cfg,'aegis','phase_o_short_live',default={}) or {}; ts=nested(cfg,'aegis','turbo',default={}) or {}; safety=phase.get('hard_safety') or {}
 for bucket,want in BUCKETS.items():
  eq(errors,f'turbo {bucket} fraction',float(nested(turbo,'sizing',bucket,'position_fraction',default=-1)),want['fraction']); eq(errors,f'turbo {bucket} leverage',float(nested(turbo,'sizing',bucket,'leverage',default=-1)),want['leverage'])
 eq(errors,'turbo max fraction',float(nested(turbo,'sizing','max_allowed_position_fraction',default=-1)),.50)
 if float(nested(turbo,'sizing','max_allowed_leverage',default=999))>30: errors.append('turbo max leverage exceeds 30')
 eq(errors,'root risk max trades/day',nested(turbo,'risk','max_turbo_trades_per_day'),20); eq(errors,'root risk consecutive losses',nested(turbo,'risk','max_consecutive_losses'),3); eq(errors,'root risk daily loss pct',float(nested(turbo,'risk','daily_loss_stop_pct',default=-1)),30.0)
 for key in ['enabled','allow_orders','require_brackets']:
  if phase.get(key) is not True: errors.append(f'phase_o_short_live.{key} must be true')
 if phase.get('max_open_phase_o_positions') != 9: errors.append(f'max_open_phase_o_positions must equal 9, got {phase.get("max_open_phase_o_positions")}')
 if phase.get('max_phase_o_trades_per_day') != 20 or ts.get('max_trades_per_day') != 20: errors.append('max trades/day must equal 20')
 if ts.get('max_consecutive_losses') != 3: errors.append('max_consecutive_losses must equal 3')
 if float(ts.get('daily_loss_stop_pct',-1)) != .30: errors.append(f'TS daily_loss_stop_pct must equal 0.30, got {ts.get("daily_loss_stop_pct")}')
 if phase.get('max_open_phase_o_positions',999)>9: errors.append('max open exceeds authorized 9')
 if phase.get('max_phase_o_trades_per_day',999)>20 or ts.get('max_trades_per_day',999)>20: errors.append('max trades/day exceeds authorized 20')
 if ts.get('max_consecutive_losses',999)>3: errors.append('max consecutive losses exceeds authorized 3')
 if float(ts.get('daily_loss_stop_pct',999))>.30: errors.append('daily loss stop exceeds authorized 0.30')
 if float(ts.get('position_fraction_cap',999)) != .50: errors.append(f'position_fraction_cap must equal 0.50, got {ts.get("position_fraction_cap")}')
 if float(ts.get('position_fraction_cap',999))>.50: errors.append('position fraction cap exceeds authorized 0.50')
 for bucket,want in [('conservative',15),('normal',20),('premium',25)]: eq(errors,f'phase leverage {bucket}',nested(phase,'leverage',bucket),want)
 if float(nested(phase,'leverage','max_allowed_leverage',default=999))>30: errors.append('phase max leverage exceeds 30')
 if phase.get('allow_link_entry') is not False or phase.get('link_avoid_only') is not True: errors.append('LINK entry must remain disabled and avoid-only enabled')
 for key in ['brackets','max_open_positions','max_trades_per_day','daily_loss_stop','exchange_min_notional','exchange_order_errors','link_no_entry']:
  if safety.get(key)!='ENFORCE': errors.append(f'hard_safety.{key} must be ENFORCE')
 if 'max_position_fraction_default' in phase: errors.append('remove duplicated phase_o_short_live fraction')
 for symbol in ENTRY_SYMBOLS:
  sc=(phase.get('symbols') or {}).get(symbol) or {}
  if sc.get('enabled') is not True: errors.append(f'{symbol}: phase entry symbol disabled')
  if 'max_position_fraction' in sc: errors.append(f'{symbol}: duplicated Phase O fraction must be removed')
 link=(phase.get('symbols') or {}).get('LINKUSDT') or {}
 if link.get('entry_enabled') is not False or link.get('avoid_only') is not True: errors.append('LINKUSDT symbol config must remain avoid-only without entry')
 rollback=root/'aegis_alpha/tools/rollback_phase_o_short_live_o2.py'
 if not rollback.exists(): errors.append('rollback script missing')
 retrain_script=root/'aegis_alpha/tools/run_turbo_scheduled_retrain.py'
 if not retrain_script.exists(): errors.append('scheduled retrain script missing')
 else:
  retrain_source=retrain_script.read_text()
  if 'apply_phase_o_overlay_to_active_manifest' not in retrain_source: errors.append('scheduled retrain Phase O overlay integration missing')
  if '--disable-phase-o-overlay' not in retrain_source: errors.append('scheduled retrain overlay CLI guard missing')
 model_root=root/'aegis_alpha/models/turbo'
 for symbol in ENTRY_SYMBOLS:
  p=model_root/symbol/'active_manifest.json'
  if not p.exists(): errors.append(f'{symbol}: missing active manifest'); continue
  m=json.loads(p.read_text()); paths=m.get('model_paths') or {}; before=m.get('pre_phase_o_live_model_paths') or {}
  if not m.get('phase_o_live_enabled'): errors.append(f'{symbol}: phase_o_live_enabled missing')
  if not m.get('phase_o_overlay_persistence_enabled'): errors.append(f'{symbol}: Phase O overlay persistence missing')
  if not m.get('phase_o_live_artifact_stamp'): errors.append(f'{symbol}: Phase O artifact stamp missing')
  if not any('phase_o_' in str(v) for k,v in paths.items() if k.startswith('short_')): errors.append(f'{symbol}: missing Phase O short model path')
  for k,v in paths.items():
   if k.startswith('long_') and before.get(k)!=v: errors.append(f'{symbol}: LONG model path changed: {k}')
 lp=model_root/'LINKUSDT'/'active_manifest.json'
 if lp.exists():
  m=json.loads(lp.read_text()); paths=m.get('model_paths') or {}
  if m.get('phase_o_link_entry_enabled') is not False or not m.get('phase_o_avoid_only'): errors.append('LINK manifest must remain avoid-only')
  if not m.get('phase_o_overlay_persistence_enabled'): errors.append('LINK overlay persistence missing')
  if not m.get('phase_o_live_artifact_stamp'): errors.append('LINK Phase O artifact stamp missing')
  if any('phase_o_' in str(v) for k,v in paths.items() if k.startswith('short_')): errors.append('LINK must not receive Phase O entry model')
 dirty=env_dirty(root)
 if dirty: errors.append(f'.env modified: {dirty}')
 return {'status':'PASSED' if not errors else 'FAILED','block_pm2_restart':bool(errors),'errors':errors,'warnings':warnings,'wallet_usdt':wallet,'exposure':exposure_rows(wallet,9),'hard_safety':safety,'rollback':str(rollback),'long_model_paths_intact':not any('LONG model path changed' in e for e in errors),'link_no_entry':not any('LINK' in e for e in errors)}
def reports(payload,out):
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); out=Path(out); out.mkdir(parents=True,exist_ok=True); base=out/f'aegis_phase_o7_aggressive_preflight_{stamp}'; base.with_suffix('.json').write_text(json.dumps(payload,indent=2)+'\n'); csvp=out/f'aegis_phase_o7_aggressive_sizing_20usdt_{stamp}.csv'
 with csvp.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(payload['exposure'][0])); w.writeheader(); w.writerows(payload['exposure'])
 md=['# Phase O.7 Aggressive Preflight','',f"Status: **{payload['status']}**",f"BLOCK_PM2_RESTART: **{str(payload['block_pm2_restart']).lower()}**",'','## Warnings']+[f'- {x}' for x in payload['warnings']]+['','## Errors']+[f'- {x}' for x in payload['errors']]+['','## Safety',f"Rollback: `{payload['rollback']}`",f"LONG model paths intact: {payload['long_model_paths_intact']}",f"LINK no entry: {payload['link_no_entry']}"]
 base.with_suffix('.md').write_text('\n'.join(md)+'\n'); return [str(base.with_suffix('.md')),str(base.with_suffix('.json')),str(csvp)]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--wallet-usdt',type=float,default=20); ap.add_argument('--out-dir',default='/home/jasan/Develop'); a=ap.parse_args(); payload=validate(wallet=a.wallet_usdt); print(json.dumps({'reports':reports(payload,a.out_dir),'status':payload['status'],'block_pm2_restart':payload['block_pm2_restart'],'warnings':payload['warnings'],'errors':payload['errors']},indent=2)); raise SystemExit(0 if payload['status']=='PASSED' else 2)
if __name__=='__main__': main()
