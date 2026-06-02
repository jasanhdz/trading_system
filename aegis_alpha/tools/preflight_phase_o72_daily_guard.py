#!/usr/bin/env python3
"""Read-only preflight for corrected Phase O.7.2 daily trade guard semantics."""
from __future__ import annotations
import argparse,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
def nested(data,*keys,default=None):
 cur=data
 for key in keys:
  if not isinstance(cur,dict): return default
  cur=cur.get(key)
  if cur is None: return default
 return cur
def _env_dirty(root):
 lines=[]
 for cwd in [root,root/'binance-futures-bot-ts']:
  try: lines+=subprocess.check_output(['git','status','--short'],cwd=cwd,text=True,stderr=subprocess.DEVNULL).splitlines()
  except Exception: pass
 return [line for line in lines if '.env' in line]
def validate(root=ROOT):
 root=Path(root); errors=[]; warnings=[]
 cfg=yaml.safe_load((root/'binance-futures-bot-ts/regime_config.live.yaml').read_text()) or {}; turbo=yaml.safe_load((root/'aegis_alpha/configs/turbo.yaml').read_text()) or {}
 phase=nested(cfg,'aegis','phase_o_short_live',default={}) or {}; ts=nested(cfg,'aegis','turbo',default={}) or {}; safety=phase.get('hard_safety') or {}
 py=(root/'aegis_alpha/turbo/turbo_risk.py').read_text(); trading=(root/'binance-futures-bot-ts/src/app/services/TradingService.ts').read_text(); loader=(root/'binance-futures-bot-ts/src/infra/config/ConfigLoader.ts').read_text()
 if phase.get('enabled') is not True: errors.append('phase_o_short_live.enabled must be true')
 if phase.get('max_phase_o_trades_per_day') != 20: errors.append('phase_o_short_live.max_phase_o_trades_per_day must equal 20')
 if ts.get('max_trades_per_day') != 20: errors.append('aegis.turbo.max_trades_per_day must equal 20')
 if phase.get('max_open_phase_o_positions') != 9: errors.append('max_open_phase_o_positions must remain 9')
 if phase.get('require_brackets') is not True: errors.append('Phase O brackets must remain required')
 if phase.get('allow_link_entry') is not False or phase.get('link_avoid_only') is not True: errors.append('LINK must remain avoid-only without entry')
 if float(ts.get('daily_loss_stop_pct',-1)) <= 0: errors.append('daily loss stop must remain enabled')
 for key in ['brackets','max_open_positions','max_trades_per_day','daily_loss_stop','exchange_min_notional','exchange_order_errors','link_no_entry']:
  if safety.get(key)!='ENFORCE': errors.append(f'hard_safety.{key} must remain ENFORCE')
 if 'count_today_turbo_signals(history) >= cfg.max_turbo_trades_per_day' in py: errors.append('Python trade guard still blocks on today_signal_count')
 if 'count_today_turbo_opened_trades(trade_events) >= cfg.max_turbo_trades_per_day' not in py: errors.append('Python trade guard must block on POSITION_CONFIRMED trades')
 if '"count_source": "position_confirmed"' not in py or '"today_trade_count"' not in py: errors.append('Python trade guard status must expose position_confirmed trade count')
 for token in ['phaseOShortTradesToday','risk_guard_max_phase_o_trades_per_day',"countSource: 'trade_opened'",'max_phase_o_trades_per_day','metadata?.aegis?.turbo']:
  if token not in trading: errors.append(f'TS Phase O trade guard missing token: {token}')
 if 'getAegisPhaseOShortLiveConfig' not in loader: errors.append('TS config loader missing Phase O short live accessor')
 link_path=root/'aegis_alpha/models/turbo/LINKUSDT/active_manifest.json'
 if link_path.exists():
  link=json.loads(link_path.read_text()); paths=link.get('model_paths') or {}
  if link.get('phase_o_link_entry_enabled') is not False or link.get('phase_o_avoid_only') is not True: errors.append('LINK active manifest must remain avoid-only')
  if any('phase_o_' in str(v) for k,v in paths.items() if str(k).startswith('short_')): errors.append('LINK must not have Phase O entry path')
 else: warnings.append('LINK active manifest missing in test/minimal root')
 dirty=_env_dirty(root)
 if dirty: errors.append(f'.env modified: {dirty}')
 try:
  sys.path.insert(0,str(root)); from aegis_alpha.turbo.turbo_risk import count_today_turbo_opened_trades,load_turbo_trade_events
  current_trade_count=count_today_turbo_opened_trades(load_turbo_trade_events())
 except Exception as exc:
  current_trade_count=None; warnings.append(f'could not calculate current trade count: {exc}')
 limit=int(nested(turbo,'risk','max_turbo_trades_per_day',default=0) or 0)
 would_allow=current_trade_count is not None and current_trade_count < limit
 return {'schema_version':'aegis_phase_o72_daily_guard_preflight_v1','created_at':datetime.now(timezone.utc).isoformat(),'status':'PASSED' if not errors else 'FAILED','block_pm2_restart':bool(errors),'errors':errors,'warnings':warnings,'count_source':'position_confirmed','current_trade_count':current_trade_count,'configured_limit':limit,'would_allow_new_phase_o_entry_by_daily_trade_guard':would_allow,'link_no_entry':not any('LINK' in x for x in errors),'brackets_required':phase.get('require_brackets') is True,'long_model_paths_touched':False}
def reports(payload,out):
 out=Path(out); out.mkdir(parents=True,exist_ok=True); stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); base=out/f'aegis_phase_o72_daily_guard_preflight_{stamp}'; jp=base.with_suffix('.json'); mp=base.with_suffix('.md'); jp.write_text(json.dumps(payload,indent=2)+'\n'); mp.write_text('\n'.join(['# Phase O.7.2 Daily Guard Preflight','',f"Status: **{payload['status']}**",f"BLOCK_PM2_RESTART: **{str(payload['block_pm2_restart']).lower()}**",f"Count source: `{payload['count_source']}`",f"Current trades: `{payload['current_trade_count']}` / `{payload['configured_limit']}`",f"Would allow new Phase O entry: `{payload['would_allow_new_phase_o_entry_by_daily_trade_guard']}`",'','## Errors']+[f'- {x}' for x in payload['errors']]+['','## Warnings']+[f'- {x}' for x in payload['warnings']])+'\n'); return [str(mp),str(jp)]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--from',dest='from_time',default='2026-06-01T19:00:00Z'); ap.add_argument('--out-dir',default='/home/jasan/Develop'); a=ap.parse_args(); payload=validate(); print(json.dumps({'reports':reports(payload,a.out_dir),'status':payload['status'],'block_pm2_restart':payload['block_pm2_restart'],'errors':payload['errors'],'warnings':payload['warnings'],'current_trade_count':payload['current_trade_count'],'configured_limit':payload['configured_limit'],'would_allow_new_phase_o_entry_by_daily_trade_guard':payload['would_allow_new_phase_o_entry_by_daily_trade_guard']},indent=2)); raise SystemExit(0 if payload['status']=='PASSED' else 2)
if __name__=='__main__': main()
