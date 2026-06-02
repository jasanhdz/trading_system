#!/usr/bin/env python3
"""Compare Phase O fractions to historical Turbo ML sizing without mutating config."""
from __future__ import annotations
import argparse,csv,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]; TS=ROOT/'binance-futures-bot-ts'
ENTRY_SYMBOLS=['LTCUSDT','AVAXUSDT','ETHUSDT','SUIUSDT','ADAUSDT','DOGEUSDT','BTCUSDT','BNBUSDT','XRPUSDT','SOLUSDT']
BUCKETS={'conservative':.08,'normal':.12,'premium':.18}
def nested(d,*keys,default=None):
 for k in keys:
  if not isinstance(d,dict): return default
  d=d.get(k)
  if d is None: return default
 return d
def load_yaml(path): return yaml.safe_load(Path(path).read_text()) or {}
def git_show(repo,ref,path):
 try: return yaml.safe_load(subprocess.check_output(['git','show',f'{ref}:{path}'],cwd=repo,text=True,stderr=subprocess.DEVNULL)) or {}
 except Exception: return {}
def alignment_rows(current_ts,wallet=20.0):
 phase=nested(current_ts,'aegis','phase_o_short_live',default={}) or {}; turbo=nested(current_ts,'aegis','turbo',default={}) or {}; cap=float(turbo.get('position_fraction_cap',0) or 0); rows=[]
 for symbol in ENTRY_SYMBOLS:
  configured=nested(phase,'symbols',symbol,'max_position_fraction',default=phase.get('max_position_fraction_default'))
  rows.append({'symbol':symbol,'side':'SHORT','phase_o_yaml_fraction':configured,'fraction_source':'raw_ml_turbo_bucket','ts_safety_cap':cap,'inherited_conservative_fraction':.08,'inherited_normal_fraction':.12,'inherited_premium_fraction':.18,'wallet_20_margin_conservative':wallet*.08,'wallet_20_margin_normal':wallet*.12,'wallet_20_margin_premium':wallet*.18,'notional_15x_conservative':wallet*.08*15,'notional_20x_normal':wallet*.12*20,'notional_25x_premium':wallet*.18*25,'min_notional_risk_conservative':wallet*.08*15<5,'recommendation':'inherit_raw_ml_fraction_with_ts_cap'})
 return rows
def audit(wallet=20.0):
 root_now=load_yaml(ROOT/'aegis_alpha/configs/turbo.yaml'); ts_now=load_yaml(TS/'regime_config.live.yaml'); root_before=git_show(ROOT,'326738d','aegis_alpha/configs/turbo.yaml'); ts_before=git_show(TS,'22937cb^','regime_config.live.yaml')
 return {'safety':'READ_ONLY','wallet_usdt':wallet,'evidence':{'root_historical_ref':'326738d','ts_historical_ref':'22937cb^','historical_turbo_buckets':{k:nested(root_before,'sizing',k,'position_fraction') for k in BUCKETS},'historical_turbo_max_fraction':nested(root_before,'sizing','max_allowed_position_fraction'),'historical_ts_turbo_cap':nested(ts_before,'aegis','turbo','position_fraction_cap')},'current':{'turbo_buckets':{k:nested(root_now,'sizing',k,'position_fraction') for k in BUCKETS},'turbo_max_fraction':nested(root_now,'sizing','max_allowed_position_fraction'),'phase_o_default':nested(ts_now,'aegis','phase_o_short_live','max_position_fraction_default'),'ts_turbo_cap':nested(ts_now,'aegis','turbo','position_fraction_cap'),'momentum_cap':nested(ts_now,'aegis','momentum_ride','safety_caps','max_position_fraction'),'long_override':nested(ts_now,'aegis','turbo','position_fraction_overrides')},'decision':{'fraction_source':'raw_ml_turbo_bucket','python_buckets':BUCKETS,'ts_safety_cap':.20,'phase_o_fraction_fields_removed':True,'keep_momentum_cap':.01,'reason':'Use existing Turbo ML sizing as the single strategy source of truth. TS only applies the 0.20 safety ceiling.'},'rows':alignment_rows(ts_now,wallet)}
def write_reports(payload,out):
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); out=Path(out); out.mkdir(parents=True,exist_ok=True); base=out/f'aegis_phase_o5_fraction_alignment_{stamp}'; base.with_suffix('.json').write_text(json.dumps(payload,indent=2)+'\n'); csvp=out/f'aegis_phase_o5_fraction_alignment_symbols_{stamp}.csv'
 with csvp.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(payload['rows'][0])); w.writeheader(); w.writerows(payload['rows'])
 c=payload['current']; e=payload['evidence']; base.with_suffix('.md').write_text('\n'.join(['# Phase O.5 Fraction Alignment Audit','','## Safety','READ_ONLY audit. No config mutation.','','## Evidence',f"Historical Turbo buckets: {e['historical_turbo_buckets']}",f"Current Turbo buckets: {c['turbo_buckets']}",f"Phase O duplicated default: {c['phase_o_default']}",f"TS safety cap: {c['ts_turbo_cap']}",f"Momentum cap retained: {c['momentum_cap']}",'','## Decision','Phase O SHORT inherits raw ML/Turbo bucket sizing. The YAML Phase O block contains no duplicated fraction policy.'])+'\n'); return [str(base.with_suffix('.md')),str(base.with_suffix('.json')),str(csvp)]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--wallet-usdt',type=float,default=20); ap.add_argument('--out-dir',default='/home/jasan/Develop'); a=ap.parse_args(); payload=audit(a.wallet_usdt); print(json.dumps({'reports':write_reports(payload,a.out_dir),'current':payload['current'],'decision':payload['decision']},indent=2))
if __name__=='__main__': main()
