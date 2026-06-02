#!/usr/bin/env python3
"""Read-only audit for the Phase O / Turbo daily trade guard semantics."""
from __future__ import annotations
import argparse,csv,glob,json,subprocess
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ENTRY_SYMBOLS={'LTCUSDT','AVAXUSDT','ETHUSDT','SUIUSDT','ADAUSDT','DOGEUSDT','BTCUSDT','BNBUSDT','XRPUSDT','SOLUSDT'}
LINK='LINKUSDT'
def parse_time(raw):
 if not raw or raw=='now': return datetime.now(timezone.utc)
 return datetime.fromisoformat(raw.replace('Z','+00:00')).astimezone(timezone.utc)
def parse_row_dt(raw):
 raw=str(raw or '')
 try:
  if len(raw)==16 and raw.endswith('Z'): return datetime.strptime(raw,'%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
  dt=datetime.fromisoformat(raw.replace('Z','+00:00'))
  return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
 except Exception: return None
def file_in_range(path,start,end):
 token=Path(path).stem.rsplit('_',1)[-1]
 try:
  fmt='%Y-%m-%d' if '-' in token else '%Y%m%d'; d=datetime.strptime(token,fmt).date(); return start.date()<=d<=end.date()
 except Exception: return True
def iter_jsonl(pattern,start,end):
 for name in sorted(glob.glob(str(pattern))):
  if not file_in_range(name,start,end): continue
  with Path(name).open(errors='replace') as handle:
   for line in handle:
    try: row=json.loads(line)
    except json.JSONDecodeError: continue
    if isinstance(row,dict): yield row
def _src(root,path): return (root/path).read_text() if (root/path).exists() else ''
def _head_source(root):
 try: return subprocess.check_output(['git','show','HEAD:aegis_alpha/turbo/turbo_risk.py'],cwd=root,text=True,stderr=subprocess.DEVNULL)
 except Exception: return ''
def audit(root=ROOT,from_time='2026-06-01T19:00:00Z',to_time='now'):
 root=Path(root); start=parse_time(from_time); end=parse_time(to_time); today=end.strftime('%Y-%m-%d')
 src=_src(root,Path('aegis_alpha/turbo/turbo_risk.py')); head_src=_head_source(root)
 signal_block='count_today_turbo_signals(history) >= cfg.max_turbo_trades_per_day' in src
 trade_block='count_today_turbo_opened_trades(trade_events) >= cfg.max_turbo_trades_per_day' in src
 legacy_signal_block='count_today_turbo_signals(history) >= cfg.max_turbo_trades_per_day' in head_src
 ec=Counter(); perday=defaultdict(Counter); persymbol=defaultdict(Counter); opened_ids=set(); orders=set(); phase_ids=set(); total_opened_ids=set(); symbols_opened=set(); link_entries=0
 for row in iter_jsonl(root/'binance-futures-bot-ts/logs/aegis/turbo_trade_events_*.jsonl',start,end):
  dt=parse_row_dt(row.get('timestamp'))
  if dt is None or not (start<=dt<=end): continue
  date=dt.strftime('%Y-%m-%d'); event=str(row.get('event') or ''); symbol=str(row.get('symbol') or 'UNKNOWN').upper(); meta=row.get('metadata') if isinstance(row.get('metadata'),dict) else {}; side=str(meta.get('side') or row.get('side') or '').upper(); trade_id=str(row.get('trade_id') or '').strip()
  ec[event]+=1; perday[date][event]+=1; persymbol[symbol][event]+=1
  if 'max_trades_per_day' in str(row.get('reason','')): perday[date]['DAILY_LIMIT_DENIED']+=1; persymbol[symbol]['denied_by_daily_limit']+=1
  if symbol==LINK and event in {'ORDER_SUBMITTED','POSITION_CONFIRMED'}: link_entries+=1; persymbol[symbol]['link_entry_attempts']+=1
  if date==today and event=='POSITION_CONFIRMED' and trade_id: opened_ids.add(trade_id)
  if date==today and event=='ORDER_SUBMITTED' and trade_id: orders.add(trade_id)
  if event=='POSITION_CONFIRMED':
   symbols_opened.add(symbol)
   if trade_id: total_opened_ids.add(trade_id)
  if date==today and event=='POSITION_CONFIRMED' and side=='SHORT' and symbol in ENTRY_SYMBOLS and trade_id: phase_ids.add(trade_id)
  if event=='POSITION_CONFIRMED' and side=='SHORT' and symbol in ENTRY_SYMBOLS: persymbol[symbol]['phase_o_short_opened']+=1
 inferred_signal=0
 for row in iter_jsonl(root/'aegis_alpha/logs/turbo/turbo_shadow_*.jsonl',start,end):
  dt=parse_row_dt(row.get('timestamp')) or parse_row_dt(row.get('logged_at'))
  if dt is None or not (start<=dt<=end): continue
  date=dt.strftime('%Y-%m-%d'); symbol=str(row.get('symbol') or 'UNKNOWN').upper(); raw=row.get('raw') if isinstance(row.get('raw'),dict) else {}; phase=raw.get('phase_o') if isinstance(raw.get('phase_o'),dict) else {}
  if date==today and bool(row.get('would_execute')): inferred_signal+=1
  if phase.get('phase_o_live_enabled') is True:
   perday[date]['PHASE_O_SIGNAL']+=1; persymbol[symbol]['phase_o_signals']+=1
   if str(raw.get('action') or '').upper()=='SHORT': perday[date]['PHASE_O_SHORT_SIGNAL']+=1; persymbol[symbol]['phase_o_short_signals']+=1
 classification='DAILY_GUARD_TOO_STRICT_COUNTS_SIGNALS' if signal_block else ('DAILY_GUARD_OK_COUNTS_REAL_TRADES' if trade_block else 'DAILY_GUARD_UNKNOWN')
 days=[{'date':date,'signal_received_count':c['SIGNAL_RECEIVED'],'phase_o_signal_count':c['PHASE_O_SIGNAL'],'phase_o_short_signal_count':c['PHASE_O_SHORT_SIGNAL'],'order_submitted_count':c['ORDER_SUBMITTED'],'trade_opened_count':c['POSITION_CONFIRMED'],'daily_limit_denied_count':c['DAILY_LIMIT_DENIED'],'configured_limit':20} for date,c in sorted(perday.items())]
 symbols=[{'symbol':symbol,'phase_o_signals':c['phase_o_signals'],'phase_o_short_signals':c['phase_o_short_signals'],'denied_by_daily_limit':c['denied_by_daily_limit'],'orders_submitted':c['ORDER_SUBMITTED'],'trades_opened':c['POSITION_CONFIRMED'],'trades_closed':c['TRADE_CLOSED'],'link_entry_attempts':c['link_entry_attempts'],'phase_o_short_opened':c['phase_o_short_opened']} for symbol,c in sorted(persymbol.items())]
 return {'schema_version':'aegis_phase_o72_daily_guard_audit_v1','created_at':datetime.now(timezone.utc).isoformat(),'from':start.isoformat(),'to':end.isoformat(),'classification':classification,'false_positive_block':signal_block and inferred_signal>=20 and len(opened_ids)<20,'legacy_bug_detected_in_head':legacy_signal_block,'legacy_false_positive_before_fix':legacy_signal_block and inferred_signal>=20 and len(opened_ids)<20,'code_findings':{'guard_file':'aegis_alpha/turbo/turbo_risk.py','blocks_on_today_signal_count':signal_block,'blocks_on_position_confirmed_trade_count':trade_block,'legacy_head_blocks_on_today_signal_count':legacy_signal_block,'current_guard_counter_name':'today_signal_count' if signal_block else 'today_trade_count','desired_count_source':'POSITION_CONFIRMED'},'metrics':{'signal_received_count':ec['SIGNAL_RECEIVED'],'phase_o_signal_count':sum(c['PHASE_O_SIGNAL'] for c in perday.values()),'phase_o_short_signal_count':sum(c['PHASE_O_SHORT_SIGNAL'] for c in perday.values()),'denied_count':ec['GATE_DENIED']+ec['DECISION_ENFORCEMENT_DENIED'],'denied_by_daily_limit_count':sum(c['DAILY_LIMIT_DENIED'] for c in perday.values()),'gate_denied_count':ec['GATE_DENIED'],'decision_denied_count':ec['DECISION_ENFORCEMENT_DENIED'],'order_submitted_count':ec['ORDER_SUBMITTED'],'order_failed_count':ec['ORDER_FAILED'],'brackets_confirmed_count':ec['BRACKETS_CONFIRMED'],'trade_opened_count':ec['POSITION_CONFIRMED'],'trade_closed_count':ec['TRADE_CLOSED'],'unique_trade_ids_opened':len(total_opened_ids),'unique_symbols_traded':len(symbols_opened),'LINK_entry_attempts':link_entries,'current_guard_counter_value':inferred_signal if signal_block else len(opened_ids),'inferred_signal_count_today':inferred_signal,'inferred_order_count_today':len(orders),'inferred_trade_count_today':len(opened_ids),'inferred_phase_o_short_trade_count_today':len(phase_ids),'mismatch_detected':inferred_signal!=len(opened_ids)},'days':days,'symbols':symbols}
def write_reports(payload,out):
 out=Path(out); out.mkdir(parents=True,exist_ok=True); stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); base=out/f'aegis_phase_o72_daily_guard_audit_{stamp}'; jp=base.with_suffix('.json'); mp=base.with_suffix('.md'); cp=out/f'aegis_phase_o72_daily_guard_counts_{stamp}.csv'; sp=out/f'aegis_phase_o72_daily_guard_symbols_{stamp}.csv'; jp.write_text(json.dumps(payload,indent=2)+'\n')
 with cp.open('w',newline='') as f:
  fields=list(payload['days'][0]) if payload['days'] else ['date']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(payload['days'])
 with sp.open('w',newline='') as f:
  fields=list(payload['symbols'][0]) if payload['symbols'] else ['symbol']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(payload['symbols'])
 m=payload['metrics']; lines=['# Phase O.7.2 Daily Guard Audit','',f"Classification: **{payload['classification']}**",f"Legacy false positive before fix: **{payload['legacy_false_positive_before_fix']}**",'', '## Current Day',f"- Inferred model signals: `{m['inferred_signal_count_today']}`",f"- Submitted orders: `{m['inferred_order_count_today']}`",f"- Confirmed opened trades: `{m['inferred_trade_count_today']}`",f"- Confirmed Phase O SHORT trades: `{m['inferred_phase_o_short_trade_count_today']}`",'', '## Code',f"- Blocks on scans now: `{payload['code_findings']['blocks_on_today_signal_count']}`",f"- Blocks on POSITION_CONFIRMED now: `{payload['code_findings']['blocks_on_position_confirmed_trade_count']}`",f"- HEAD before uncommitted fix blocked on scans: `{payload['code_findings']['legacy_head_blocks_on_today_signal_count']}`"]
 mp.write_text('\n'.join(lines)+'\n'); return [str(mp),str(jp),str(cp),str(sp)]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--from',dest='from_time',default='2026-06-01T19:00:00Z'); ap.add_argument('--to',dest='to_time',default='now'); ap.add_argument('--out-dir',default='/home/jasan/Develop'); ap.add_argument('--symbol',default='ALL'); ap.add_argument('--phase-o-only',action='store_true'); ap.add_argument('--include-denied',action='store_true'); ap.add_argument('--include-orders',action='store_true'); ap.add_argument('--include-trades',action='store_true'); a=ap.parse_args(); payload=audit(from_time=a.from_time,to_time=a.to_time); print(json.dumps({'reports':write_reports(payload,a.out_dir),'classification':payload['classification'],'legacy_false_positive_before_fix':payload['legacy_false_positive_before_fix'],'metrics':payload['metrics']},indent=2))
if __name__=='__main__': main()
