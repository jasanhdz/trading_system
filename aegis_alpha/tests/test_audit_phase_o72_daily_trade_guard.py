#!/usr/bin/env python3
import json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'aegis_alpha/tools'))
from audit_phase_o72_daily_trade_guard import audit,write_reports

def line(path,obj): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj)+'\n',append=False) if False else path.open('a').write(json.dumps(obj)+'\n')
def fixture(old_guard=True):
 root=Path(tempfile.mkdtemp()); (root/'aegis_alpha/turbo').mkdir(parents=True); (root/'binance-futures-bot-ts/logs/aegis').mkdir(parents=True); (root/'aegis_alpha/logs/turbo').mkdir(parents=True)
 src='count_today_turbo_signals(history) >= cfg.max_turbo_trades_per_day' if old_guard else 'count_today_turbo_opened_trades(trade_events) >= cfg.max_turbo_trades_per_day'
 (root/'aegis_alpha/turbo/turbo_risk.py').write_text(src)
 shadow=root/'aegis_alpha/logs/turbo/turbo_shadow_20260602.jsonl'
 for i in range(25): line(shadow,{'logged_at':'20260602T010000Z','timestamp':'2026-06-02T01:00:00Z','symbol':'SOLUSDT','would_execute':True,'raw':{'action':'SHORT','phase_o':{'phase_o_live_enabled':True}}})
 events=root/'binance-futures-bot-ts/logs/aegis/turbo_trade_events_2026-06-02.jsonl'
 line(events,{'timestamp':'2026-06-02T01:01:00Z','symbol':'LINKUSDT','event':'GATE_DENIED','reason':'avoid_only','metadata':{'side':'SHORT'}})
 line(events,{'timestamp':'2026-06-02T01:02:00Z','symbol':'SOLUSDT','event':'GATE_DENIED','reason':'risk_guard_max_turbo_trades_per_day','metadata':{'side':'SHORT'}})
 return root

def main():
 old=audit(fixture(True),'2026-06-02T00:00:00Z','2026-06-02T02:00:00Z'); assert old['classification']=='DAILY_GUARD_TOO_STRICT_COUNTS_SIGNALS'; assert old['false_positive_block'] is True; assert old['metrics']['inferred_signal_count_today']==25; assert old['metrics']['inferred_trade_count_today']==0; assert old['metrics']['LINK_entry_attempts']==0
 new=audit(fixture(False),'2026-06-02T00:00:00Z','2026-06-02T02:00:00Z'); assert new['classification']=='DAILY_GUARD_OK_COUNTS_REAL_TRADES'; assert new['metrics']['phase_o_short_signal_count']==25; reports=write_reports(new,Path(tempfile.mkdtemp())); assert len(reports)==4 and all(Path(p).exists() for p in reports)
 print('PASS test_audit_phase_o72_daily_trade_guard')
if __name__=='__main__': main()
