#!/usr/bin/env python3
from datetime import datetime,timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from types import SimpleNamespace
from unittest.mock import patch
from aegis_alpha.turbo import turbo_risk

def cfg(limit=2): return SimpleNamespace(risk=SimpleNamespace(max_turbo_trades_per_day=limit,max_consecutive_losses=99))
def ts(): return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
def main():
 rows=[{'timestamp':ts(),'would_execute':True} for _ in range(50)]
 events=[]
 with patch('aegis_alpha.turbo.turbo_risk.get_runtime_turbo_config',return_value=cfg(2)):
  blocked,reason=turbo_risk.should_block_turbo_today(rows,events); assert not blocked and reason is None
  events=[{'timestamp':ts(),'event':'POSITION_CONFIRMED','trade_id':'a'},{'timestamp':ts(),'event':'POSITION_CONFIRMED','trade_id':'a'},{'timestamp':ts(),'event':'GATE_DENIED','trade_id':'b'}]
  assert turbo_risk.count_today_turbo_opened_trades(events)==1
  events.append({'timestamp':ts(),'event':'POSITION_CONFIRMED','trade_id':'b'})
  blocked,reason=turbo_risk.should_block_turbo_today(rows,events); assert blocked and reason=='max_turbo_trades_per_day'
  status=turbo_risk.build_turbo_risk_status(rows,events); assert status['count_source']=='position_confirmed'; assert status['today_signal_count']==50; assert status['today_trade_count']==2
 print('PASS test_turbo_risk_daily_trade_guard')
if __name__=='__main__': main()
