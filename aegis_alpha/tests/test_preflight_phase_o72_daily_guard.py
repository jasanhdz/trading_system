#!/usr/bin/env python3
import json,sys,tempfile
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'aegis_alpha/tools'))
from preflight_phase_o72_daily_guard import validate,reports

def fixture(signal_block=False):
 root=Path(tempfile.mkdtemp()); (root/'aegis_alpha/turbo').mkdir(parents=True); (root/'aegis_alpha/configs').mkdir(parents=True); (root/'binance-futures-bot-ts/src/app/services').mkdir(parents=True); (root/'binance-futures-bot-ts/src/infra/config').mkdir(parents=True); (root/'binance-futures-bot-ts/logs/aegis').mkdir(parents=True); (root/'aegis_alpha/models/turbo/LINKUSDT').mkdir(parents=True)
 py='count_today_turbo_signals(history) >= cfg.max_turbo_trades_per_day' if signal_block else 'count_today_turbo_opened_trades(trade_events) >= cfg.max_turbo_trades_per_day\n"count_source": "position_confirmed"\n"today_trade_count"'
 (root/'aegis_alpha/turbo/turbo_risk.py').write_text(py)
 (root/'aegis_alpha/configs/turbo.yaml').write_text(yaml.safe_dump({'risk':{'max_turbo_trades_per_day':20}}))
 phase={'enabled':True,'max_phase_o_trades_per_day':20,'max_open_phase_o_positions':9,'require_brackets':True,'allow_link_entry':False,'link_avoid_only':True,'hard_safety':{k:'ENFORCE' for k in ['brackets','max_open_positions','max_trades_per_day','daily_loss_stop','exchange_min_notional','exchange_order_errors','link_no_entry']}}
 cfg={'aegis':{'phase_o_short_live':phase,'turbo':{'max_trades_per_day':20,'daily_loss_stop_pct':.30}}}
 (root/'binance-futures-bot-ts/regime_config.live.yaml').write_text(yaml.safe_dump(cfg))
 (root/'binance-futures-bot-ts/src/app/services/TradingService.ts').write_text("phaseOShortTradesToday risk_guard_max_phase_o_trades_per_day countSource: 'trade_opened' max_phase_o_trades_per_day metadata?.aegis?.turbo")
 (root/'binance-futures-bot-ts/src/infra/config/ConfigLoader.ts').write_text('getAegisPhaseOShortLiveConfig')
 (root/'aegis_alpha/models/turbo/LINKUSDT/active_manifest.json').write_text(json.dumps({'phase_o_link_entry_enabled':False,'phase_o_avoid_only':True,'model_paths':{'short_30d':'base'}}))
 return root

def main():
 ok=validate(fixture(False)); assert ok['status']=='PASSED'; assert ok['count_source']=='position_confirmed'; assert len(reports(ok,Path(tempfile.mkdtemp())))==2
 bad=validate(fixture(True)); assert bad['status']=='FAILED'; assert any('today_signal_count' in e for e in bad['errors'])
 link=fixture(False); p=link/'aegis_alpha/models/turbo/LINKUSDT/active_manifest.json'; data=json.loads(p.read_text()); data['phase_o_link_entry_enabled']=True; p.write_text(json.dumps(data)); assert validate(link)['status']=='FAILED'
 print('PASS test_preflight_phase_o72_daily_guard')
if __name__=='__main__': main()
