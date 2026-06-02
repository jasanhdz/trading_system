#!/usr/bin/env python3
import json,sys,tempfile
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_phase_o5_fraction_alignment import ENTRY_SYMBOLS
from preflight_phase_o7_aggressive_mode import validate

def fixture():
 root=Path(tempfile.mkdtemp()); (root/'aegis_alpha/configs').mkdir(parents=True); (root/'binance-futures-bot-ts').mkdir(); (root/'aegis_alpha/tools').mkdir()
 (root/'aegis_alpha/tools/rollback_phase_o_short_live_o2.py').write_text('rollback')
 (root/'aegis_alpha/tools/run_turbo_scheduled_retrain.py').write_text('apply_phase_o_overlay_to_active_manifest --disable-phase-o-overlay')
 turbo={'sizing':{'conservative':{'position_fraction':.20,'leverage':15.0},'normal':{'position_fraction':.35,'leverage':20.0},'premium':{'position_fraction':.50,'leverage':25.0},'max_allowed_position_fraction':.50,'max_allowed_leverage':30.0},'risk':{'max_turbo_trades_per_day':20,'max_consecutive_losses':3,'daily_loss_stop_pct':30.0}}
 phase={'enabled':True,'allow_orders':True,'require_brackets':True,'max_open_phase_o_positions':9,'max_phase_o_trades_per_day':20,'allow_link_entry':False,'link_avoid_only':True,'leverage':{'conservative':15,'normal':20,'premium':25,'max_allowed_leverage':30},'symbols':{s:{'enabled':True} for s in ENTRY_SYMBOLS},'hard_safety':{k:'ENFORCE' for k in ['brackets','max_open_positions','max_trades_per_day','daily_loss_stop','exchange_min_notional','exchange_order_errors','link_no_entry']}}
 phase['symbols']['LINKUSDT']={'enabled':True,'entry_enabled':False,'avoid_only':True}
 cfg={'aegis':{'phase_o_short_live':phase,'turbo':{'position_fraction_cap':.50,'max_trades_per_day':20,'max_consecutive_losses':3,'daily_loss_stop_pct':.30}}}
 (root/'aegis_alpha/configs/turbo.yaml').write_text(yaml.safe_dump(turbo)); (root/'binance-futures-bot-ts/regime_config.live.yaml').write_text(yaml.safe_dump(cfg))
 for symbol in ENTRY_SYMBOLS:
  d=root/'aegis_alpha/models/turbo'/symbol; d.mkdir(parents=True); model=d/f'phase_o_test_{symbol}.joblib'; model.write_text('fake'); (d/'active_manifest.json').write_text(json.dumps({'phase_o_live_enabled':True,'phase_o_overlay_persistence_enabled':True,'phase_o_live_artifact_stamp':'test','model_paths':{'long_30d':'old-long','short_30d':str(model)},'pre_phase_o_live_model_paths':{'long_30d':'old-long','short_30d':'old-short'}}))
 d=root/'aegis_alpha/models/turbo/LINKUSDT'; d.mkdir(parents=True); (d/'active_manifest.json').write_text(json.dumps({'phase_o_link_entry_enabled':False,'phase_o_avoid_only':True,'phase_o_overlay_persistence_enabled':True,'phase_o_live_artifact_stamp':'test','model_paths':{'long_30d':'old-long'}}))
 return root

def mutate(root,fn,path='binance-futures-bot-ts/regime_config.live.yaml'):
 p=root/path; d=yaml.safe_load(p.read_text()); fn(d); p.write_text(yaml.safe_dump(d))
def fails(fn):
 root=fixture(); fn(root); assert validate(root)['status']=='FAILED'
def main():
 r=validate(fixture(),20); assert r['status']=='PASSED'; rows={x['bucket']:x for x in r['exposure']}
 assert rows['conservative']['worst_case_margin_usdt']==36
 assert rows['normal']['worst_case_margin_usdt']==63
 assert rows['premium']['worst_case_margin_usdt']==90
 assert 'HIGH_RISK_AGGRESSIVE_MODE' in r['warnings']
 fails(lambda root: mutate(root,lambda d:d['aegis']['phase_o_short_live'].__setitem__('max_open_phase_o_positions',10)))
 fails(lambda root: mutate(root,lambda d:d['aegis']['phase_o_short_live'].__setitem__('max_phase_o_trades_per_day',21)))
 fails(lambda root: mutate(root,lambda d:d['aegis']['turbo'].__setitem__('max_consecutive_losses',4)))
 fails(lambda root: mutate(root,lambda d:d['aegis']['turbo'].__setitem__('daily_loss_stop_pct',.31)))
 fails(lambda root: mutate(root,lambda d:d['aegis']['phase_o_short_live'].__setitem__('allow_link_entry',True)))
 fails(lambda root: mutate(root,lambda d:d['aegis']['phase_o_short_live'].__setitem__('require_brackets',False)))
 fails(lambda root: mutate(root,lambda d:d['aegis']['turbo'].__setitem__('position_fraction_cap',.51)))
 fails(lambda root: mutate(root,lambda d:d['aegis']['phase_o_short_live']['leverage'].__setitem__('max_allowed_leverage',31)))
 print('PASS test_preflight_phase_o7_aggressive_mode')
if __name__=='__main__': main()
