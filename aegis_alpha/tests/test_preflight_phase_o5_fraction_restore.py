#!/usr/bin/env python3
import json,sys,tempfile
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_phase_o5_fraction_alignment import ENTRY_SYMBOLS
from preflight_phase_o5_fraction_restore import validate

def fixture():
 root=Path(tempfile.mkdtemp()); (root/'aegis_alpha/configs').mkdir(parents=True); (root/'binance-futures-bot-ts').mkdir()
 turbo={'sizing':{'conservative':{'position_fraction':.08,'leverage':15.0},'normal':{'position_fraction':.12,'leverage':20.0},'premium':{'position_fraction':.18,'leverage':25.0},'max_allowed_position_fraction':.20,'max_allowed_leverage':30.0}}
 phase={'enabled':True,'allow_orders':True,'require_brackets':True,'max_open_phase_o_positions':1,'max_phase_o_trades_per_day':3,'allow_link_entry':False,'link_avoid_only':True,'leverage':{'conservative':15,'normal':20,'premium':25,'max_allowed_leverage':30},'symbols':{s:{'enabled':True} for s in ENTRY_SYMBOLS},'hard_safety':{k:'ENFORCE' for k in ['brackets','max_open_positions','max_trades_per_day','daily_loss_stop','exchange_min_notional','exchange_order_errors','link_no_entry']}}
 phase['symbols']['LINKUSDT']={'enabled':True,'entry_enabled':False,'avoid_only':True}
 cfg={'aegis':{'phase_o_short_live':phase,'turbo':{'position_fraction_cap':.20,'max_trades_per_day':3,'max_consecutive_losses':2,'daily_loss_stop_pct':.10}}}
 (root/'aegis_alpha/configs/turbo.yaml').write_text(yaml.safe_dump(turbo)); (root/'binance-futures-bot-ts/regime_config.live.yaml').write_text(yaml.safe_dump(cfg))
 for s in ENTRY_SYMBOLS:
  d=root/'aegis_alpha/models/turbo'/s; d.mkdir(parents=True); model=d/f'phase_o_test_{s}.joblib'; model.write_text('fake'); (d/'active_manifest.json').write_text(json.dumps({'phase_o_live_enabled':True,'model_paths':{'long_30d':'old-long','short_30d':str(model)},'pre_phase_o_live_model_paths':{'long_30d':'old-long','short_30d':'old-short'}}))
 d=root/'aegis_alpha/models/turbo/LINKUSDT'; d.mkdir(parents=True); (d/'active_manifest.json').write_text(json.dumps({'phase_o_link_entry_enabled':False,'phase_o_avoid_only':True,'model_paths':{'long_30d':'old-long'}}))
 return root

def assert_fail(mutator):
 root=fixture(); mutator(root); assert validate(root)['status']=='FAILED'
def yaml_mut(root,fn):
 p=root/'binance-futures-bot-ts/regime_config.live.yaml'; d=yaml.safe_load(p.read_text()); fn(d); p.write_text(yaml.safe_dump(d))
def test_pass_and_sizing():
 p=validate(fixture(),20); assert p['status']=='PASSED'; row=p['sizing'][0]; assert round(row['margin_normal'],8)==2.4 and round(row['notional_25x_premium'],8)==90
def main():
 test_pass_and_sizing()
 assert_fail(lambda r: yaml_mut(r,lambda d:d['aegis']['phase_o_short_live'].__setitem__('allow_link_entry',True)))
 assert_fail(lambda r: yaml_mut(r,lambda d:d['aegis']['phase_o_short_live'].__setitem__('require_brackets',False)))
 assert_fail(lambda r: yaml_mut(r,lambda d:d['aegis']['phase_o_short_live'].__setitem__('max_open_phase_o_positions',2)))
 assert_fail(lambda r: yaml_mut(r,lambda d:d['aegis']['turbo'].__setitem__('max_trades_per_day',4)))
 assert_fail(lambda r: yaml_mut(r,lambda d:d['aegis']['turbo'].__setitem__('position_fraction_cap',.10)))
 assert_fail(lambda r: yaml_mut(r,lambda d:d['aegis']['phase_o_short_live'].__setitem__('max_position_fraction_default',.12)))
 assert_fail(lambda r: yaml_mut(r,lambda d:d['aegis']['phase_o_short_live']['leverage'].__setitem__('max_allowed_leverage',31)))
 print('PASS test_preflight_phase_o5_fraction_restore')
if __name__=='__main__': main()
