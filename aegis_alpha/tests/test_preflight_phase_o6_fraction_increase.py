#!/usr/bin/env python3
import json,sys,tempfile
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_phase_o5_fraction_alignment import ENTRY_SYMBOLS
from preflight_phase_o6_fraction_increase import validate

def fixture():
 root=Path(tempfile.mkdtemp()); (root/'aegis_alpha/configs').mkdir(parents=True); (root/'binance-futures-bot-ts').mkdir()
 turbo={'sizing':{'conservative':{'position_fraction':.20,'leverage':15.0},'normal':{'position_fraction':.35,'leverage':20.0},'premium':{'position_fraction':.50,'leverage':25.0},'max_allowed_position_fraction':.50,'max_allowed_leverage':30.0}}
 phase={'enabled':True,'allow_orders':True,'require_brackets':True,'max_open_phase_o_positions':1,'max_phase_o_trades_per_day':3,'allow_link_entry':False,'link_avoid_only':True,'leverage':{'conservative':15,'normal':20,'premium':25,'max_allowed_leverage':30},'symbols':{s:{'enabled':True} for s in ENTRY_SYMBOLS},'hard_safety':{k:'ENFORCE' for k in ['brackets','max_open_positions','max_trades_per_day','daily_loss_stop','exchange_min_notional','exchange_order_errors','link_no_entry']}}
 phase['symbols']['LINKUSDT']={'enabled':True,'entry_enabled':False,'avoid_only':True}
 cfg={'aegis':{'phase_o_short_live':phase,'turbo':{'position_fraction_cap':.50,'max_trades_per_day':3,'max_consecutive_losses':2,'daily_loss_stop_pct':.10}}}
 (root/'aegis_alpha/configs/turbo.yaml').write_text(yaml.safe_dump(turbo)); (root/'binance-futures-bot-ts/regime_config.live.yaml').write_text(yaml.safe_dump(cfg))
 for symbol in ENTRY_SYMBOLS:
  d=root/'aegis_alpha/models/turbo'/symbol; d.mkdir(parents=True); model=d/f'phase_o_test_{symbol}.joblib'; model.write_text('fake'); (d/'active_manifest.json').write_text(json.dumps({'phase_o_live_enabled':True,'model_paths':{'long_30d':'old-long','short_30d':str(model)},'pre_phase_o_live_model_paths':{'long_30d':'old-long','short_30d':'old-short'}}))
 d=root/'aegis_alpha/models/turbo/LINKUSDT'; d.mkdir(parents=True); (d/'active_manifest.json').write_text(json.dumps({'phase_o_link_entry_enabled':False,'phase_o_avoid_only':True,'model_paths':{'long_30d':'old-long'}}))
 return root

def yaml_mut(root,fn,path='binance-futures-bot-ts/regime_config.live.yaml'):
 p=root/path; d=yaml.safe_load(p.read_text()); fn(d); p.write_text(yaml.safe_dump(d))
def fail(fn):
 root=fixture(); fn(root); assert validate(root)['status']=='FAILED'
def main():
 result=validate(fixture(),20); assert result['status']=='PASSED'; rows={r['bucket']:r for r in result['sizing']}
 assert rows['conservative']['margin_usdt']==4 and rows['conservative']['notional_usdt']==60
 assert rows['normal']['margin_usdt']==7 and rows['normal']['notional_usdt']==140
 assert rows['premium']['margin_usdt']==10 and rows['premium']['notional_usdt']==250
 assert any('HIGH_RISK_TEST_CAPITAL' in w for w in result['warnings'])
 fail(lambda r: yaml_mut(r,lambda d:d['sizing']['premium'].__setitem__('position_fraction',.51),'aegis_alpha/configs/turbo.yaml'))
 fail(lambda r: yaml_mut(r,lambda d:d['aegis']['turbo'].__setitem__('position_fraction_cap',.51)))
 fail(lambda r: yaml_mut(r,lambda d:d['aegis']['turbo'].__setitem__('position_fraction_cap',.49)))
 fail(lambda r: yaml_mut(r,lambda d:d['aegis']['phase_o_short_live'].__setitem__('allow_link_entry',True)))
 fail(lambda r: yaml_mut(r,lambda d:d['aegis']['phase_o_short_live'].__setitem__('require_brackets',False)))
 fail(lambda r: yaml_mut(r,lambda d:d['aegis']['phase_o_short_live'].__setitem__('max_open_phase_o_positions',2)))
 fail(lambda r: yaml_mut(r,lambda d:d['aegis']['phase_o_short_live'].__setitem__('max_phase_o_trades_per_day',4)))
 print('PASS test_preflight_phase_o6_fraction_increase')
if __name__=='__main__': main()
