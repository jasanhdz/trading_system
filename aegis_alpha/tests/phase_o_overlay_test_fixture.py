from __future__ import annotations
import json
from pathlib import Path
ENTRY=('LTCUSDT','AVAXUSDT','ETHUSDT','SUIUSDT','ADAUSDT','DOGEUSDT','BTCUSDT','BNBUSDT','XRPUSDT','SOLUSDT')
ALL=ENTRY+('LINKUSDT',)
LOOKBACK={'LTCUSDT':14,'AVAXUSDT':14,'SUIUSDT':7}
def dump(path,payload): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2))
def fixture(root: Path):
 base=root/'aegis_alpha/models/turbo'; stamp='20260601T070114Z'; globalp=base/f'phase_o_global_short_manifest_{stamp}.json'; dump(base/'phase_o_short_manifest.json',{'latest_manifest':str(globalp),'latest_artifact_stamp':stamp,'entry_symbols':list(ENTRY),'avoid_only_symbols':['LINKUSDT']}); dump(globalp,{'artifact_stamp':stamp,'symbols':list(ALL),'entry_symbols':list(ENTRY),'avoid_only_symbols':['LINKUSDT']})
 for symbol in ALL:
  days=LOOKBACK.get(symbol,30); art=base/symbol/'backups'/'old'/f'phase_o_{stamp}'; art.mkdir(parents=True)
  if symbol=='LINKUSDT': files=[]
  else:
   m=art/f'turbo_short_edge_{days}d_phase_o_{stamp}.joblib'; m.write_text('fake'); files=[str(m)]
  dump(art/'symbol_shadow_manifest.json',{'symbol':symbol,'lookback_days':days,'shadow_type':'avoid_only_filter' if symbol=='LINKUSDT' else 'entry_model','model_files':files})
  if symbol=='LINKUSDT':
   for name in ('micro_hit_classifier.joblib','micro_quality_regressor.joblib','micro_danger_classifier.joblib'): (art/name).write_text('fake')
  active=base/symbol/'active_manifest.json'; dump(active,{'symbol':symbol,'model_paths':{'long_30d':f'{symbol}-long','short_30d':f'{symbol}-base-short'}})
 return base
