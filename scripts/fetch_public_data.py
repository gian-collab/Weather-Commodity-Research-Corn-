from __future__ import annotations
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
import json
from pathlib import Path
import yaml
from weatherlab.data.corn_market import fetch_public_corn_market
from weatherlab.data.usda import quickstats,corn_yield_query,corn_production_query
from weatherlab.data.open_meteo import fetch_reanalysis_daily, fetch_previous_runs

cfg=yaml.safe_load(Path('configs/full.yaml').read_text())
out={'market':fetch_public_corn_market(start=cfg['market']['start'],symbol=cfg['market']['symbol'])}
out['usda_yield']=quickstats(corn_yield_query(1980),'data/raw/usda/corn_yield.parquet')
out['usda_production']=quickstats(corn_production_query(1980),'data/raw/usda/corn_production.parquet')
out['weather_realized']=[]; out['weather_forecasts']=[]
for i,(name,(lat,lon)) in enumerate(cfg['corn_belt']['regions'].items()):
    slug=name.lower().replace(' ','_')
    try: out['weather_realized'].append({'region':name,**fetch_reanalysis_daily(lat,lon,cfg['weather']['realized_start'],cfg['data_cutoff'],f'data/raw/open_meteo/reanalysis/{slug}.parquet',pause_seconds=cfg['weather'].get('request_pause_seconds',1.0))})
    except Exception as e: out['weather_realized'].append({'status':'UNAVAILABLE','region':name,'error':str(e)})
    try: out['weather_forecasts'].append({'region':name,**fetch_previous_runs(lat,lon,cfg['weather']['forecast_start'],cfg['data_cutoff'],f'data/raw/open_meteo/previous_runs/{slug}.parquet',cfg['weather']['horizons_days'],cfg['weather'].get('open_meteo_model'),pause_seconds=cfg['weather'].get('request_pause_seconds',1.0))})
    except Exception as e: out['weather_forecasts'].append({'status':'UNAVAILABLE','region':name,'error':str(e)})
Path('outputs').mkdir(exist_ok=True)
Path('outputs/fetch_report.json').write_text(json.dumps(out,indent=2,default=str))
print(json.dumps(out,indent=2,default=str))
