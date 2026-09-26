from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
"""Run hardened real-data studies after `scripts/fetch_public_data.py`.
Reanalysis is ex-post truth/physical data only; it is never substituted for a historical forecast vintage.
"""
from pathlib import Path
import json, pandas as pd, yaml
from weatherlab.data.storage import read_table
from weatherlab.data.usda import normalize_annual
from weatherlab.data.open_meteo import add_run_revisions
from weatherlab.data.corn_market import forward_returns
from weatherlab.features.weather import aggregate_growing_season
from weatherlab.experiments.real import run_market_revision_association
from weatherlab.experiments.yield_nested import run_nested_yield_oos
from weatherlab.experiments.weather_verification import verify_archived_forecasts, revision_instability_skill
from weatherlab.timekeys import normalize_region_name

cfg=yaml.safe_load(Path('configs/full.yaml').read_text()); regions=cfg['corn_belt']['regions']
Path('outputs/results').mkdir(parents=True,exist_ok=True); results={}
realized=[]; archived=[]
for name in regions:
    slug=name.lower().replace(' ','_')
    try:
        d=read_table(f'data/raw/open_meteo/reanalysis/{slug}.parquet'); d['region']=normalize_region_name(name); realized.append(d)
    except FileNotFoundError: pass
    try:
        d=read_table(f'data/raw/open_meteo/previous_runs/{slug}.parquet'); d['region']=normalize_region_name(name); archived.append(d)
    except FileNotFoundError: pass

# A) Verify genuine archived forecasts against ex-post reanalysis truth.
if realized and archived:
    fc=pd.concat(archived,ignore_index=True); rr=pd.concat(realized,ignore_index=True)
    verified,skill=verify_archived_forecasts(fc,rr); skill.to_csv('outputs/results/weather_forecast_skill_by_lead.csv',index=False)
    revised=pd.concat([add_run_revisions(x) for x in archived],ignore_index=True)
    instability=revision_instability_skill(revised,rr,lead_days=1)
    results['weather_forecast_verification']={'status':'AVAILABLE','matched_rows':len(verified),'regions_forecast':len(archived),'regions_realized':len(realized),'skill':skill.to_dict('records'),'revision_instability':instability}
else:
    results['weather_forecast_verification']={'status':'NOT_EVALUABLE','regions_forecast':len(archived),'regions_realized':len(realized)}

# B) Physical channel: growing-season realized weather adds value beyond state + technology/time trend.
if realized:
    wr=[]
    for d in realized:
        z=d.rename(columns={'temperature_2m_max':'T2M_MAX','temperature_2m_min':'T2M_MIN','precipitation_sum':'PRECTOTCORR'}); wr.append(z)
    annual=aggregate_growing_season(pd.concat(wr,ignore_index=True))
    try:
        y=normalize_annual(read_table('data/raw/usda/corn_yield.parquet'),'yield_bu_acre').rename(columns={'state_name':'region'}); y['region']=y.region.map(normalize_region_name)
        panel=annual.merge(y,on=['region','year'],how='inner')
        pred,met,paired,hold=run_nested_yield_oos(panel,final_years=cfg['validation']['final_test_years'],min_train_years=cfg['validation']['min_train_years'],max_finalized_year=cfg['yield']['max_finalized_crop_year'])
        pred.to_csv('outputs/results/yield_nested_oos_predictions.csv',index=False); met.to_csv('outputs/results/yield_nested_oos_metrics.csv',index=False); paired.to_csv('outputs/results/yield_weather_incremental.csv',index=False)
        results['yield']={'status':'AVAILABLE','rows':int((panel.year <= cfg['yield']['max_finalized_crop_year']).sum()),'regions':int(panel.loc[panel.year <= cfg['yield']['max_finalized_crop_year'],'region'].nunique()),'years':int(panel.loc[panel.year <= cfg['yield']['max_finalized_crop_year'],'year'].nunique()),'holdout_years':hold,'max_finalized_crop_year':cfg['yield']['max_finalized_crop_year'],'metrics':met.to_dict('records'),'mean_abs_error_improvement_weather':float(paired.ae_improvement.mean()) if not paired.empty else None}
    except FileNotFoundError: results['yield']={'status':'NOT_EVALUABLE','reason':'USDA yield missing'}
else: results['yield']={'status':'NOT_EVALUABLE','reason':'No realized/reanalysis weather'}

# C) Genuine point-in-time forecast revision -> subsequent corn returns. Continuous proxy is explicitly labelled a proxy.
market=None; market_source=None
for mp in ['data/raw/market/corn_local.parquet','data/raw/market/corn_yahoo.parquet','data/raw/market/corn_yfinance.parquet']:
    try: market=forward_returns(read_table(mp),cfg['market']['horizons_days']); market_source=mp; break
    except FileNotFoundError: pass
production=None
try:
    production=normalize_annual(read_table('data/raw/usda/corn_production.parquet'),'production').rename(columns={'state_name':'region'}); production['region']=production.region.map(normalize_region_name)
except FileNotFoundError: pass
if archived and market is not None:
    fcasts=[]
    for d in archived: fcasts.append(add_run_revisions(d))
    met,pred=run_market_revision_association(pd.concat(fcasts,ignore_index=True),market,lead_days=1,horizon=1,production=production)
    if not pred.empty: pred.to_csv('outputs/results/market_revision_predictions.csv',index=False)
    met['market_source']=market_source; met['forecast_regions']=len(archived); met['market_data_warning']='continuous proxy; roll artifacts possible; not official CME settlement history'
    results['market_revision']=met
else: results['market_revision']={'status':'NOT_EVALUABLE','forecast_regions':len(archived),'market_available':market is not None}

Path('outputs/results/real_run_summary.json').write_text(json.dumps(results,indent=2,default=str)); print(json.dumps(results,indent=2,default=str))
