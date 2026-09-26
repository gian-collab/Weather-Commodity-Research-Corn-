from pathlib import Path
import sys, json, subprocess
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))

# Preserve the v0.3.1 baseline research outputs first.
subprocess.run([sys.executable,'scripts/run_real.py'],check=True,cwd=ROOT)

import pandas as pd, yaml
from weatherlab.data.storage import read_table
from weatherlab.data.usda import normalize_annual
from weatherlab.data.open_meteo import add_run_revisions
from weatherlab.data.corn_market import forward_returns
from weatherlab.features.crop_stage import aggregate_stage_weather
from weatherlab.experiments.yield_mechanism import run_yield_mechanism_oos
from weatherlab.experiments.market_mechanism import run_market_mechanism_grid, run_revision_vol_grid
from weatherlab.experiments.market_controls import run_incremental_weather_controls
from weatherlab.reporting.publication import build_publication_outputs
from weatherlab.experiments.ensemble_archive import inventory_ensemble_snapshots
from weatherlab.experiments.inference import yield_paired_inference, market_grid_inference
from weatherlab.timekeys import normalize_region_name

cfg=yaml.safe_load((ROOT/'configs/full.yaml').read_text()); regions=cfg['corn_belt']['regions']; out=ROOT/'outputs/results'; out.mkdir(parents=True,exist_ok=True)
realized=[]; archived=[]
for name in regions:
    slug=name.lower().replace(' ','_')
    try:
        d=read_table(ROOT/f'data/raw/open_meteo/reanalysis/{slug}.parquet'); d['region']=normalize_region_name(name); realized.append(d)
    except FileNotFoundError: pass
    try:
        d=read_table(ROOT/f'data/raw/open_meteo/previous_runs/{slug}.parquet'); d['region']=normalize_region_name(name); archived.append(d)
    except FileNotFoundError: pass

summary={'version':'0.4.2','baseline_summary':'outputs/results/real_run_summary.json'}
# Stage/extreme yield models
if realized:
    rr=[]
    for d in realized:
        rr.append(d.copy())
    stage=aggregate_stage_weather(pd.concat(rr,ignore_index=True)); stage.to_csv(out/'stage_weather_features.csv',index=False)
    # Explicit coverage ledger prevents unavailable weather variables from being
    # silently treated as observed.
    feature_cols=[c for c in stage.columns if c not in ('region','year')]
    cov=pd.DataFrame([{'feature':c,'n_nonmissing':int(stage[c].notna().sum()),'coverage':float(stage[c].notna().mean())} for c in feature_cols])
    cov.to_csv(out/'stage_weather_feature_coverage.csv',index=False)
    try:
        y=normalize_annual(read_table(ROOT/'data/raw/usda/corn_yield.parquet'),'yield_bu_acre').rename(columns={'state_name':'region'}); y['region']=y.region.map(normalize_region_name)
        panel=stage.merge(y,on=['region','year'],how='inner')
        pred,met,paired,hold=run_yield_mechanism_oos(panel,final_years=cfg['validation']['final_test_years'],min_train_years=max(10,cfg['validation']['min_train_years']),max_finalized_year=cfg['yield']['max_finalized_crop_year'])
        pred.to_csv(out/'yield_mechanism_predictions.csv',index=False); met.to_csv(out/'yield_mechanism_metrics.csv',index=False); paired.to_csv(out/'yield_mechanism_paired.csv',index=False)
        yinf=yield_paired_inference(paired); yinf.to_csv(out/'yield_mechanism_inference.csv',index=False)
        summary['yield_mechanism']={'status':'AVAILABLE','rows':len(panel),'holdout_years':hold,'best_model':met.iloc[0].to_dict() if not met.empty else None}
    except FileNotFoundError: summary['yield_mechanism']={'status':'NOT_EVALUABLE'}

# Market mechanism grid
market=None; market_source=None
for mp in [ROOT/'data/raw/market/corn_local.parquet',ROOT/'data/raw/market/corn_yahoo.parquet',ROOT/'data/raw/market/corn_yfinance.parquet']:
    try: market=forward_returns(read_table(mp),cfg['market']['horizons_days']); market_source=str(mp); break
    except FileNotFoundError: pass
production=None
try:
    production=normalize_annual(read_table(ROOT/'data/raw/usda/corn_production.parquet'),'production').rename(columns={'state_name':'region'}); production['region']=production.region.map(normalize_region_name)
except FileNotFoundError: pass
if archived and market is not None:
    f=pd.concat([add_run_revisions(d) for d in archived],ignore_index=True)
    grid,pred=run_market_mechanism_grid(f,market,production=production,leads=tuple(cfg['market_mechanism']['lead_days']),horizons=tuple(cfg['market_mechanism']['return_horizons']))
    grid.to_csv(out/'market_mechanism_grid.csv',index=False); pred.to_csv(out/'market_mechanism_predictions.csv',index=False)
    minf=market_grid_inference(pred); minf.to_csv(out/'market_mechanism_inference.csv',index=False)
    vol=run_revision_vol_grid(f,market,production=production,leads=tuple(cfg['market_mechanism']['lead_days']),horizons=tuple(cfg['market_mechanism']['vol_horizons']))
    vol.to_csv(out/'market_revision_vol_grid.csv',index=False)
    summary['market_mechanism']={'status':'AVAILABLE','market_source':market_source,'tests':len(grid),'best_oos_r2':float(grid.loc[grid.get('status','AVAILABLE').eq('AVAILABLE'),'oos_r2_vs_zero'].max()) if (not grid.empty and 'status' in grid.columns and grid.status.eq('AVAILABLE').any()) else (float(grid.oos_r2_vs_zero.max()) if not grid.empty else None),'vol_tests':len(vol)}
    cgrid,cpred,cinf=run_incremental_weather_controls(f,market,production=production,leads=tuple(cfg['market_mechanism']['lead_days']),horizons=tuple(cfg['market_mechanism']['return_horizons']))
    cgrid.to_csv(out/'market_controls_grid.csv',index=False); cpred.to_csv(out/'market_controls_predictions.csv',index=False); cinf.to_csv(out/'market_controls_inference.csv',index=False)
    summary['market_controls']={'status':'AVAILABLE' if not cgrid.empty else 'NOT_EVALUABLE','tests':len(cgrid),'best_incremental_oos_r2_vs_controls':float(cgrid.loc[cgrid.status.eq('AVAILABLE'),'incremental_oos_r2_vs_controls'].max()) if (not cgrid.empty and 'status' in cgrid.columns and cgrid.status.eq('AVAILABLE').any()) else None}
else: summary['market_mechanism']={'status':'NOT_EVALUABLE'}
summary['ensemble_archive']=inventory_ensemble_snapshots(ROOT/'data/snapshots/ecmwf')

# Conservative hypothesis ledger for article drafting.
hyp=[]
try:
    yinf=pd.read_csv(out/'yield_mechanism_inference.csv')
    if not yinf.empty:
        r=yinf.iloc[0]; status='SUPPORTED_RESEARCH' if r.ci_low>0 else ('REJECTED' if r.ci_high<0 else 'INCONCLUSIVE')
        hyp.append({'hypothesis_id':'H3_stage_weather_yield','status':status,'evidence':f"best={r['model']}; year-clustered AE improvement={r['year_clustered_estimate']:.3f}; 95% CI=[{r['ci_low']:.3f},{r['ci_high']:.3f}]"})
except Exception: pass
try:
    minf=pd.read_csv(out/'market_mechanism_inference.csv')
    if not minf.empty:
        r=minf.iloc[0]; status='SUPPORTED_RESEARCH' if r.ci_low>0 else ('REJECTED' if r.ci_high<0 else 'INCONCLUSIVE')
        hyp.append({'hypothesis_id':'H5_weather_revision_market','status':status,'evidence':f"best lead={int(r.lead_days)}d horizon={int(r.horizon_days)}d spec={r.spec}; MSE improvement={r.mse_improvement:.8f}; 95% CI=[{r.ci_low:.8f},{r.ci_high:.8f}]"})
except Exception: pass
pd.DataFrame(hyp).to_csv(out/'hypothesis_results_v0_4.csv',index=False)
summary['hypotheses']=hyp
(out/'v0_4_summary.json').write_text(json.dumps(summary,indent=2,default=str))
pub_manifest=build_publication_outputs(out, ROOT/'outputs/publication')
summary['publication_outputs']='outputs/publication'
(out/'v0_4_summary.json').write_text(json.dumps(summary,indent=2,default=str)); print(json.dumps(summary,indent=2,default=str))
