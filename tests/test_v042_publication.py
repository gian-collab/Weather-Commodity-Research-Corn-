from pathlib import Path
import numpy as np
import pandas as pd

from weatherlab.features.crop_stage import aggregate_stage_weather
from weatherlab.experiments.yield_nested import run_nested_yield_oos
from weatherlab.experiments.market_controls import run_incremental_weather_controls
from weatherlab.reporting.publication import build_publication_outputs


def test_crop_stage_tmax_uses_open_meteo_alias():
    d = pd.date_range('2020-06-20', periods=5, freq='D')
    x = pd.DataFrame({
        'date': d,
        'region': 'Iowa',
        'temperature_2m_max': [30,31,32,33,34],
        'temperature_2m_min': [18,19,20,21,22],
        'precipitation_sum': [0,2,0,5,1],
        'et0_fao_evapotranspiration': [4,4,4,4,4],
    })
    out = aggregate_stage_weather(x)
    assert np.isfinite(out.loc[0, 'silking_tmax_mean'])
    assert np.isfinite(out.loc[0, 'silking_precip_total'])


def test_baseline_yield_excludes_unfinalized_years():
    rows=[]
    for year in range(2010, 2027):
        for region in ['Iowa','Illinois']:
            rows.append({'year':year,'region':region,'yield_bu_acre':150+year-2010,'tmean':20,'precip_total':100,'gdd10':800,'heat35_days':1,'dry_days':5})
    pred, met, paired, hold = run_nested_yield_oos(pd.DataFrame(rows), final_years=2, min_train_years=5, max_finalized_year=2025)
    assert 2026 not in hold
    assert max(hold) <= 2025
    if not pred.empty:
        assert pred['year'].max() <= 2023


def test_incremental_market_controls_runs_on_synthetic_data():
    dates = pd.date_range('2024-04-01', periods=220, freq='D')
    market = pd.DataFrame({'date':dates,'close':100+np.cumsum(np.sin(np.arange(len(dates))/10)*0.1 + 0.02)})
    for h in [1,3,5,10]:
        market[f'ret_fwd_{h}d'] = np.log(market['close'].shift(-h)/market['close'])
    forecasts=[]
    for lead in [1,3,5]:
        for d in dates:
            forecasts.append({'decision_date':d,'valid_date':d+pd.Timedelta(days=lead),'lead_days':lead,'region':'Iowa','temperature_mean_revision':0.1,'temperature_max_revision':0.2,'precipitation_sum_revision':0.0})
    f=pd.DataFrame(forecasts)
    grid,pred,inf=run_incremental_weather_controls(f,market,production=None,leads=(1,),horizons=(1,3))
    assert not grid.empty
    assert set(grid['status']) == {'AVAILABLE'}
    assert 'incremental_oos_r2_vs_controls' in grid
    assert not pred.empty


def test_publication_excludes_zero_coverage_features(tmp_path: Path):
    results=tmp_path/'results'; pub=tmp_path/'pub'; results.mkdir()
    pd.DataFrame([
        {'feature':'silking_tmean','n_nonmissing':10,'coverage':1.0},
        {'feature':'silking_water_balance','n_nonmissing':0,'coverage':0.0},
    ]).to_csv(results/'stage_weather_feature_coverage.csv',index=False)
    manifest=build_publication_outputs(results,pub)
    observed=pd.read_csv(pub/'tables/stage_weather_observed_features.csv')
    excluded=pd.read_csv(pub/'tables/stage_weather_excluded_zero_coverage.csv')
    assert observed['feature'].tolist()==['silking_tmean']
    assert excluded['feature'].tolist()==['silking_water_balance']
    assert manifest['stage_features_zero_coverage_excluded']==1
