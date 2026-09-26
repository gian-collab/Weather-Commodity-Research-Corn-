import numpy as np
import pandas as pd
from weatherlab.features.crop_stage import aggregate_stage_weather
from weatherlab.experiments.yield_mechanism import run_yield_mechanism_oos
from weatherlab.experiments.market_mechanism import run_market_mechanism_grid
from weatherlab.data.open_meteo import add_run_revisions


def test_stage_features_capture_extremes_and_streaks():
    d=pd.date_range('2020-06-20','2020-07-31',freq='D')
    x=pd.DataFrame({'date':d,'region':'Iowa','T2M_MAX':36.0,'T2M_MIN':20.0,'PRECTOTCORR':0.0,'et0_fao_evapotranspiration':4.0})
    a=aggregate_stage_weather(x)
    assert a.loc[0,'silking_heat35_days'] > 30
    assert a.loc[0,'silking_max_dry_streak'] > 30
    assert a.loc[0,'silking_water_balance'] < 0


def test_yield_excludes_unfinalized_2026():
    rows=[]
    for y in range(2000,2027):
        for r in ['Iowa','Illinois']:
            rows.append({'year':y,'region':r,'yield_bu_acre':100+2*(y-2000),
                         'silking_tmean':25,'silking_precip_total':100,'silking_gdd10':200,
                         'silking_heat30_days':5,'silking_heat32_days':2,'silking_heat35_days':1,
                         'silking_dry_days':10,'silking_heavy_rain_days':1,'silking_hot_dry_days':1,
                         'silking_water_balance':30,'silking_max_hot_streak32':2,'silking_max_dry_streak':4})
    p=pd.DataFrame(rows)
    pred,met,paired,hold=run_yield_mechanism_oos(p,final_years=2,min_train_years=8,max_finalized_year=2025)
    assert 2026 not in pred.year.unique()
    assert 2026 not in hold


def test_market_grid_is_growing_season_and_multihorizon():
    dates=pd.date_range('2024-04-01','2025-09-30',freq='D')
    base=[]
    for region in ['Iowa','Illinois']:
        for d in dates:
            for lead in [1,2,3,4,5,6,7]:
                base.append({'date':d,'decision_date':d-pd.Timedelta(days=lead),'lead_days':lead,'region':region,
                             'temperature_mean':20+0.01*lead,'temperature_max':30+0.02*lead,'precipitation_sum':2+0.01*lead})
    f=add_run_revisions(pd.DataFrame(base))
    m=pd.DataFrame({'date':dates,'close':500+np.arange(len(dates))*0.05})
    for h in [1,3,5,10]: m[f'ret_fwd_{h}d']=0.001
    g,p=run_market_mechanism_grid(f,m,production=None,leads=(1,3),horizons=(1,3))
    assert set(g.horizon_days.unique()) == {1,3}
    assert set(g.lead_days.unique()) == {1,3}


def test_market_grid_skips_structurally_missing_revision_lead():
    dates=pd.date_range('2024-04-01','2025-09-30',freq='D')
    rows=[]
    # Only leads 1..7 => lead-7 revision needs lead 8 and is structurally missing.
    for region in ['Iowa','Illinois']:
        for d in dates:
            for lead in range(1,8):
                rows.append({'date':d,'decision_date':d-pd.Timedelta(days=lead),'lead_days':lead,'region':region,
                             'temperature_mean':20+lead*.1,'temperature_max':30+lead*.1,'precipitation_sum':2+lead*.05})
    f=add_run_revisions(pd.DataFrame(rows))
    m=pd.DataFrame({'date':dates,'close':500+np.arange(len(dates))*.05,'ret_fwd_1d':0.001})
    g,p=run_market_mechanism_grid(f,m,production=None,leads=(7,),horizons=(1,))
    assert not g.empty
    assert set(g.status)=={'NOT_EVALUABLE_NO_TRAIN_FEATURES'}
    assert p.empty


def test_stage_features_resolve_open_meteo_aliases():
    d=pd.date_range('2020-06-20','2020-07-31',freq='D')
    x=pd.DataFrame({'date':d,'region':'Iowa','temperature_2m_max':36.0,'temperature_2m_min':20.0,
                    'precipitation_sum':2.0,'et0_fao_evapotranspiration':4.0})
    a=aggregate_stage_weather(x)
    assert a.loc[0,'silking_precip_total'] > 0
    assert a.loc[0,'silking_water_balance'] < 0
