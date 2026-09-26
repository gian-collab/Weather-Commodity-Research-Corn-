import numpy as np, pandas as pd
from weatherlab.data.open_meteo import add_run_revisions
from weatherlab.data.usda import normalize_annual,parse_numeric_value
from weatherlab.features.weather import aggregate_growing_season
from weatherlab.data.corn_market import forward_returns
from weatherlab.validation.walk_forward import year_splits,final_holdout
from weatherlab.experiments.real import run_yield_oos

def test_usda_numeric_and_state_normalization():
    d=pd.DataFrame({'year':['2020','2021'],'state_name':['IOWA','ILLINOIS'],'Value':['178.0',' 201 ']})
    x=normalize_annual(d,'yield_bu_acre'); assert list(x.state_name)==['Iowa','Illinois']; assert x.yield_bu_acre.sum()==379

def test_open_meteo_revision_direction_is_new_minus_old():
    d=pd.DataFrame({'date':pd.to_datetime(['2024-01-10','2024-01-10'],utc=True),'lead_days':[1,2],'temperature_mean':[10.,8.],'temperature_max':[12.,10.],'precipitation_sum':[3.,5.]})
    z=add_run_revisions(d); r=z[z.lead_days==1].iloc[0]; assert r.temperature_mean_revision==2; assert r.precipitation_sum_revision==-2

def test_growing_season_aggregation():
    dates=pd.date_range('2020-04-01','2020-04-10'); d=pd.DataFrame({'date':dates,'region':'Iowa','T2M_MAX':30.,'T2M_MIN':20.,'PRECTOTCORR':2.})
    a=aggregate_growing_season(d); assert len(a)==1; assert a.iloc[0].precip_total==20; assert a.iloc[0].gdd10>0

def test_forward_returns_are_forward_not_backward():
    d=pd.DataFrame({'date':pd.date_range('2020-01-01',periods=4,tz='UTC'),'close':[100,101,103,106]}); x=forward_returns(d,[1]); assert abs(x.iloc[0].ret_fwd_1d-.01)<1e-12

def test_year_split_preserves_final_holdout():
    ys=list(range(2000,2015)); sp=list(year_splits(ys,min_train_years=5,final_years=2)); assert sp; assert all(2013 not in te and 2014 not in te for _,te in sp); assert final_holdout(ys,2)==[2013,2014]

def test_yield_oos_runs_on_panel():
    rng=np.random.default_rng(2); rows=[]
    for y in range(2000,2020):
      for r in ['Iowa','Illinois','Nebraska']:
        t=20+rng.normal(); p=500+rng.normal(scale=50); rows.append({'year':y,'region':r,'tmean':t,'precip_total':p,'gdd10':1000+rng.normal(scale=30),'heat35_days':5+rng.normal(),'dry_days':20+rng.normal(scale=2),'yield_bu_acre':150+0.05*p-1.2*(t-20)+rng.normal(scale=5)})
    pred,met,hold=run_yield_oos(pd.DataFrame(rows),final_years=2,min_train_years=8); assert not pred.empty; assert set(met.model)=={'ridge','boosting'}; assert hold==[2018,2019]

def test_market_csv_normalization(tmp_path):
    from weatherlab.data.corn_market import ingest_market_csv
    p=tmp_path/'corn.csv'
    pd.DataFrame({'Date':['2024-01-01','2024-01-02'],'Settle':[450.0,455.0],'Volume':[100,120]}).to_csv(p,index=False)
    out=ingest_market_csv(p,tmp_path/'out.csv')
    assert out['status']=='AVAILABLE' and out['rows']==2

def test_nass_queries_are_minimal_and_valid_shape():
    from weatherlab.data.usda import corn_yield_query,corn_production_query
    y=corn_yield_query(1980); p=corn_production_query(1980)
    assert y['commodity_desc']=='CORN' and y['statisticcat_desc']=='YIELD' and y['year__GE']=='1980'
    assert 'unit_desc' not in y and 'class_desc' not in y
    assert p['statisticcat_desc']=='PRODUCTION'


def test_growing_season_handles_mixed_cached_date_formats():
    d=pd.DataFrame({
        'date':['2020-04-01 00:00:00+00:00','2020-04-02','2020-04-03T00:00:00Z'],
        'region':['Iowa']*3,'T2M_MAX':[30.,31.,32.],'T2M_MIN':[20.,21.,22.],'PRECTOTCORR':[1.,2.,3.]})
    a=aggregate_growing_season(d)
    assert len(a)==1 and a.iloc[0].precip_total==6
