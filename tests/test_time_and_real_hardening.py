import numpy as np, pandas as pd
from weatherlab.timekeys import daily_key, normalize_region_name
from weatherlab.experiments.real import run_market_revision_association, aggregate_forecast_revisions
from weatherlab.data.open_meteo import add_run_revisions
from weatherlab.data.corn_market import forward_returns


def test_daily_key_aligns_string_and_tz_aware():
    a=daily_key(pd.Series(['2025-01-02']))
    b=daily_key(pd.Series(pd.to_datetime(['2025-01-02 18:00'],utc=True)))
    assert a.dtype==b.dtype and a.iloc[0]==b.iloc[0]


def test_region_normalization_handles_underscores():
    assert normalize_region_name('South_Dakota')=='South Dakota'


def test_market_revision_merge_handles_mixed_date_types_and_weights():
    rows=[]
    for valid in pd.date_range('2025-01-03',periods=80,freq='D',tz='UTC'):
        for region,base in [('Iowa',10.),('Illinois',12.)]:
            rows += [
                {'date':valid,'decision_date':(valid-pd.Timedelta(days=1)).strftime('%Y-%m-%d'),'lead_days':1,'temperature_mean':base+1+0.05*valid.dayofyear,'temperature_max':base+3+0.03*valid.dayofyear,'precipitation_sum':2.+0.01*valid.dayofyear,'region':region},
                {'date':valid,'decision_date':(valid-pd.Timedelta(days=2)).strftime('%Y-%m-%d'),'lead_days':2,'temperature_mean':base+0.02*valid.dayofyear,'temperature_max':base+2+0.01*valid.dayofyear,'precipitation_sum':3.-0.005*valid.dayofyear,'region':region},
            ]
    fc=add_run_revisions(pd.DataFrame(rows))
    dates=pd.date_range('2025-01-01',periods=100,freq='D',tz='UTC')
    m=forward_returns(pd.DataFrame({'date':dates,'close':100+np.arange(100)*.1}),[1])
    prod=pd.DataFrame({'year':[2024,2024],'region':['Iowa','Illinois'],'production':[100,300]})
    met,pred=run_market_revision_association(fc,m,1,1,prod)
    assert met['status']=='AVAILABLE' and met['n_test']>0 and not pred.empty
    assert met['aggregation']=='lagged_production_weighted'


def test_aggregate_forecast_reports_region_coverage():
    f=pd.DataFrame({'decision_date':['2025-01-01','2025-01-01'],'lead_days':[1,1],'region':['Iowa','Illinois'],
                    'temperature_mean_revision':[1.,2.],'temperature_max_revision':[1.,2.],'precipitation_sum_revision':[0.,1.]})
    z=aggregate_forecast_revisions(f,None,1)
    assert z.iloc[0].regions_available==2 and z.iloc[0].aggregation=='equal_weight'


def test_daily_key_handles_mixed_iso_and_plain_dates():
    x=daily_key(pd.Series(['1980-01-01 00:00:00+00:00','1981-01-01','1982-01-01T12:30:00Z']))
    assert x.notna().all()
    assert [v.strftime('%Y-%m-%d') for v in x]==['1980-01-01','1981-01-01','1982-01-01']
