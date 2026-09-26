import pandas as pd
from weatherlab.experiments.weather_verification import verify_archived_forecasts, revision_instability_skill
from weatherlab.data.open_meteo import add_run_revisions


def test_weather_verification_by_lead():
    r=pd.DataFrame({'date':pd.date_range('2025-01-01',periods=4,tz='UTC'),'region':'Iowa','temperature_2m_max':[10,11,12,13],'temperature_2m_min':[0,1,2,3],'precipitation_sum':[1,0,2,1]})
    rows=[]
    for d in r.date:
        rows.append({'date':d,'decision_date':d-pd.Timedelta(days=1),'region':'Iowa','lead_days':1,'temperature_mean':5.5,'temperature_max':10.5,'precipitation_sum':1.})
    _,s=verify_archived_forecasts(pd.DataFrame(rows),r)
    assert set(s.variable)=={'temperature_mean','temperature_max','precipitation_sum'}
