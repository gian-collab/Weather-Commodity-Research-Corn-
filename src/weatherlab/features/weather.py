from __future__ import annotations
import pandas as pd, numpy as np
from weatherlab.timekeys import daily_key

def add_weather_features(df, tmax='T2M_MAX', tmin='T2M_MIN', precip='PRECTOTCORR'):
    x=df.copy(); x['tmean']=(pd.to_numeric(x[tmax],errors='coerce')+pd.to_numeric(x[tmin],errors='coerce'))/2
    x['gdd10']=np.clip(x['tmean']-10,0,None); x['extreme_heat_35c']=(pd.to_numeric(x[tmax],errors='coerce')>=35).astype(float)
    x['dry_day']=(pd.to_numeric(x[precip],errors='coerce')<1).astype(float)
    return x

def aggregate_growing_season(df, date='date', region='region', start_month=4, end_month=9,
                             tmax='T2M_MAX', tmin='T2M_MIN', precip='PRECTOTCORR'):
    x=add_weather_features(df,tmax,tmin,precip); x[date]=daily_key(x[date]); x=x.dropna(subset=[date]); x=x[x[date].dt.month.between(start_month,end_month)].copy(); x['year']=x[date].dt.year
    ag=x.groupby([region,'year'],as_index=False).agg(
        tmean=('tmean','mean'),tmax_mean=(tmax,'mean'),precip_total=(precip,'sum'),
        gdd10=('gdd10','sum'),heat35_days=('extreme_heat_35c','sum'),dry_days=('dry_day','sum'))
    return ag

def production_weight(panel, weights, region='region', date='date'):
    z=panel.merge(weights,on=region,how='inner',validate='many_to_one')
    if (z['weight']<0).any(): raise ValueError('weights must be nonnegative')
    num=[c for c in z.select_dtypes('number') if c!='weight']
    def agg(g):
        w=g['weight'].to_numpy(float); s=w.sum()
        return pd.Series({c:np.average(g[c],weights=w) if s>0 else np.nan for c in num})
    return z.groupby(date).apply(agg,include_groups=False).reset_index()
