from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.base import clone
from weatherlab.validation.walk_forward import year_splits, final_holdout
from weatherlab.validation.metrics import point
from weatherlab.timekeys import normalize_region_name

WEATHER=['tmean','precip_total','gdd10','heat35_days','dry_days']

def _pipe(features):
    cat=['region']; num=[c for c in features if c!='region']
    pre=ColumnTransformer([('region',OneHotEncoder(handle_unknown='ignore'),cat),('num',StandardScaler(),num)])
    return Pipeline([('pre',pre),('ridge',Ridge(alpha=10))])

def run_nested_yield_oos(panel, final_years=2, min_train_years=8, max_finalized_year=2025):
    x=panel.copy(); x['region']=x['region'].map(normalize_region_name); x['year']=pd.to_numeric(x['year'],errors='coerce')
    x=x[x['year'] <= max_finalized_year].copy()
    # Absolute year is okay because each fold only fits coefficients on past years.
    specs={'trend_state':['region','year'],'trend_state_weather':['region','year',*WEATHER]}
    rows=[]
    for spec,features in specs.items():
        d=x.dropna(subset=['yield_bu_acre',*features]).sort_values(['year','region'])
        for trys,teys in year_splits(d.year,min_train_years=min_train_years,final_years=final_years):
            tr=d[d.year.isin(trys)]; te=d[d.year.isin(teys)]
            if tr.empty or te.empty: continue
            m=_pipe(features); m.fit(tr[features],tr.yield_bu_acre); p=m.predict(te[features])
            for idx,pr in zip(te.index,p): rows.append({'year':int(te.loc[idx,'year']),'region':te.loc[idx,'region'],'model':spec,'prediction':float(pr),'actual':float(te.loc[idx,'yield_bu_acre'])})
    pred=pd.DataFrame(rows); metrics=[]
    if not pred.empty:
        for name,g in pred.groupby('model'): metrics.append({'model':name,**point(g.actual,g.prediction)})
        # Paired incremental error improvement on identical state-years.
        a=pred[pred.model=='trend_state'][['year','region','prediction','actual']].rename(columns={'prediction':'p0'})
        b=pred[pred.model=='trend_state_weather'][['year','region','prediction']].rename(columns={'prediction':'p1'})
        q=a.merge(b,on=['year','region']); q['ae_baseline']=abs(q.actual-q.p0); q['ae_weather']=abs(q.actual-q.p1); q['ae_improvement']=q.ae_baseline-q.ae_weather
    else: q=pd.DataFrame()
    return pred,pd.DataFrame(metrics),q,final_holdout(x.year.dropna(),final_years)
