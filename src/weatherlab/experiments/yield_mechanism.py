from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from weatherlab.validation.walk_forward import year_splits, final_holdout
from weatherlab.validation.metrics import point
from weatherlab.features.crop_stage import mechanism_feature_groups
from weatherlab.timekeys import normalize_region_name


def _linear(features):
    cat = ['region']; num = [c for c in features if c != 'region']
    pre = ColumnTransformer([
        ('region', OneHotEncoder(handle_unknown='ignore'), cat),
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),('scale', StandardScaler())]), num)
    ])
    return Pipeline([('pre', pre), ('ridge', Ridge(alpha=20.0))])


def _tree(features):
    # Tree model uses numeric weather + year; state effects are represented via one-hot outside model.
    cat = ['region']; num = [c for c in features if c != 'region']
    pre = ColumnTransformer([
        ('region', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat),
        ('num', SimpleImputer(strategy='median'), num)
    ])
    return Pipeline([('pre', pre), ('hgb', HistGradientBoostingRegressor(max_depth=3, learning_rate=.04, max_iter=250, l2_regularization=2.0, random_state=42))])


def run_yield_mechanism_oos(panel: pd.DataFrame, final_years=2, min_train_years=10, max_finalized_year=2025):
    x = panel.copy()
    x['region'] = x['region'].map(normalize_region_name)
    x['year'] = pd.to_numeric(x['year'], errors='coerce')
    x['yield_bu_acre'] = pd.to_numeric(x['yield_bu_acre'], errors='coerce')
    x = x[x['year'] <= max_finalized_year].dropna(subset=['region','year','yield_bu_acre']).copy()
    groups = mechanism_feature_groups(x.columns)
    specs = {
        'trend_state': ['region','year'],
        'trend_state_means': ['region','year',*groups['means']],
        'trend_state_extremes': ['region','year',*groups['means'],*groups['extremes']],
        'trend_state_all_weather': ['region','year',*groups['all_weather']],
    }
    rows=[]
    for spec, feats in specs.items():
        feats=[f for f in feats if f in x.columns]
        for trys, teys in year_splits(x.year, min_train_years=min_train_years, final_years=final_years):
            tr=x[x.year.isin(trys)]; te=x[x.year.isin(teys)]
            if tr.empty or te.empty: continue
            # Feature availability is determined from TRAINING DATA ONLY. Completely
            # missing weather fields are excluded rather than silently dropped by an
            # imputer, preserving stable train/test dimensionality and clear auditability.
            active=['region','year']
            for f in feats:
                if f in ('region','year') or f not in tr.columns: continue
                v=pd.to_numeric(tr[f],errors='coerce').replace([np.inf,-np.inf],np.nan)
                if v.notna().any(): active.append(f)
            active=list(dict.fromkeys(active))
            for family, builder in [('ridge',_linear),('hgb',_tree)]:
                # Keep HGB only for weather specs; baseline HGB adds no interpretive value.
                if family=='hgb' and spec=='trend_state': continue
                # If a weather spec has no observed weather variables in this training
                # fold, it is not a distinct model and must not masquerade as one.
                if spec!='trend_state' and len([f for f in active if f not in ('region','year')])==0:
                    continue
                m=builder(active); m.fit(tr[active],tr.yield_bu_acre); p=m.predict(te[active])
                for idx,pr in zip(te.index,p):
                    rows.append({'year':int(te.loc[idx,'year']),'region':te.loc[idx,'region'],'spec':spec,'family':family,'model':f'{spec}_{family}','prediction':float(pr),'actual':float(te.loc[idx,'yield_bu_acre']),'n_active_features':len(active)-2})
    pred=pd.DataFrame(rows)
    metrics=[]
    if not pred.empty:
        for name,g in pred.groupby('model'):
            metrics.append({'model':name,**point(g.actual,g.prediction)})
    metrics=pd.DataFrame(metrics).sort_values('rmse') if metrics else pd.DataFrame()
    # Paired improvement against linear state+trend baseline on identical state-years.
    paired=[]
    if not pred.empty:
        b=pred[pred.model=='trend_state_ridge'][['year','region','actual','prediction']].rename(columns={'prediction':'baseline_prediction'})
        for name,g in pred[pred.model!='trend_state_ridge'].groupby('model'):
            q=b.merge(g[['year','region','prediction']],on=['year','region'],how='inner')
            q['baseline_ae']=(q.actual-q.baseline_prediction).abs(); q['model_ae']=(q.actual-q.prediction).abs()
            q['ae_improvement']=q.baseline_ae-q.model_ae; q['model']=name
            paired.append(q)
    paired=pd.concat(paired,ignore_index=True) if paired else pd.DataFrame()
    hold=final_holdout(x.year.dropna(),final_years)
    return pred,metrics,paired,hold
