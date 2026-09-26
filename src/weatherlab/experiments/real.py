from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from weatherlab.models.yield_model import models as yield_models
from weatherlab.validation.metrics import point
from weatherlab.validation.walk_forward import year_splits, final_holdout
from weatherlab.timekeys import daily_key, normalize_region_name


def run_yield_oos(panel: pd.DataFrame, target='yield_bu_acre', features=None, final_years=2, min_train_years=8, seed=42):
    features=features or ['tmean','precip_total','gdd10','heat35_days','dry_days']
    x=panel.copy()
    x['region']=x['region'].map(normalize_region_name)
    x=x.dropna(subset=[target,*features,'year']).sort_values(['year','region']).copy()
    rows=[]
    for model_name,base in yield_models(seed).items():
        for tr_years,te_years in year_splits(x.year,min_train_years=min_train_years,final_years=final_years):
            tr=x[x.year.isin(tr_years)]; te=x[x.year.isin(te_years)]
            if tr.empty or te.empty: continue
            m=clone(base); m.fit(tr[features],tr[target]); pred=m.predict(te[features])
            for idx,p in zip(te.index,pred):
                rows.append({'index':int(idx),'year':int(x.loc[idx,'year']),'region':x.loc[idx,'region'],'model':model_name,'prediction':float(p),'actual':float(x.loc[idx,target]),'split':'research_oos'})
    out=pd.DataFrame(rows)
    metrics=[]
    if not out.empty:
        for name,g in out.groupby('model'):
            metrics.append({'model':name,**point(g.actual,g.prediction)})
    hold=final_holdout(x.year,final_years)
    return out,pd.DataFrame(metrics),hold


def _prepare_production_weights(production: pd.DataFrame | None, decision_dates: pd.Series) -> pd.DataFrame | None:
    """Create lagged state production weights without using same-year future production.

    Forecasts in calendar year Y use USDA production from Y-1. This is intentionally
    conservative and point-in-time safer than contemporaneous final production.
    """
    if production is None or production.empty: return None
    p=production.copy()
    p['region']=p['region'].map(normalize_region_name)
    p['year']=pd.to_numeric(p['year'], errors='coerce')
    p['production']=pd.to_numeric(p['production'], errors='coerce')
    p=p.dropna(subset=['year','production']); p=p[p.production>0]
    p['decision_year']=p['year'].astype(int)+1
    totals=p.groupby('decision_year')['production'].transform('sum')
    p['weight']=p['production']/totals
    return p[['decision_year','region','weight']]


def aggregate_forecast_revisions(forecasts: pd.DataFrame, production: pd.DataFrame | None=None, lead_days=1):
    f=forecasts[forecasts.lead_days==lead_days].copy()
    if f.empty: return f
    f['region']=f.get('region','Unknown').map(normalize_region_name) if 'region' in f else 'Unknown'
    f['decision_date']=daily_key(f['decision_date'])
    f=f.dropna(subset=['decision_date'])
    f['decision_year']=f['decision_date'].dt.year
    rev_cols=['temperature_mean_revision','temperature_max_revision','precipitation_sum_revision']
    for c in rev_cols: f[c]=pd.to_numeric(f.get(c), errors='coerce')
    weights=_prepare_production_weights(production, f['decision_date'])
    if weights is not None:
        f=f.merge(weights,on=['decision_year','region'],how='left')
    else:
        f['weight']=np.nan
    rows=[]
    for d,g in f.groupby('decision_date',sort=True):
        available_regions=int(g['region'].nunique())
        weighted=g['weight'].notna() & (g['weight']>0)
        vals={'decision_date':d,'regions_available':available_regions}
        if weighted.any():
            gg=g.loc[weighted].copy(); w=gg['weight'].to_numpy(float); w=w/w.sum()
            vals['aggregation']='lagged_production_weighted'
            vals['weight_coverage']=float(g.loc[weighted,'weight'].sum())
            for c in rev_cols:
                v=gg[c].to_numpy(float); m=np.isfinite(v)
                vals[c]=float(np.average(v[m],weights=w[m]/w[m].sum())) if m.any() else np.nan
        else:
            vals['aggregation']='equal_weight'
            vals['weight_coverage']=np.nan
            for c in rev_cols: vals[c]=float(g[c].mean())
        rows.append(vals)
    return pd.DataFrame(rows)


def run_market_revision_association(forecasts: pd.DataFrame, market: pd.DataFrame, lead_days=1, horizon=1, production: pd.DataFrame | None=None):
    f=aggregate_forecast_revisions(forecasts,production=production,lead_days=lead_days)
    target=f'ret_fwd_{horizon}d'
    m=market.copy()
    m['decision_date']=daily_key(m['date'])
    m[target]=pd.to_numeric(m.get(target),errors='coerce')
    z=f.merge(m[['decision_date',target]],on='decision_date',how='inner').sort_values('decision_date')
    feature_cols=['temperature_mean_revision','temperature_max_revision','precipitation_sum_revision']
    z=z.dropna(subset=[target]).copy()
    # Require at least one revision feature and impute remaining feature gaps from train medians only.
    z=z[z[feature_cols].notna().any(axis=1)]
    if len(z)<30:
        return {'status':'INSUFFICIENT_DATA','n':len(z),'lead_days':lead_days,'horizon':horizon},pd.DataFrame()
    cut=max(20,int(len(z)*.7))
    tr=z.iloc[:cut].copy(); te=z.iloc[cut:].copy()
    med=tr[feature_cols].median(numeric_only=True)
    Xtr=tr[feature_cols].fillna(med); Xte=te[feature_cols].fillna(med)
    good=[c for c in feature_cols if np.isfinite(Xtr[c]).all() and Xtr[c].nunique(dropna=True)>1]
    if not good:
        return {'status':'INSUFFICIENT_FEATURE_VARIATION','n':len(z),'lead_days':lead_days,'horizon':horizon},pd.DataFrame()
    ytr=tr[target]; yte=te[target]
    model=Pipeline([('scale',StandardScaler()),('ridge',Ridge(alpha=10))]); model.fit(Xtr[good],ytr); p=model.predict(Xte[good])
    met=point(yte,p)
    # Baseline is zero return forecast; report incremental OOS R2 explicitly.
    sse=float(np.sum((yte.to_numpy(float)-p)**2)); sse0=float(np.sum(yte.to_numpy(float)**2))
    met.update({'status':'AVAILABLE','n_train':len(tr),'n_test':len(te),'lead_days':lead_days,'horizon':horizon,
                'oos_r2_vs_zero':float(1-sse/sse0) if sse0>0 else np.nan,
                'features':good,'regions_median':float(te.regions_available.median()) if 'regions_available' in te else np.nan,
                'aggregation':te['aggregation'].mode().iat[0] if 'aggregation' in te and not te['aggregation'].mode().empty else 'unknown'})
    pred=te[['decision_date',target,'regions_available','aggregation','weight_coverage']].copy()
    pred['prediction']=p
    return met,pred
