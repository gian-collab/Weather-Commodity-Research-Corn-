from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from weatherlab.experiments.real import aggregate_forecast_revisions
from weatherlab.timekeys import daily_key


def _add_revision_mechanism_features(f: pd.DataFrame) -> pd.DataFrame:
    x=f.copy()
    # Revisions in economically relevant threshold exceedance proxies.
    if {'temperature_max','temperature_max_older'}.issubset(x.columns):
        mx=pd.to_numeric(x.temperature_max,errors='coerce'); old=pd.to_numeric(x.temperature_max_older,errors='coerce')
        for th in (30,32,35):
            valid=mx.notna() & old.notna()
            cur=(mx>=th).astype(float); prev=(old>=th).astype(float)
            x[f'heat{th}_event_revision']=(cur-prev).where(valid, np.nan)
    if {'precipitation_sum','precipitation_sum_older'}.issubset(x.columns):
        pr=pd.to_numeric(x.precipitation_sum,errors='coerce'); old=pd.to_numeric(x.precipitation_sum_older,errors='coerce')
        valid=pr.notna() & old.notna()
        x['dry_event_revision']=((pr<1).astype(float)-(old<1).astype(float)).where(valid, np.nan)
        x['heavy_rain_event_revision']=((pr>=25).astype(float)-(old>=25).astype(float)).where(valid, np.nan)
    return x


def _aggregate(forecasts, production, lead_days):
    f=_add_revision_mechanism_features(forecasts[forecasts.lead_days==lead_days].copy())
    base=aggregate_forecast_revisions(f,production=production,lead_days=lead_days)
    extra=[c for c in f.columns if c.endswith('_event_revision')]
    if not extra: return base
    f['decision_date']=daily_key(f['decision_date']); f=f.dropna(subset=['decision_date'])
    e=f.groupby('decision_date',as_index=False)[extra].mean(numeric_only=True)
    return base.merge(e,on='decision_date',how='left')



def _active_numeric_features(train: pd.DataFrame, candidates):
    """Select usable numeric features using training data only."""
    active=[]
    for c in candidates:
        if c not in train.columns:
            continue
        v=pd.to_numeric(train[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
        if v.notna().any():
            active.append(c)
    return active

def _supply_state(production: pd.DataFrame | None):
    if production is None or production.empty: return None
    p=production.copy(); p['year']=pd.to_numeric(p.year,errors='coerce'); p['production']=pd.to_numeric(p.production,errors='coerce')
    annual=p.groupby('year',as_index=False).production.sum().sort_values('year')
    annual['prod_trend']=annual.production.rolling(5,min_periods=3).mean().shift(1)
    annual['tight_supply_proxy']=-(annual.production.shift(1)-annual.prod_trend)/annual.prod_trend
    annual['decision_year']=annual.year.astype('Int64')
    return annual[['decision_year','tight_supply_proxy']]


def run_market_mechanism_grid(forecasts: pd.DataFrame, market: pd.DataFrame, production=None,
                              leads=(1,3,5,7), horizons=(1,3,5,10), growing_months=(4,5,6,7,8,9)):
    m=market.copy(); m['decision_date']=daily_key(m['date']); m['month']=m.decision_date.dt.month
    supply=_supply_state(production)
    rows=[]; preds=[]
    for lead in leads:
        a=_aggregate(forecasts,production,lead)
        if a.empty: continue
        a['decision_year']=a.decision_date.dt.year
        if supply is not None: a=a.merge(supply,on='decision_year',how='left')
        base_feats=['temperature_mean_revision','temperature_max_revision','precipitation_sum_revision']
        mech_feats=base_feats+[c for c in a.columns if c.endswith('_event_revision')]
        for horizon in horizons:
            target=f'ret_fwd_{horizon}d'
            if target not in m.columns: continue
            z=a.merge(m[['decision_date','month',target]],on='decision_date',how='inner').sort_values('decision_date')
            z=z[z.month.isin(growing_months)].dropna(subset=[target]).copy()
            # Coarse ex-ante crop vulnerability weighting: low in planting/harvest, highest around silking/grain fill.
            vuln={4:0.20,5:0.35,6:0.65,7:1.00,8:0.80,9:0.30}
            z['crop_vulnerability']=z['month'].map(vuln).astype(float)
            if len(z)<80: continue
            for spec, feats in [('revision_linear',base_feats),('revision_extremes',mech_feats),('revision_stage_weighted',mech_feats)]:
                feats=[c for c in feats if c in z.columns]
                if spec=='revision_stage_weighted':
                    for c in list(feats):
                        z[f'{c}_x_stage']=pd.to_numeric(z[c],errors='coerce')*z['crop_vulnerability']
                    feats=feats+[f'{c}_x_stage' for c in feats]
                if 'tight_supply_proxy' in z.columns:
                    for c in list(feats): z[f'{c}_x_tight']=pd.to_numeric(z[c],errors='coerce')*pd.to_numeric(z.tight_supply_proxy,errors='coerce')
                    feats=feats+[f'{c}_x_tight' for c in feats]
                cut=max(50,int(len(z)*.7)); tr=z.iloc[:cut]; te=z.iloc[cut:]
                Xtr=tr[feats].replace([np.inf,-np.inf],np.nan); Xte=te[feats].replace([np.inf,-np.inf],np.nan)
                active=_active_numeric_features(Xtr,feats)
                if not active:
                    rows.append({'lead_days':lead,'horizon_days':horizon,'spec':spec,'n_train':len(tr),'n_test':len(te),'status':'NOT_EVALUABLE_NO_TRAIN_FEATURES','n_active_features':0,'oos_r2_vs_zero':np.nan,'corr':np.nan,'mean_prediction':np.nan,'mean_actual':float(te[target].mean())})
                    continue
                pipe=Pipeline([('imp',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler()),('ridge',Ridge(alpha=20))])
                pipe.fit(Xtr[active],tr[target]); p=pipe.predict(Xte[active])
                y=te[target].to_numpy(float); sse=((y-p)**2).sum(); sse0=(y**2).sum()
                r2=1-sse/sse0 if sse0>0 else np.nan
                corr=float(np.corrcoef(y,p)[0,1]) if len(y)>2 and np.std(p)>0 and np.std(y)>0 else np.nan
                rows.append({'lead_days':lead,'horizon_days':horizon,'spec':spec,'status':'AVAILABLE','n_train':len(tr),'n_test':len(te),'n_active_features':len(active),'active_features':'|'.join(active),'oos_r2_vs_zero':float(r2),'corr':corr,'mean_prediction':float(np.mean(p)),'mean_actual':float(np.mean(y))})
                q=te[['decision_date',target,'regions_available','aggregation','weight_coverage']].copy(); q['prediction']=p; q['lead_days']=lead; q['horizon_days']=horizon; q['spec']=spec; q['active_features']='|'.join(active); preds.append(q)
    return pd.DataFrame(rows), (pd.concat(preds,ignore_index=True) if preds else pd.DataFrame())


def realized_vol_targets(market: pd.DataFrame, horizons=(3,5,10)):
    x=market.copy(); x=x.sort_values('date'); close=pd.to_numeric(x['close'],errors='coerce')
    r=np.log(close).diff()
    for h in horizons:
        x[f'rv_fwd_{h}d']=np.sqrt(sum((r.shift(-i)**2 for i in range(1,h+1))))
    return x


def run_revision_vol_grid(forecasts: pd.DataFrame, market: pd.DataFrame, production=None, leads=(1,3,5), horizons=(3,5,10)):
    m=realized_vol_targets(market,horizons); m['decision_date']=daily_key(m.date); m['month']=m.decision_date.dt.month
    rows=[]
    for lead in leads:
        a=_aggregate(forecasts,production,lead)
        if a.empty: continue
        feats=[c for c in ['temperature_mean_revision','temperature_max_revision','precipitation_sum_revision'] if c in a]
        for h in horizons:
            target=f'rv_fwd_{h}d'; z=a.merge(m[['decision_date','month',target]],on='decision_date').query('month >= 4 and month <= 9').dropna(subset=[target])
            if len(z)<80: continue
            cut=max(50,int(len(z)*.7)); tr=z.iloc[:cut]; te=z.iloc[cut:]
            active=_active_numeric_features(tr,feats)
            if not active:
                rows.append({'lead_days':lead,'horizon_days':h,'status':'NOT_EVALUABLE_NO_TRAIN_FEATURES','n_train':len(tr),'n_test':len(te),'n_active_features':0,'oos_r2_vs_train_mean':np.nan,'corr':np.nan})
                continue
            pipe=Pipeline([('imp',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler()),('ridge',Ridge(alpha=20))]); pipe.fit(tr[active],tr[target]); p=pipe.predict(te[active])
            y=te[target].to_numpy(float); base=np.full_like(y,float(tr[target].mean())); sse=((y-p)**2).sum(); sse0=((y-base)**2).sum()
            rows.append({'lead_days':lead,'horizon_days':h,'status':'AVAILABLE','n_train':len(tr),'n_test':len(te),'n_active_features':len(active),'active_features':'|'.join(active),'oos_r2_vs_train_mean':float(1-sse/sse0) if sse0>0 else np.nan,'corr':float(np.corrcoef(y,p)[0,1]) if np.std(p)>0 and np.std(y)>0 else np.nan})
    return pd.DataFrame(rows)
