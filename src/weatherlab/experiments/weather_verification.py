from __future__ import annotations
import numpy as np, pandas as pd
from weatherlab.timekeys import daily_key, normalize_region_name


def verify_archived_forecasts(forecasts: pd.DataFrame, realized: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Verify fixed-lead archived forecasts against realized/reanalysis weather.

    This measures the professional-model forecast product we actually observe; it does
    not claim reanalysis was known ex ante. Reanalysis is used only as the ex-post truth proxy.
    """
    f=forecasts.copy(); r=realized.copy()
    f['date']=daily_key(f['date']); r['date']=daily_key(r['date'])
    f['region']=f['region'].map(normalize_region_name); r['region']=r['region'].map(normalize_region_name)
    # Reanalysis daily mean reconstructed from daily min/max; temperature max is direct.
    r['actual_temperature_mean']=(pd.to_numeric(r['temperature_2m_max'],errors='coerce')+pd.to_numeric(r['temperature_2m_min'],errors='coerce'))/2
    r['actual_temperature_max']=pd.to_numeric(r['temperature_2m_max'],errors='coerce')
    r['actual_precipitation_sum']=pd.to_numeric(r['precipitation_sum'],errors='coerce')
    z=f.merge(r[['date','region','actual_temperature_mean','actual_temperature_max','actual_precipitation_sum']],on=['date','region'],how='inner')
    pairs=[('temperature_mean','actual_temperature_mean'),('temperature_max','actual_temperature_max'),('precipitation_sum','actual_precipitation_sum')]
    rows=[]
    for lead,g in z.groupby('lead_days'):
        for fc,ac in pairs:
            y=pd.to_numeric(g[ac],errors='coerce').to_numpy(float); p=pd.to_numeric(g[fc],errors='coerce').to_numpy(float)
            m=np.isfinite(y)&np.isfinite(p)
            if m.sum()==0: continue
            err=p[m]-y[m]
            rows.append({'lead_days':int(lead),'variable':fc,'n':int(m.sum()),'mae':float(np.mean(np.abs(err))),'rmse':float(np.sqrt(np.mean(err**2))),'bias':float(np.mean(err))})
    return z,pd.DataFrame(rows)


def revision_instability_skill(forecasts_with_revisions: pd.DataFrame, realized: pd.DataFrame, lead_days=1) -> dict:
    """Test whether a large run-to-run revision identifies hard-to-forecast cases.

    This is NOT ensemble spread. It is explicitly labelled run-revision instability.
    """
    f=forecasts_with_revisions[forecasts_with_revisions.lead_days==lead_days].copy(); r=realized.copy()
    f['date']=daily_key(f['date']); r['date']=daily_key(r['date'])
    f['region']=f['region'].map(normalize_region_name); r['region']=r['region'].map(normalize_region_name)
    r['actual_temperature_mean']=(pd.to_numeric(r['temperature_2m_max'],errors='coerce')+pd.to_numeric(r['temperature_2m_min'],errors='coerce'))/2
    z=f.merge(r[['date','region','actual_temperature_mean']],on=['date','region'],how='inner')
    rev=np.abs(pd.to_numeric(z['temperature_mean_revision'],errors='coerce')).to_numpy(float)
    err=np.abs(pd.to_numeric(z['temperature_mean'],errors='coerce').to_numpy(float)-pd.to_numeric(z['actual_temperature_mean'],errors='coerce').to_numpy(float))
    m=np.isfinite(rev)&np.isfinite(err)
    if m.sum()<30 or np.std(rev[m])==0 or np.std(err[m])==0: return {'status':'INSUFFICIENT_DATA','n':int(m.sum())}
    return {'status':'AVAILABLE','n':int(m.sum()),'lead_days':lead_days,'corr_abs_revision_abs_error':float(np.corrcoef(rev[m],err[m])[0,1]),'label':'run_revision_instability_not_ensemble_spread'}
