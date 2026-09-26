from __future__ import annotations
from pathlib import Path
import time
import pandas as pd
import numpy as np
from weatherlab.data.http import get_json
from weatherlab.data.storage import write_table, read_table
from weatherlab.provenance import ArtifactMeta, now_utc, write_meta
from weatherlab.timekeys import daily_key

ARCHIVE='https://archive-api.open-meteo.com/v1/archive'
PREVIOUS='https://previous-runs-api.open-meteo.com/v1/forecast'
SINGLE='https://single-runs-api.open-meteo.com/v1/forecast'
ENSEMBLE='https://ensemble-api.open-meteo.com/v1/ensemble'


def _frame(block: dict, key: str) -> pd.DataFrame:
    b = block.get(key) or {}
    if 'time' not in b: raise ValueError(f'Missing {key}.time in Open-Meteo response')
    df = pd.DataFrame(b)
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce', format='mixed')
    return df


def _chunks(start_date: str, end_date: str, years=2):
    start=pd.Timestamp(start_date); end=pd.Timestamp(end_date); cur=start
    while cur<=end:
        nxt=min(end,cur+pd.DateOffset(years=years)-pd.Timedelta(days=1)); yield cur.date().isoformat(),nxt.date().isoformat(); cur=nxt+pd.Timedelta(days=1)


def _fetch_chunked(url, params_builder, chunks, key, pause_seconds=1.0):
    frames=[]; errors=[]
    for i,(a,b) in enumerate(chunks):
        try:
            frames.append(_frame(get_json(url,params=params_builder(a,b)),key))
        except Exception as e:
            errors.append({'start':a,'end':b,'error':str(e)})
        if pause_seconds and i < len(chunks)-1: time.sleep(pause_seconds)
    if not frames:
        raise RuntimeError('All Open-Meteo chunks failed: '+str(errors[:3]))
    return pd.concat(frames,ignore_index=True), errors


def fetch_reanalysis_daily(lat: float, lon: float, start_date: str, end_date: str, out: str, model: str='era5_land', pause_seconds=1.0, reuse_existing=True):
    try:
        if reuse_existing:
            old=read_table(out)
            if not old.empty: return {'status':'CACHED','rows':len(old),'path':str(Path(out).with_suffix('.csv.gz') if not Path(out).exists() else Path(out)),'failed_chunks':[]}
    except FileNotFoundError:
        pass
    chunks=list(_chunks(start_date,end_date,years=3))
    def params(a,b): return {'latitude':lat,'longitude':lon,'start_date':a,'end_date':b,
                'daily':'temperature_2m_max,temperature_2m_min,precipitation_sum,et0_fao_evapotranspiration','timezone':'UTC','models':model}
    raw,errors=_fetch_chunked(ARCHIVE,params,chunks,'daily',pause_seconds)
    df=raw.drop_duplicates('time').sort_values('time').rename(columns={'time':'date'})
    df['date']=daily_key(df['date']); df['lat']=lat; df['lon']=lon; df['weather_kind']='REANALYSIS'; df['model']=model
    p=write_table(df,out)
    write_meta(str(p)+'.meta.json',ArtifactMeta('Open-Meteo Historical Weather / ERA5-Land',now_utc(),license='Open-Meteo terms / upstream attribution applies',notes=f'Reanalysis; chunks_failed={len(errors)}; never a substitute for historical forecast vintages'))
    return {'status':'AVAILABLE' if not errors else 'PARTIAL','rows':len(df),'path':str(p),'failed_chunks':errors}


def fetch_previous_runs(lat: float, lon: float, start_date: str, end_date: str, out: str, leads=(1,2,3,4,5,6,7), model: str | None='ecmwf_ifs025', pause_seconds=1.0, reuse_existing=True):
    try:
        if reuse_existing:
            old=read_table(out)
            if not old.empty: return {'status':'CACHED','rows':len(old),'path':str(Path(out).with_suffix('.csv.gz') if not Path(out).exists() else Path(out)),'failed_chunks':[]}
    except FileNotFoundError:
        pass
    vars_base=['temperature_2m','precipitation']; hourly=[]
    for v in vars_base: hourly.extend([f'{v}_previous_day{d}' for d in leads])
    chunks=list(_chunks(start_date,end_date,years=1))
    def params(a,b):
        q={'latitude':lat,'longitude':lon,'start_date':a,'end_date':b,'hourly':','.join(hourly),'timezone':'UTC'}
        if model: q['models']=model
        return q
    h,errors=_fetch_chunked(PREVIOUS,params,chunks,'hourly',pause_seconds)
    h=h.drop_duplicates('time').sort_values('time'); h['date']=daily_key(h['time'])
    rows=[]
    for d in leads:
        tc=f'temperature_2m_previous_day{d}'; pc=f'precipitation_previous_day{d}'
        if tc not in h.columns and pc not in h.columns: continue
        agg={}
        if tc in h.columns: agg.update(temperature_mean=(tc,'mean'),temperature_max=(tc,'max'))
        if pc in h.columns: agg.update(precipitation_sum=(pc,'sum'))
        g=h.groupby('date',as_index=False).agg(**agg)
        if 'temperature_mean' not in g: g['temperature_mean']=np.nan; g['temperature_max']=np.nan
        if 'precipitation_sum' not in g: g['precipitation_sum']=np.nan
        g['lead_days']=d; g['decision_date']=g['date']-pd.to_timedelta(d,unit='D'); rows.append(g)
    if not rows: raise ValueError('Open-Meteo previous-runs response had no requested variables')
    df=pd.concat(rows,ignore_index=True); df['decision_date']=daily_key(df['decision_date']); df['lat']=lat; df['lon']=lon; df['weather_kind']='HISTORICAL_FORECAST'; df['model']=model or 'open_meteo_best_match'
    p=write_table(df,out)
    write_meta(str(p)+'.meta.json',ArtifactMeta('Open-Meteo Previous Model Runs',now_utc(),license='Open-Meteo terms / upstream attribution applies',notes=f'Point-in-time fixed-lead archived model forecasts; model={model}; chunks_failed={len(errors)}'))
    return {'status':'AVAILABLE' if not errors else 'PARTIAL','rows':len(df),'path':str(p),'failed_chunks':errors}


def add_run_revisions(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy(); x['date']=daily_key(x['date'])
    if 'decision_date' not in x.columns: x['decision_date']=x['date']-pd.to_timedelta(pd.to_numeric(x['lead_days'],errors='coerce'),unit='D')
    x['decision_date']=daily_key(x['decision_date']); x=x.sort_values(['date','lead_days'])
    older=x.copy(); older['lead_days']=older['lead_days']-1
    older=older.rename(columns={'temperature_mean':'temperature_mean_older','temperature_max':'temperature_max_older','precipitation_sum':'precipitation_sum_older'})
    keep=['date','lead_days','temperature_mean_older','temperature_max_older','precipitation_sum_older']
    z=x.merge(older[keep],on=['date','lead_days'],how='left')
    for c in ['temperature_mean','temperature_max','precipitation_sum']: z[f'{c}_revision']=pd.to_numeric(z[c],errors='coerce')-pd.to_numeric(z[f'{c}_older'],errors='coerce')
    return z
