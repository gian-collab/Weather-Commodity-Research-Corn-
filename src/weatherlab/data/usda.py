from __future__ import annotations
from pathlib import Path
import os
import pandas as pd
from weatherlab.data.http import session
from weatherlab.data.storage import write_table
from weatherlab.provenance import ArtifactMeta, now_utc, write_meta
API='https://quickstats.nass.usda.gov/api/api_GET/'


def quickstats(params, out, api_key=None):
    key=api_key or os.getenv('NASS_API_KEY')
    if not key:
        return {'status':'REQUIRES_KEY','source':'USDA NASS Quick Stats','how':'Create a free NASS API key and set NASS_API_KEY'}
    q=dict(params); q['key']=key; q['format']='json'
    r=session().get(API,params=q,timeout=180)
    if not r.ok:
        body=r.text[:1200]
        raise RuntimeError(f'NASS Quick Stats HTTP {r.status_code}: {body}; query={{{', '.join([repr(k)+': '+repr(v) for k,v in q.items() if k != 'key'])}}}')
    j=r.json()
    rows=j.get('data',[])
    if not rows and j.get('error'): raise RuntimeError(j['error'])
    df=pd.DataFrame(rows)
    p=write_table(df,out)
    write_meta(str(p)+'.meta.json',ArtifactMeta('USDA NASS Quick Stats',now_utc(),license='US Government public data',notes='API query='+str({k:v for k,v in q.items() if k!='key'})))
    return {'status':'AVAILABLE','rows':len(df),'path':str(p)}


def corn_yield_query(year_ge=1980):
    return {
        'source_desc':'SURVEY','commodity_desc':'CORN','statisticcat_desc':'YIELD',
        'agg_level_desc':'STATE','freq_desc':'ANNUAL','year__GE':str(year_ge)
    }

def corn_production_query(year_ge=1980):
    return {
        'source_desc':'SURVEY','commodity_desc':'CORN','statisticcat_desc':'PRODUCTION',
        'agg_level_desc':'STATE','freq_desc':'ANNUAL','year__GE':str(year_ge)
    }

def parse_numeric_value(s):
    if pd.isna(s): return float('nan')
    t=str(s).replace(',','').strip()
    if t in {'(D)','(Z)','(NA)',''}: return float('nan')
    try: return float(t)
    except ValueError: return float('nan')

def normalize_annual(df: pd.DataFrame, value_name='value') -> pd.DataFrame:
    x=df.copy()
    req={'year','state_name','Value'}
    if not req.issubset(x.columns): raise ValueError(f'NASS data missing {sorted(req-set(x.columns))}')
    x[value_name]=x['Value'].map(parse_numeric_value)
    if 'domain_desc' in x.columns and (x['domain_desc'].astype(str).str.upper()=='TOTAL').any(): x=x[x['domain_desc'].astype(str).str.upper()=='TOTAL']
    x['state_name']=x['state_name'].astype(str).str.title()
    x['year']=pd.to_numeric(x['year'],errors='coerce').astype('Int64')
    return x[['year','state_name',value_name]].dropna(subset=['year',value_name]).drop_duplicates(['year','state_name'],keep='last')
