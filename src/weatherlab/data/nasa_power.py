from pathlib import Path
import pandas as pd
from weatherlab.data.http import get_json
from weatherlab.data.storage import write_table
from weatherlab.provenance import ArtifactMeta, now_utc, write_meta

def fetch_daily(lat, lon, start, end, out):
    params={'parameters':'T2M_MAX,T2M_MIN,PRECTOTCORR','community':'AG','longitude':lon,'latitude':lat,'start':start,'end':end,'format':'JSON'}
    u='https://power.larc.nasa.gov/api/temporal/daily/point'
    j=get_json(u,params=params); prm=j['properties']['parameter']
    df=pd.DataFrame(prm); df.index=pd.to_datetime(df.index,format='%Y%m%d',errors='coerce'); df.index.name='date'; df=df.reset_index()
    for c in ['T2M_MAX','T2M_MIN','PRECTOTCORR']:
        if c in df: df[c]=pd.to_numeric(df[c],errors='coerce').replace(-999.0,pd.NA)
    df['lat']=lat; df['lon']=lon; df['weather_kind']='OBSERVED_MODELLED'
    p=write_table(df,out)
    write_meta(str(p)+'.meta.json', ArtifactMeta('NASA POWER',now_utc(),license='NASA open data'))
    return {'status':'AVAILABLE','rows':len(df),'path':str(p)}
