from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import os
import pandas as pd
import numpy as np
from weatherlab.data.http import get_json
from weatherlab.data.storage import write_table
from weatherlab.provenance import ArtifactMeta, now_utc, write_meta

YAHOO_CHART='https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'


def _normalize_market(df: pd.DataFrame, symbol='ZC=F', kind='CONTINUOUS_PROXY') -> pd.DataFrame:
    x=df.copy()
    cmap={c.lower():c for c in x.columns}
    date_col=next((cmap[k] for k in ('date','datetime','timestamp') if k in cmap),None)
    if not date_col: raise ValueError('Market data requires a date/datetime/timestamp column')
    x['date']=pd.to_datetime(x[date_col],utc=True,errors='coerce',format='mixed')
    for dst, aliases in {'open':['open'],'high':['high'],'low':['low'],'close':['close','settle','settlement','last'],'volume':['volume','vol'],'open_interest':['open_interest','open interest','oi']}.items():
        src=next((cmap[a] for a in aliases if a in cmap),None)
        if src: x[dst]=pd.to_numeric(x[src],errors='coerce')
    if 'close' not in x: raise ValueError('Market data requires close/settle/settlement/last')
    keep=['date']+[c for c in ['open','high','low','close','volume','open_interest'] if c in x]
    x=x[keep].dropna(subset=['date','close']).sort_values('date').drop_duplicates('date')
    x['symbol']=symbol; x['market_data_kind']=kind
    return x


def fetch_yahoo_continuous(start='2000-01-01', end=None, symbol='ZC=F', out='data/raw/market/corn_yahoo.parquet'):
    end=end or datetime.now(timezone.utc).date().isoformat()
    p1=int(pd.Timestamp(start,tz='UTC').timestamp()); p2=int((pd.Timestamp(end,tz='UTC')+pd.Timedelta(days=1)).timestamp())
    try:
        j=get_json(YAHOO_CHART.format(symbol=symbol),params={'period1':p1,'period2':p2,'interval':'1d','events':'history','includeAdjustedClose':'true'})
        result=j['chart']['result'][0]; ts=result['timestamp']; q=result['indicators']['quote'][0]
        adj=(result['indicators'].get('adjclose') or [{}])[0].get('adjclose',[None]*len(ts))
        df=pd.DataFrame({'date':pd.to_datetime(ts,unit='s',utc=True),'open':q.get('open'),'high':q.get('high'),'low':q.get('low'),'close':q.get('close'),'volume':q.get('volume'),'adj_close':adj})
        df=_normalize_market(df,symbol,'CONTINUOUS_PROXY_YAHOO')
        p=write_table(df,out)
        write_meta(str(p)+'.meta.json',ArtifactMeta('Yahoo Finance chart API '+symbol,now_utc(),license='Provider terms apply',notes='Unofficial continuous proxy; not official settlement or contract-level history'))
        return {'status':'AVAILABLE','rows':len(df),'path':str(p),'source':'yahoo'}
    except Exception as e:
        return {'status':'UNAVAILABLE','source':'Yahoo Finance chart API','error':str(e)}


def fetch_yfinance_continuous(start='2000-01-01', end=None, symbol='ZC=F', out='data/raw/market/corn_yfinance.parquet'):
    try:
        import yfinance as yf
    except ImportError:
        return {'status':'OPTIONAL_DEPENDENCY','source':'yfinance','how':'pip install -e .[market]'}
    try:
        end=end or datetime.now(timezone.utc).date().isoformat()
        df=yf.download(symbol,start=start,end=(pd.Timestamp(end)+pd.Timedelta(days=1)).date().isoformat(),auto_adjust=False,progress=False,threads=False)
        if df is None or df.empty: raise RuntimeError('empty yfinance response')
        if isinstance(df.columns,pd.MultiIndex): df.columns=[str(c[0]) for c in df.columns]
        df=df.reset_index()
        df=_normalize_market(df,symbol,'CONTINUOUS_PROXY_YFINANCE')
        p=write_table(df,out)
        write_meta(str(p)+'.meta.json',ArtifactMeta('yfinance/Yahoo '+symbol,now_utc(),license='Provider terms apply',notes='Unofficial continuous proxy; fallback when direct chart API is rate-limited'))
        return {'status':'AVAILABLE','rows':len(df),'path':str(p),'source':'yfinance'}
    except Exception as e:
        return {'status':'UNAVAILABLE','source':'yfinance','error':str(e)}


def ingest_market_csv(path, out='data/raw/market/corn_local.parquet', symbol='ZC'):
    p=Path(path)
    if not p.exists(): return {'status':'UNAVAILABLE','reason':f'file not found: {p}'}
    if p.suffix.lower() in {'.xlsx','.xls'}: df=pd.read_excel(p)
    else: df=pd.read_csv(p)
    x=_normalize_market(df,symbol,'USER_SUPPLIED_MARKET_HISTORY')
    o=write_table(x,out)
    write_meta(str(o)+'.meta.json',ArtifactMeta('User-supplied corn market history',now_utc(),notes=f'Normalized from {p.name}'))
    return {'status':'AVAILABLE','rows':len(x),'path':str(o),'source':'local'}


def fetch_public_corn_market(start='2000-01-01', end=None, symbol='ZC=F'):
    """Best-effort free-market chain. Never fabricates a series.
    Direct Yahoo -> optional yfinance. User-supplied institutional/CSV history is preferred when available.
    """
    y=fetch_yahoo_continuous(start,end,symbol)
    if y.get('status')=='AVAILABLE': return y
    yf=fetch_yfinance_continuous(start,end,symbol)
    if yf.get('status')=='AVAILABLE': return yf
    return {'status':'UNAVAILABLE','source':'public_market_chain','attempts':[y,yf],
            'notes':'Daily historical CME settlements are not bulk-free. Supply a local CSV/FactSet history or install .[market] and retry.'}


def ingest_factset(path, out='data/processed/market/factset.parquet'):
    p=Path(path); df=pd.read_excel(p)
    o=write_table(df,out)
    return {'status':'AVAILABLE','rows':len(df),'path':str(o)}


def forward_returns(df: pd.DataFrame, horizons=(1,3,5,10,20), price='close') -> pd.DataFrame:
    x=df.sort_values('date').copy()
    for h in horizons: x[f'ret_fwd_{h}d']=x[price].shift(-h)/x[price]-1
    x['ret_1d']=x[price].pct_change()
    x['rv_20d']=x['ret_1d'].rolling(20).std()*np.sqrt(252)
    return x
