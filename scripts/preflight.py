from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
import json,os,requests,pandas as pd
checks={}
urls={
'open_meteo_archive':'https://archive-api.open-meteo.com/v1/archive?latitude=41.878&longitude=-93.098&start_date=2025-01-01&end_date=2025-01-02&daily=temperature_2m_max&timezone=UTC&models=era5_land',
'open_meteo_previous':'https://previous-runs-api.open-meteo.com/v1/forecast?latitude=41.878&longitude=-93.098&start_date=2025-07-01&end_date=2025-07-02&hourly=temperature_2m_previous_day1&timezone=UTC&models=ecmwf_ifs025',
'yahoo_corn':'https://query1.finance.yahoo.com/v8/finance/chart/ZC=F?interval=1d&period1=1735689600&period2=1735948800'}
for k,u in urls.items():
    try:r=requests.get(u,timeout=20); checks[k]={'ok':r.ok,'status_code':r.status_code,'bytes':len(r.content)}
    except Exception as e:checks[k]={'ok':False,'error':str(e)}
checks['NASS_API_KEY']={'ok':bool(os.getenv('NASS_API_KEY')),'note':'Free key required for Quick Stats yield/production downloads'}
try:
 import ecmwf.opendata; checks['ecmwf_opendata_package']={'ok':True}
except Exception as e:checks['ecmwf_opendata_package']={'ok':False,'note':'Install .[weather] for prospective AIFS-ENS archiving'}
print(json.dumps(checks,indent=2))
