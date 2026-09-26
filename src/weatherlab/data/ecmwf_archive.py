from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib

def _sha(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def archive_latest(out_root='data/snapshots/ecmwf', model='aifs-ens', params=('2t','tp'), steps=range(0,361,6)):
    try: from ecmwf.opendata import Client
    except ImportError: return {'status':'OPTIONAL_DEPENDENCY','how':'pip install -e .[weather]'}
    ts=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); root=Path(out_root)/ts; root.mkdir(parents=True,exist_ok=True)
    client=Client(source='aws', model=model)
    files=[]; errors=[]
    # Control + perturbed forecasts. ECMWF open AIFS-ENS supports type cf/pf.
    for typ in ('cf','pf'):
      for param in params:
        target=root/f'{model}_{typ}_{param}.grib2'
        try:
            client.retrieve(type=typ,param=param,step=list(steps),target=str(target))
            files.append({'path':str(target),'sha256':_sha(target),'bytes':target.stat().st_size,'type':typ,'param':param})
        except Exception as e:
            errors.append({'type':typ,'param':param,'error':str(e)})
    manifest={'retrieved_at':ts,'model':model,'files':files,'errors':errors,'status':'AVAILABLE' if files else 'UNAVAILABLE'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return {'status':manifest['status'],'root':str(root),'files':files,'errors':errors}
