from __future__ import annotations
from pathlib import Path
import json


def inventory_ensemble_snapshots(root='data/snapshots/ecmwf'):
    root=Path(root)
    rows=[]
    if not root.exists(): return {'status':'NO_SNAPSHOTS','snapshots':0,'files':0}
    for manifest in sorted(root.glob('*/manifest.json')):
        try: j=json.loads(manifest.read_text())
        except Exception: continue
        rows.append(j)
    return {'status':'AVAILABLE' if rows else 'NO_SNAPSHOTS','snapshots':len(rows),'files':sum(len(r.get('files',[])) for r in rows),'latest':rows[-1].get('retrieved_at') if rows else None}
