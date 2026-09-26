from pathlib import Path
import json, pandas as pd
REQUIRED=['hypothesis_id','experiment_id','status','sample','metric','estimate','uncertainty','figure','table','notes']
def write_manifest(rows, out='outputs/results'):
    out=Path(out); out.mkdir(parents=True, exist_ok=True)
    df=pd.DataFrame(rows)
    for c in REQUIRED:
        if c not in df: df[c]=None
    df=df[REQUIRED]
    df.to_csv(out/'results_manifest.csv', index=False)
    (out/'results_manifest.json').write_text(json.dumps(df.to_dict('records'), indent=2, default=str), encoding='utf-8')
    return df
