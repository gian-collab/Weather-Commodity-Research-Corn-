from __future__ import annotations
from pathlib import Path
import pandas as pd

def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == '.parquet':
        try:
            df.to_parquet(p, index=False)
            return p
        except ImportError:
            p = p.with_suffix('.csv.gz')
    df.to_csv(p, index=False, compression='gzip' if p.suffix == '.gz' else None)
    return p

def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.exists():
        if p.suffix == '.parquet': return pd.read_parquet(p)
        return pd.read_csv(p)
    if p.suffix == '.parquet':
        alt = p.with_suffix('.csv.gz')
        if alt.exists(): return pd.read_csv(alt)
    raise FileNotFoundError(path)
