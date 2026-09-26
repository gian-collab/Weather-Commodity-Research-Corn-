from __future__ import annotations
import numpy as np
import pandas as pd
from weatherlab.validation.bootstrap import block_bootstrap_stat


def yield_paired_inference(paired: pd.DataFrame) -> pd.DataFrame:
    if paired is None or paired.empty: return pd.DataFrame()
    rows=[]
    for model,g in paired.groupby('model'):
        # Cluster first by year so ten states in the same crop year are not treated as independent.
        annual=g.groupby('year',as_index=False).ae_improvement.mean().sort_values('year')
        b=block_bootstrap_stat(annual.ae_improvement.to_numpy(float),block=3,n_boot=3000,seed=42)
        rows.append({'model':model,'n_state_years':len(g),'n_years':len(annual),'mean_ae_improvement':float(g.ae_improvement.mean()),'year_clustered_estimate':b['estimate'],'ci_low':b['ci_low'],'ci_high':b['ci_high'],'positive_years':int((annual.ae_improvement>0).sum())})
    return pd.DataFrame(rows).sort_values('year_clustered_estimate',ascending=False)


def market_grid_inference(pred: pd.DataFrame) -> pd.DataFrame:
    if pred is None or pred.empty: return pd.DataFrame()
    rows=[]
    target_cols=[c for c in pred.columns if c.startswith('ret_fwd_')]
    for (lead,h,spec),g in pred.groupby(['lead_days','horizon_days','spec']):
        target=f'ret_fwd_{int(h)}d'
        if target not in g: continue
        z=g.dropna(subset=[target,'prediction']).sort_values('decision_date')
        if len(z)<30: continue
        y=z[target].to_numpy(float); p=z.prediction.to_numpy(float)
        improvement=y**2-(y-p)**2  # positive = model lowers squared error vs zero-return forecast
        b=block_bootstrap_stat(improvement,block=max(5,int(h)),n_boot=2000,seed=42)
        rows.append({'lead_days':lead,'horizon_days':h,'spec':spec,'n':len(z),'mse_improvement':float(improvement.mean()),'ci_low':b['ci_low'],'ci_high':b['ci_high'],'directional_accuracy':float(np.mean(np.sign(y)==np.sign(p)))})
    return pd.DataFrame(rows).sort_values('mse_improvement',ascending=False)
