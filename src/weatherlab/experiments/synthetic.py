from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from weatherlab.reporting.bsic import apply_bsic, export_figure
from weatherlab.reporting.manifest import write_manifest

def run(seed=42):
    rng=np.random.default_rng(seed); n=700; spread=rng.uniform(.2,2,n); err=rng.normal(0,spread); pred=rng.normal(20,4,n); actual=pred+err
    corr=float(np.corrcoef(spread,np.abs(err))[0,1])
    fig,ax=plt.subplots(figsize=(6.8,3.8)); ax.scatter(spread,np.abs(err),s=10,alpha=.35); ax.set(xlabel='Ensemble spread',ylabel='Absolute forecast error'); apply_bsic(fig,ax,'Synthetic spread-skill diagnostic',['Synthetic smoke data']); export_figure(fig,'outputs/figures/diagnostics/synthetic_spread_skill',{'synthetic':True})
    rows=[{'hypothesis_id':'H2_spread_skill','experiment_id':'SYNTHETIC_SMOKE','status':'SMOKE_ONLY','sample':n,'metric':'corr(spread,abs(error))','estimate':corr,'uncertainty':None,'figure':'synthetic_spread_skill.svg','table':None,'notes':'Synthetic only; not research evidence'}]; write_manifest(rows); return rows
