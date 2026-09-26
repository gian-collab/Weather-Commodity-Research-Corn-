from __future__ import annotations
import numpy as np

def block_bootstrap_stat(values,stat=np.mean,block=5,n_boot=2000,seed=42):
    x=np.asarray(values,float); x=x[np.isfinite(x)]; n=len(x)
    if n==0:return {'estimate':np.nan,'ci_low':np.nan,'ci_high':np.nan}
    rng=np.random.default_rng(seed); starts=np.arange(max(1,n-block+1)); vals=[]
    for _ in range(n_boot):
        out=[]
        while len(out)<n:
            s=int(rng.choice(starts)); out.extend(x[s:min(n,s+block)].tolist())
        vals.append(stat(np.asarray(out[:n])))
    return {'estimate':float(stat(x)),'ci_low':float(np.quantile(vals,.025)),'ci_high':float(np.quantile(vals,.975))}
