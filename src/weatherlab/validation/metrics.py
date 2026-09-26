import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def point(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); m=np.isfinite(y)&np.isfinite(p); y=y[m]; p=p[m]
    return {'n':len(y),'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**.5,'r2':r2_score(y,p) if len(y)>1 else np.nan}
def pinball(y,q,tau): y=np.asarray(y); q=np.asarray(q); e=y-q; return float(np.nanmean(np.maximum(tau*e,(tau-1)*e)))
def coverage(y,lo,hi): y=np.asarray(y); return float(np.nanmean((y>=np.asarray(lo))&(y<=np.asarray(hi))))
def spread_skill(spread,error):
    a=np.asarray(spread,float); b=np.abs(np.asarray(error,float)); m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<3 or np.nanstd(a[m])==0 or np.nanstd(b[m])==0: return float('nan')
    return float(np.corrcoef(a[m],b[m])[0,1])
