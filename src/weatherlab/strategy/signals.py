import numpy as np
def risk_scaled(mu,sigma,threshold=0.0,cap=1.0):
    s=np.divide(mu,sigma,out=np.zeros_like(np.asarray(mu,dtype=float)),where=np.asarray(sigma)>0); s=np.where(np.abs(s)>=threshold,s,0); return np.clip(s,-cap,cap)
