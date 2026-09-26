import numpy as np
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

def model_zoo(seed=42):
    return {'ridge':Ridge(alpha=10),'rf':RandomForestRegressor(n_estimators=300,min_samples_leaf=5,random_state=seed,n_jobs=-1),'hgb':HistGradientBoostingRegressor(random_state=seed),'q10':QuantileRegressor(quantile=.1,alpha=.01),'q50':QuantileRegressor(quantile=.5,alpha=.01),'q90':QuantileRegressor(quantile=.9,alpha=.01)}

def ensemble_mean(predictions): return np.mean(np.column_stack(predictions),axis=1)
def ensemble_spread(predictions): return np.std(np.column_stack(predictions),axis=1,ddof=1)
