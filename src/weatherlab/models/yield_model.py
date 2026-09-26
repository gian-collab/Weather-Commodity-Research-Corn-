from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

def models(seed=42):
    return {
        'ridge':Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=10))]),
        'boosting':HistGradientBoostingRegressor(random_state=seed,l2_regularization=1.0,min_samples_leaf=8)
    }
