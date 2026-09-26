from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from weatherlab.experiments.market_mechanism import _aggregate, _active_numeric_features, _supply_state
from weatherlab.timekeys import daily_key
from weatherlab.validation.bootstrap import block_bootstrap_stat


def _market_controls(market: pd.DataFrame) -> pd.DataFrame:
    """Construct strictly lagged market-state controls available before the target return.

    Same-day forward targets are never used as features. Price-derived controls are lagged by
    one observation to remain conservative about forecast publication/execution timing.
    """
    x = market.copy().sort_values('date').reset_index(drop=True)
    x['decision_date'] = daily_key(x['date'])
    close = pd.to_numeric(x['close'], errors='coerce')
    log_ret = np.log(close).diff()
    x['ret_lag1'] = log_ret.shift(1)
    x['rv5_lag1'] = np.sqrt(log_ret.pow(2).rolling(5, min_periods=3).sum()).shift(1)
    doy = x['decision_date'].dt.dayofyear.astype(float)
    x['season_sin'] = np.sin(2 * np.pi * doy / 365.25)
    x['season_cos'] = np.cos(2 * np.pi * doy / 365.25)
    return x


def _fit_ridge(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str):
    active = _active_numeric_features(train, features)
    if not active:
        return None, [], None
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scale', StandardScaler()),
        ('ridge', Ridge(alpha=20.0)),
    ])
    pipe.fit(train[active], train[target])
    return pipe, active, pipe.predict(test[active])


def run_incremental_weather_controls(
    forecasts: pd.DataFrame,
    market: pd.DataFrame,
    production: pd.DataFrame | None = None,
    leads=(1, 3, 5),
    horizons=(1, 3, 5, 10),
    growing_months=(4, 5, 6, 7, 8, 9),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Test whether weather revisions add OOS information beyond simple market controls.

    Two models are estimated on identical dates for every lead/horizon pair:
      controls-only = lagged return, lagged 5-day realized volatility, seasonality, lagged supply state
      controls+weather = controls-only plus point-in-time weather revisions

    Positive incremental R2 means weather lowers OOS squared error versus the controls-only model.
    """
    m = _market_controls(market)
    supply = _supply_state(production)
    rows, preds = [], []

    for lead in leads:
        a = _aggregate(forecasts, production, lead)
        if a.empty:
            continue
        a['decision_year'] = a['decision_date'].dt.year.astype('Int64')
        if supply is not None:
            a = a.merge(supply, on='decision_year', how='left')

        weather = [
            c for c in (
                'temperature_mean_revision',
                'temperature_max_revision',
                'precipitation_sum_revision',
            ) if c in a.columns
        ]

        for horizon in horizons:
            target = f'ret_fwd_{int(horizon)}d'
            if target not in m.columns:
                continue
            cols = ['decision_date', target, 'ret_lag1', 'rv5_lag1', 'season_sin', 'season_cos']
            z = a.merge(m[cols], on='decision_date', how='inner').sort_values('decision_date')
            z = z[z['decision_date'].dt.month.isin(growing_months)].dropna(subset=[target]).copy()
            if len(z) < 80:
                continue

            controls = ['ret_lag1', 'rv5_lag1', 'season_sin', 'season_cos']
            if 'tight_supply_proxy' in z.columns:
                controls.append('tight_supply_proxy')

            cut = max(50, int(len(z) * 0.7))
            tr, te = z.iloc[:cut].copy(), z.iloc[cut:].copy()
            base_model, base_active, p0 = _fit_ridge(tr, te, controls, target)
            wx_model, wx_active, p1 = _fit_ridge(tr, te, controls + weather, target)
            if p0 is None or p1 is None:
                rows.append({
                    'lead_days': lead, 'horizon_days': horizon,
                    'status': 'NOT_EVALUABLE_NO_TRAIN_FEATURES',
                    'n_train': len(tr), 'n_test': len(te),
                    'incremental_oos_r2_vs_controls': np.nan,
                })
                continue

            y = te[target].to_numpy(float)
            sse0 = float(np.sum((y - p0) ** 2))
            sse1 = float(np.sum((y - p1) ** 2))
            inc = 1.0 - sse1 / sse0 if sse0 > 0 else np.nan
            rows.append({
                'lead_days': lead,
                'horizon_days': horizon,
                'status': 'AVAILABLE',
                'n_train': len(tr),
                'n_test': len(te),
                'controls_features': '|'.join(base_active),
                'controls_weather_features': '|'.join(wx_active),
                'incremental_oos_r2_vs_controls': float(inc),
                'controls_mse': float(np.mean((y - p0) ** 2)),
                'controls_weather_mse': float(np.mean((y - p1) ** 2)),
            })
            q = te[['decision_date', target]].copy()
            q['lead_days'] = lead
            q['horizon_days'] = horizon
            q['controls_prediction'] = p0
            q['controls_weather_prediction'] = p1
            preds.append(q)

    grid = pd.DataFrame(rows)
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()

    inf_rows = []
    if not pred.empty:
        for (lead, horizon), g in pred.groupby(['lead_days', 'horizon_days']):
            target = f'ret_fwd_{int(horizon)}d'
            z = g.dropna(subset=[target, 'controls_prediction', 'controls_weather_prediction']).sort_values('decision_date')
            if len(z) < 30:
                continue
            y = z[target].to_numpy(float)
            p0 = z['controls_prediction'].to_numpy(float)
            p1 = z['controls_weather_prediction'].to_numpy(float)
            improvement = (y - p0) ** 2 - (y - p1) ** 2
            b = block_bootstrap_stat(improvement, block=max(5, int(horizon)), n_boot=2000, seed=42)
            inf_rows.append({
                'lead_days': int(lead),
                'horizon_days': int(horizon),
                'n': len(z),
                'mse_improvement_weather_vs_controls': float(np.mean(improvement)),
                'ci_low': b['ci_low'],
                'ci_high': b['ci_high'],
            })
    inference = pd.DataFrame(inf_rows)
    if not inference.empty:
        inference = inference.sort_values('mse_improvement_weather_vs_controls', ascending=False)
    return grid, pred, inference
