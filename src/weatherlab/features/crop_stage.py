from __future__ import annotations
import numpy as np
import pandas as pd
from weatherlab.timekeys import daily_key, normalize_region_name

# Coarse Corn Belt phenology windows. These are intentionally stable calendar windows,
# not retrospective realized stage dates. They are used as a robustness baseline when
# point-in-time crop-progress vintages are unavailable.
STAGE_WINDOWS = {
    'planting': ((4, 1), (5, 31)),
    'vegetative': ((5, 15), (6, 30)),
    'silking': ((6, 20), (7, 31)),
    'grain_fill': ((7, 15), (8, 31)),
    'maturity': ((8, 15), (9, 30)),
}


def _longest_run(mask: pd.Series) -> int:
    a = mask.fillna(False).astype(bool).to_numpy()
    if len(a) == 0:
        return 0
    best = cur = 0
    for v in a:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def _in_window(ts: pd.Series, start, end):
    md = ts.dt.month * 100 + ts.dt.day
    a = start[0] * 100 + start[1]
    b = end[0] * 100 + end[1]
    return md.between(a, b)


def _numeric_alias(x: pd.DataFrame, candidates) -> pd.Series:
    """Return the first observed numeric weather field among accepted aliases.

    Open-Meteo and other adapters use different column names. A candidate that is
    present but entirely missing is not accepted if a later alias contains data.
    """
    fallback = pd.Series(np.nan, index=x.index, dtype=float)
    for c in candidates:
        if c in x.columns:
            v = pd.to_numeric(x[c], errors='coerce')
            if v.notna().any():
                return v
            fallback = v.astype(float)
    return fallback


def add_daily_stress(df: pd.DataFrame,
                     date='date', region='region',
                     tmax='T2M_MAX', tmin='T2M_MIN', precip='PRECTOTCORR',
                     et0='et0_fao_evapotranspiration') -> pd.DataFrame:
    x = df.copy()
    x[date] = daily_key(x[date])
    x = x.dropna(subset=[date]).copy()
    x[region] = x[region].map(normalize_region_name)
    mx = _numeric_alias(x, [tmax, 'temperature_2m_max', 'T2M_MAX'])
    mn = _numeric_alias(x, [tmin, 'temperature_2m_min', 'T2M_MIN'])
    pr = _numeric_alias(x, [precip, 'precipitation_sum', 'PRECTOTCORR'])
    et = _numeric_alias(x, [et0, 'et0_fao_evapotranspiration', 'ET0'])
    x['tmax_resolved'] = mx
    x['tmin_resolved'] = mn
    x['tmean'] = (mx + mn) / 2
    x['gdd10'] = np.clip(x['tmean'] - 10.0, 0, None)
    for th in (30, 32, 35):
        x[f'heat{th}'] = (mx >= th).astype(float)
    x['dry_day'] = (pr < 1.0).astype(float)
    x['heavy_rain_25'] = (pr >= 25.0).astype(float)
    x['precip'] = pr
    x['et0'] = et
    x['water_balance'] = pr - et
    x['hot_dry_32'] = ((mx >= 32) & (pr < 1.0)).astype(float)
    return x


def aggregate_stage_weather(df: pd.DataFrame, windows=None) -> pd.DataFrame:
    windows = windows or STAGE_WINDOWS
    x = add_daily_stress(df)
    x['year'] = x['date'].dt.year
    rows = []
    for (region, year), g in x.groupby(['region', 'year'], sort=True):
        base = {'region': region, 'year': int(year)}
        for stage, (a, b) in windows.items():
            s = g.loc[_in_window(g['date'], a, b)].sort_values('date')
            if s.empty:
                continue
            prefix = stage
            base[f'{prefix}_tmean'] = float(s['tmean'].mean())
            base[f'{prefix}_tmax_mean'] = float(pd.to_numeric(s['tmax_resolved'], errors='coerce').mean())
            base[f'{prefix}_precip_total'] = float(s['precip'].sum(min_count=1))
            base[f'{prefix}_gdd10'] = float(s['gdd10'].sum(min_count=1))
            base[f'{prefix}_heat30_days'] = float(s['heat30'].sum(min_count=1))
            base[f'{prefix}_heat32_days'] = float(s['heat32'].sum(min_count=1))
            base[f'{prefix}_heat35_days'] = float(s['heat35'].sum(min_count=1))
            base[f'{prefix}_dry_days'] = float(s['dry_day'].sum(min_count=1))
            base[f'{prefix}_heavy_rain_days'] = float(s['heavy_rain_25'].sum(min_count=1))
            base[f'{prefix}_hot_dry_days'] = float(s['hot_dry_32'].sum(min_count=1))
            base[f'{prefix}_water_balance'] = float(s['water_balance'].sum(min_count=1)) if s['water_balance'].notna().any() else np.nan
            base[f'{prefix}_max_hot_streak32'] = _longest_run(s['heat32'] > 0)
            base[f'{prefix}_max_dry_streak'] = _longest_run(s['dry_day'] > 0)
        rows.append(base)
    return pd.DataFrame(rows)


def mechanism_feature_groups(columns):
    cols = set(columns)
    groups = {
        'means': [c for c in cols if c.endswith(('_tmean', '_tmax_mean', '_precip_total', '_gdd10'))],
        'extremes': [c for c in cols if any(k in c for k in ('heat30_days','heat32_days','heat35_days','dry_days','heavy_rain_days','hot_dry_days','max_hot_streak32','max_dry_streak'))],
        'water_balance': [c for c in cols if c.endswith('_water_balance')],
    }
    groups['all_weather'] = sorted(set(sum(groups.values(), [])))
    return groups
