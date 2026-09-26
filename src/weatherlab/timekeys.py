from __future__ import annotations
import pandas as pd


def daily_key(values) -> pd.Series:
    """Canonical daily join key: timezone-naive midnight datetime64[ns].

    Parse all inputs as UTC first so strings, naive datetimes, and tz-aware datetimes
    represent the same calendar day. Returning tz-naive normalized timestamps avoids
    pandas merge dtype mismatches across providers.
    """
    s = pd.to_datetime(values, utc=True, errors='coerce', format='mixed')
    # Series/DatetimeIndex both support tz_localize(None) differently.
    if isinstance(s, pd.Series):
        return s.dt.tz_convert('UTC').dt.tz_localize(None).dt.normalize()
    return pd.Series(s.tz_convert('UTC').tz_localize(None).normalize(), index=getattr(values, 'index', None))


def normalize_region_name(value: str) -> str:
    return str(value).replace('_', ' ').strip().title()
