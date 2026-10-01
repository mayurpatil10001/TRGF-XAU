"""
src/features/sessions.py
Session/time feature helpers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Default session windows (UTC)
DEFAULT_SESSIONS = {
    "asia":    ("00:00", "07:00"),
    "london":  ("07:00", "12:00"),
    "overlap": ("12:00", "16:00"),
    "ny":      ("16:00", "21:00"),
}


def _to_minutes(ts: pd.Series) -> pd.Series:
    """Convert timestamp series to minutes since midnight (UTC)."""
    return ts.dt.hour * 60 + ts.dt.minute


def session_features(df: pd.DataFrame, sessions: dict | None = None) -> pd.DataFrame:
    """
    Add time/session features to df (in-place copy).
    Returns a new DataFrame with added columns.
    """
    if sessions is None:
        sessions = DEFAULT_SESSIONS

    out = df.copy()
    t = out["time"]
    minutes = _to_minutes(t)

    # Minute-of-day sin/cos
    out["min_of_day_sin"] = np.sin(2 * np.pi * minutes / 1440)
    out["min_of_day_cos"] = np.cos(2 * np.pi * minutes / 1440)

    # Day-of-week one-hot (0=Mon..4=Fri)
    dow = t.dt.dayofweek
    for d in range(5):
        out[f"dow_{d}"] = (dow == d).astype(np.float32)

    # Session flags
    for name, (start_s, end_s) in sessions.items():
        sh, sm = map(int, start_s.split(":"))
        eh, em = map(int, end_s.split(":"))
        start_m = sh * 60 + sm
        end_m = eh * 60 + em
        out[f"session_{name}"] = ((minutes >= start_m) & (minutes < end_m)).astype(np.float32)

    # Minutes since London open (07:00 UTC)
    london_open_m = 7 * 60
    out["mins_since_london_open"] = (minutes - london_open_m).clip(lower=0).astype(np.float32)

    # Minutes since NY open (16:00 UTC)
    ny_open_m = 16 * 60
    out["mins_since_ny_open"] = (minutes - ny_open_m).clip(lower=0).astype(np.float32)

    return out
