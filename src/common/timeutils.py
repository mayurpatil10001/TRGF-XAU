"""
src/common/timeutils.py
Time and calendar utilities.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Tuple

import pandas as pd


def to_utc(ts: pd.Timestamp, tz_in: str) -> pd.Timestamp:
    """Convert a naive or tz-aware timestamp to UTC."""
    if ts.tzinfo is None:
        if tz_in.upper() == "UTC":
            return ts.tz_localize("UTC")
        return ts.tz_localize(tz_in).tz_convert("UTC")
    return ts.tz_convert("UTC")


def parse_blackout(spec: str) -> Tuple[time, time]:
    """Parse '21:55-23:05' -> (time(21,55), time(23,5))."""
    start_s, end_s = spec.split("-")
    sh, sm = map(int, start_s.split(":"))
    eh, em = map(int, end_s.split(":"))
    return time(sh, sm), time(eh, em)


def is_in_blackout(ts: pd.Timestamp, blackout_utc: list[str]) -> bool:
    """True if the UTC timestamp falls within any blackout window."""
    t = ts.time()
    for spec in blackout_utc:
        start, end = parse_blackout(spec)
        if start <= end:
            if start <= t < end:
                return True
        else:  # wraps midnight
            if t >= start or t < end:
                return True
    return False


def get_session(ts: pd.Timestamp, sessions: dict) -> str:
    """Return the trading session name for a UTC timestamp."""
    t = ts.time()
    for name, (start_s, end_s) in sessions.items():
        sh, sm = map(int, start_s.split(":"))
        eh, em = map(int, end_s.split(":"))
        start, end = time(sh, sm), time(eh, em)
        if start <= t < end:
            return name
    return "off"


def next_minute_boundary() -> float:
    """Seconds until the next whole minute (+ a small buffer)."""
    import time as _time
    now = datetime.now(timezone.utc)
    seconds_past_minute = now.second + now.microsecond / 1_000_000
    return 60.0 - seconds_past_minute
