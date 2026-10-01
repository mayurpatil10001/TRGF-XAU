"""
src/features/timing.py
Group E – Timing features.

OHLC proxy (no tick data needed historically):
  high_first_proxy: 1 if bar structure suggests high formed before low
  timing_surprise_proxy: 1 if implied high/low order contradicts bar direction
  timing_continuous: sign(C-O) * (upper_wick - lower_wick)

True intra-bar timing (requires MT5 tick data):
  compute_true_timing() — for live and recent historical validation only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger("features.timing")


def timing_proxy(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    Add OHLC-based timing proxy features.

    These are clearly labelled 'proxy' in feature names.
    Their agreement rate with true intra-bar tick timing is validated
    in scripts/compare_timing_true_vs_proxy.py.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = h - l + eps
    body = c - o

    # Upper/lower wick fractions
    upper_wick = (h - np.maximum(o, c)) / rng
    lower_wick = (np.minimum(o, c) - l) / rng

    # close_location in [0,1]: 1=close near high, 0=close near low
    close_loc = (c - l) / rng

    # Continuous timing measure: bull bars tend to hit low first then rally
    timing_cont = np.sign(body) * (upper_wick - lower_wick)

    # Proxy for high-first: bearish bars (C<O) with small lower wick
    # suggest price fell quickly (high formed early)
    high_first_proxy = (
        ((body < 0) & (lower_wick < 0.2)) |   # bearish, small lower wick
        ((body >= 0) & (upper_wick < 0.2))     # bullish, small upper wick (low first)
    ).astype(np.float32)
    # Flip for bull: high first = price ran up quickly then reversed → upper_wick large
    high_first_proxy = np.where(body >= 0,
                                (upper_wick > lower_wick).astype(np.float32),
                                (lower_wick < upper_wick).astype(np.float32))

    # Timing surprise: implied order contradicts bar direction
    # Bull bar but high formed before low = timing surprise for bulls (reversal risk)
    timing_surprise_proxy = np.where(
        body >= 0,
        (upper_wick > lower_wick).astype(np.float32),   # high before low in bull bar
        (lower_wick > upper_wick).astype(np.float32),   # low before high in bear bar
    )

    out = df.copy()
    out["timing_high_first_proxy"] = high_first_proxy.astype(np.float32)
    out["timing_surprise_proxy"] = timing_surprise_proxy.astype(np.float32)
    out["timing_continuous_proxy"] = timing_cont.astype(np.float32)

    return out


def compute_true_timing(
    symbol: str,
    bar_times: pd.DatetimeIndex,
    mt5_module=None,
) -> pd.DataFrame:
    """
    Compute TRUE intra-bar high/low timestamps using MT5 tick data.

    Returns a DataFrame with columns:
      bar_time, true_high_first (bool), true_low_first (bool), tick_count
    Available only where MT5 is connected and ticks exist.
    """
    if mt5_module is None:
        try:
            import MetaTrader5 as mt5
            mt5_module = mt5
        except ImportError:
            logger.warning("MetaTrader5 not available; true timing unavailable.")
            return pd.DataFrame()

    records = []
    for bar_start in bar_times:
        bar_end = bar_start + pd.Timedelta(minutes=1)
        ticks = mt5_module.copy_ticks_range(
            symbol,
            bar_start.to_pydatetime(),
            bar_end.to_pydatetime(),
            mt5_module.COPY_TICKS_ALL,
        )
        if ticks is None or len(ticks) == 0:
            continue
        tick_df = pd.DataFrame(ticks)
        tick_df["time"] = pd.to_datetime(tick_df["time"], unit="s", utc=True)
        mid = (tick_df["bid"] + tick_df["ask"]) / 2
        tick_df["mid"] = mid
        idx_high = mid.idxmax()
        idx_low = mid.idxmin()
        records.append({
            "bar_time": bar_start,
            "true_high_first": idx_high < idx_low,
            "true_low_first": idx_low < idx_high,
            "tick_count": len(tick_df),
        })

    return pd.DataFrame(records)
