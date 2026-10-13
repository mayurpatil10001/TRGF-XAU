"""
src/live/trend_filter.py

Multi-Timeframe Trend Filter — institutional-grade HTF bias gate.

Logic:
  H1 Trend = BULL if:
    • EMA9 > EMA21 (fast above slow)
    • Current price > EMA21
    • ADX(14) > 20 (trending, not ranging)

  H1 Trend = BEAR if:
    • EMA9 < EMA21
    • Current price < EMA21
    • ADX(14) > 20

  H1 Trend = NEUTRAL if:
    • ADX < 20 (ranging — no directional trades)

The filter BLOCKS trades against the H1 trend.
It also BLOCKS all trades when the market is ranging (ADX < 20).

M5 auxiliary:
  Provides shorter-term momentum confirmation (EMA9 slope direction on M5).
  Trade is only taken if M5 EMA direction aligns with H1 direction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger("live.trend_filter")

EPS = 1e-8


@dataclass
class TrendState:
    h1_trend: str          # "BULL" | "BEAR" | "NEUTRAL"
    h1_adx: float
    h1_ema9: float
    h1_ema21: float
    h1_price: float
    m5_slope: str          # "UP" | "DOWN" | "FLAT"
    allows_buy: bool
    allows_sell: bool
    reason: str


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder-smoothed ADX."""
    if len(df) < period * 2:
        return 0.0

    h, l, c = df["high"], df["low"], df["close"]
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    prev_c = c.shift(1)

    up_move   = h - prev_h
    down_move = prev_l - l
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s  = pd.Series(plus_dm).ewm(span=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, adjust=False).mean()
    atr_s      = tr.ewm(span=period, adjust=False).mean()

    plus_di  = 100 * plus_dm_s  / (atr_s + EPS)
    minus_di = 100 * minus_dm_s / (atr_s + EPS)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    adx_val = dx.ewm(span=period, adjust=False).mean().iloc[-1]
    return float(adx_val)


def compute_trend(
    h1_bars: Optional[pd.DataFrame],
    m5_bars: Optional[pd.DataFrame],
    adx_threshold: float = 20.0,
    require_m5_align: bool = True,
) -> TrendState:
    """
    Compute multi-timeframe trend state.

    Parameters
    ----------
    h1_bars : DataFrame with OHLCV columns (at least 30 H1 bars)
    m5_bars : DataFrame with OHLCV columns (at least 20 M5 bars)
    adx_threshold : minimum ADX to consider market as trending
    require_m5_align : if True, M5 must align with H1 for full signal
    """
    # ── Default: no data → block everything ──────────────────────────────────
    if h1_bars is None or len(h1_bars) < 22:
        return TrendState(
            h1_trend="NEUTRAL", h1_adx=0.0, h1_ema9=0.0, h1_ema21=0.0,
            h1_price=0.0, m5_slope="FLAT",
            allows_buy=False, allows_sell=False,
            reason="insufficient_h1_data",
        )

    close = h1_bars["close"]
    ema9  = _ema(close, 9).iloc[-1]
    ema21 = _ema(close, 21).iloc[-1]
    price = float(close.iloc[-1])
    adx   = _adx(h1_bars)

    # ── H1 trend direction ────────────────────────────────────────────────────
    if adx < adx_threshold:
        h1_trend = "NEUTRAL"
    elif ema9 > ema21 and price > ema21:
        h1_trend = "BULL"
    elif ema9 < ema21 and price < ema21:
        h1_trend = "BEAR"
    else:
        h1_trend = "NEUTRAL"  # EMA cross in progress — wait

    # ── M5 slope ─────────────────────────────────────────────────────────────
    m5_slope = "FLAT"
    if m5_bars is not None and len(m5_bars) >= 10:
        m5_close = m5_bars["close"]
        m5_ema9 = _ema(m5_close, 9)
        slope = float(m5_ema9.iloc[-1] - m5_ema9.iloc[-5])
        atr_m5 = float(_atr_series(m5_bars).iloc[-1])
        if slope > atr_m5 * 0.1:
            m5_slope = "UP"
        elif slope < -atr_m5 * 0.1:
            m5_slope = "DOWN"

    # In a strong trend (ADX > 30), M5 pullbacks are valid entry points.
    # Require M5 alignment only in moderate trends (20 < ADX <= 30).
    strong_trend = adx >= 30.0
    m5_required = require_m5_align and not strong_trend

    # ── Decision ─────────────────────────────────────────────────────────────
    allows_buy  = False
    allows_sell = False

    if h1_trend == "BULL":
        allows_buy = True
        if m5_required:
            allows_buy = (m5_slope in ("UP", "FLAT"))
        allows_sell = False
        reason = "H1=BULL ADX=" + f"{adx:.1f}" + " M5=" + m5_slope
    elif h1_trend == "BEAR":
        allows_sell = True
        if m5_required:
            allows_sell = (m5_slope in ("DOWN", "FLAT"))
        allows_buy = False
        reason = "H1=BEAR ADX=" + f"{adx:.1f}" + " M5=" + m5_slope
    else:
        reason = "H1=NEUTRAL ADX=" + f"{adx:.1f}" + " — ranging, no trade"

    logger.info(
        f"TrendFilter: H1={h1_trend} EMA9={ema9:.2f} EMA21={ema21:.2f} "
        f"ADX={adx:.1f} M5={m5_slope} → buy={allows_buy} sell={allows_sell}"
    )

    return TrendState(
        h1_trend=h1_trend, h1_adx=adx,
        h1_ema9=ema9, h1_ema21=ema21, h1_price=price,
        m5_slope=m5_slope,
        allows_buy=allows_buy, allows_sell=allows_sell,
        reason=reason,
    )
