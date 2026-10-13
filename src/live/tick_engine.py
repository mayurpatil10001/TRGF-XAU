"""
src/live/tick_engine.py

HFT Tick Engine — collects live tick data and computes micro-features
for the HFT scalping model. Polls MT5 every ~0.2s and maintains a
rolling buffer of recent ticks.

Micro-features computed:
  - tick_momentum   : price velocity over last N ticks
  - bid_ask_slope   : spread trend (widening = volatility event)
  - micro_rsi       : fast RSI on last 14 ticks
  - price_accel     : second derivative (acceleration / deceleration)
  - vol_burst       : is current tick velocity > 2σ of recent mean?
  - h1_bear / h1_bull: regime from HTF filter
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque
import time
import numpy as np

EPS = 1e-8

@dataclass
class Tick:
    ts: float       # unix timestamp
    bid: float
    ask: float
    mid: float
    spread: float


@dataclass
class TickFeatures:
    momentum_5:   float   # avg price change over last 5 ticks
    momentum_20:  float   # avg price change over last 20 ticks
    accel:        float   # acceleration = momentum_5 - momentum_20
    micro_rsi:    float   # 0-100 RSI on last 14 tick mid-prices
    spread_delta: float   # change in spread vs 10-tick mean
    vol_burst:    bool    # tick velocity > 2σ
    tick_count:   int     # total ticks collected


class TickEngine:
    """
    Maintains a rolling window of raw ticks and computes micro-features
    on demand. Thread-safe for single-threaded asyncio use.
    """

    def __init__(self, maxlen: int = 200):
        self.buf: Deque[Tick] = deque(maxlen=maxlen)
        self._last_ts: float = 0.0

    def push(self, tick: Tick) -> None:
        """Add a new tick to the buffer."""
        self.buf.append(tick)
        self._last_ts = tick.ts

    def push_from_mt5(self, mt5_tick) -> Optional[Tick]:
        """Convert MT5 tick object and push. Returns None if duplicate."""
        ts = float(mt5_tick.time_msc) / 1000.0
        if ts <= self._last_ts:
            return None
        t = Tick(
            ts=ts,
            bid=float(mt5_tick.bid),
            ask=float(mt5_tick.ask),
            mid=(float(mt5_tick.bid) + float(mt5_tick.ask)) / 2.0,
            spread=float(mt5_tick.ask) - float(mt5_tick.bid),
        )
        self.push(t)
        return t

    def features(self) -> Optional[TickFeatures]:
        """Compute micro-features from current buffer. Returns None if insufficient data."""
        if len(self.buf) < 20:
            return None

        mids = np.array([t.mid for t in self.buf], dtype=np.float64)
        spreads = np.array([t.spread for t in self.buf], dtype=np.float64)
        diffs = np.diff(mids)  # tick-to-tick changes

        # ── Momentum ─────────────────────────────────────────────────────────
        mom5  = float(np.mean(diffs[-5:]))   if len(diffs) >= 5  else 0.0
        mom20 = float(np.mean(diffs[-20:]))  if len(diffs) >= 20 else 0.0
        accel = mom5 - mom20

        # ── Micro RSI (14 ticks) ─────────────────────────────────────────────
        rsi_diffs = diffs[-14:] if len(diffs) >= 14 else diffs
        gains = rsi_diffs[rsi_diffs > 0]
        losses = -rsi_diffs[rsi_diffs < 0]
        avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else EPS
        rs = avg_gain / (avg_loss + EPS)
        micro_rsi = 100.0 - (100.0 / (1.0 + rs))

        # ── Spread delta ─────────────────────────────────────────────────────
        spread_mean10 = float(np.mean(spreads[-10:])) if len(spreads) >= 10 else spreads[-1]
        spread_delta = float(spreads[-1]) - spread_mean10

        # ── Volatility burst ─────────────────────────────────────────────────
        if len(diffs) >= 30:
            vel_mean = float(np.mean(np.abs(diffs[-30:])))
            vel_std  = float(np.std(np.abs(diffs[-30:]))) + EPS
            vol_burst = abs(diffs[-1]) > vel_mean + 2.0 * vel_std
        else:
            vol_burst = False

        return TickFeatures(
            momentum_5=mom5,
            momentum_20=mom20,
            accel=accel,
            micro_rsi=micro_rsi,
            spread_delta=spread_delta,
            vol_burst=vol_burst,
            tick_count=len(self.buf),
        )

    @property
    def latest_mid(self) -> Optional[float]:
        return self.buf[-1].mid if self.buf else None

    @property
    def latest_spread(self) -> Optional[float]:
        return self.buf[-1].spread if self.buf else None
