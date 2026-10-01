"""
tests/test_labels.py
Hand-crafted label tests: TP-first, SL-first, both-in-one-bar (SL wins), time exit.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config, reset_config
from src.labels.triple_barrier import (
    _scan_vectorized, LABEL_LONG, LABEL_SHORT, LABEL_NONE
)


def _make_bars(opens, highs, lows, closes, spread=0.25):
    n = len(opens)
    times = pd.date_range("2020-01-02 08:00", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "time": times,
        "open": opens, "high": highs, "low": lows, "close": closes,
        "tick_volume": np.ones(n) * 100.0,
        "spread": spread,
        "gap_before_minutes": np.zeros(n),
        "session": "london",
    })


class TestScanVectorized:
    """Test the scan_vectorized function directly with known inputs."""

    def _spreads(self, n, val=0.25):
        return np.full(n, val)

    def test_tp_hit_long(self):
        """TP hit on bar 2 for a long trade."""
        highs  = np.array([1800.5, 1801.5, 1802.0, 1800.0, 1800.0])
        lows   = np.array([1799.5, 1800.0, 1800.5, 1799.5, 1799.5])
        closes = np.array([1800.0, 1801.0, 1801.8, 1799.8, 1799.8])
        gaps   = np.zeros(5)
        spreads = self._spreads(5)
        entry = 1800.25  # ASK = open(1800) + spread(0.25)
        tp = 1801.5      # TP
        sl = 1799.5      # SL
        pnl = _scan_vectorized(highs, lows, closes, spreads, gaps,
                               t_start=1, t_end=4, tp=tp, sl=sl,
                               side=1, max_gap=30, entry=entry)
        assert pnl > 0, f"Expected positive PnL for TP hit, got {pnl}"

    def test_sl_hit_long(self):
        """SL hit on bar 1 for a long trade."""
        highs  = np.array([1800.5, 1800.1, 1800.0, 1800.0])
        lows   = np.array([1799.5, 1799.0, 1799.0, 1799.0])
        closes = np.array([1800.0, 1799.2, 1799.5, 1799.5])
        gaps   = np.zeros(4)
        spreads = self._spreads(4)
        entry = 1800.25
        tp = 1801.5
        sl = 1799.5  # low[1]=1799.0 < sl=1799.5 → hit
        pnl = _scan_vectorized(highs, lows, closes, spreads, gaps,
                               t_start=1, t_end=3, tp=tp, sl=sl,
                               side=1, max_gap=30, entry=entry)
        assert pnl < 0, f"Expected negative PnL for SL hit, got {pnl}"

    def test_sl_first_when_both_in_same_bar(self):
        """If both TP and SL are in same bar, SL should win (conservative)."""
        # high[1] >= tp AND low[1] <= sl → SL first
        highs  = np.array([1800.5, 1802.0, 1800.0])
        lows   = np.array([1799.5, 1798.0, 1800.0])
        closes = np.array([1800.0, 1800.0, 1800.0])
        gaps   = np.zeros(3)
        spreads = self._spreads(3)
        entry = 1800.25
        tp = 1801.5
        sl = 1799.0
        pnl = _scan_vectorized(highs, lows, closes, spreads, gaps,
                               t_start=1, t_end=2, tp=tp, sl=sl,
                               side=1, max_gap=30, entry=entry)
        # Result should be a loss (SL hit first)
        assert pnl < 0, f"Expected SL-first loss, got pnl={pnl}"

    def test_time_exit_long(self):
        """Time exit: no TP or SL hit within horizon."""
        highs  = np.array([1800.5, 1800.8, 1801.0])
        lows   = np.array([1799.5, 1799.8, 1799.9])
        closes = np.array([1800.0, 1800.3, 1800.5])
        gaps   = np.zeros(3)
        spreads = self._spreads(3)
        entry = 1800.25
        tp = 1803.0   # very far TP — won't be hit
        sl = 1797.0   # very far SL — won't be hit
        pnl = _scan_vectorized(highs, lows, closes, spreads, gaps,
                               t_start=1, t_end=2, tp=tp, sl=sl,
                               side=1, max_gap=30, entry=entry)
        # close[2]=1800.5, entry=1800.25 → positive but small
        assert isinstance(pnl, float), "Time exit should return float"


class TestFillCosts:
    """Test that fill prices are correct (R3)."""

    def test_long_entry_at_ask(self):
        from src.backtest.costs import entry_price_long
        bid = 1800.0
        spread = 0.25
        slip = 0.05
        ask = entry_price_long(bid, spread, slip)
        assert ask == pytest.approx(1800.30, abs=1e-6), \
            f"LONG entry should be at ASK (bid+spread+slip), got {ask}"

    def test_short_entry_at_bid(self):
        from src.backtest.costs import entry_price_short
        open_p = 1800.0
        slip = 0.05
        bid = entry_price_short(open_p, slip)
        assert bid == pytest.approx(1799.95, abs=1e-6), \
            f"SHORT entry should be at BID (open-slip), got {bid}"

    def test_commission_applied(self):
        from src.backtest.costs import net_pnl
        reset_config()
        cfg = load_config("config/config.yaml")
        # 1 lot BUY: entry 1800, exit 1801 → gross=100 USD, commission=7
        result = net_pnl("BUY", 1800.0, 1801.0, 1.0, cfg)
        assert result == pytest.approx(100.0 - 7.0, abs=0.01), \
            f"Commission not correctly deducted: {result}"
