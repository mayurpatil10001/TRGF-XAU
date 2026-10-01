"""tests/conftest.py — shared fixtures."""
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config, reset_config, Config


@pytest.fixture(scope="session")
def cfg() -> Config:
    reset_config()
    return load_config("config/config.yaml")


@pytest.fixture
def simple_bars() -> pd.DataFrame:
    """5-bar OHLCV test fixture with known structure."""
    times = pd.date_range("2020-01-02 08:00", periods=20, freq="1min", tz="UTC")
    np.random.seed(42)
    opens = 1800.0 + np.cumsum(np.random.randn(20) * 0.1)
    closes = opens + np.random.randn(20) * 0.15
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(20) * 0.05)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(20) * 0.05)
    return pd.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.random.randint(100, 500, 20).astype(float),
        "tick_volume": np.random.randint(50, 200, 20).astype(float),
        "spread": 0.25,
        "gap_before_minutes": 0.0,
        "session": "london",
        "spread_is_synthetic": True,
    })
