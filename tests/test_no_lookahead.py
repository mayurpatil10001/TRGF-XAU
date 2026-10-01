"""
tests/test_no_lookahead.py
R2 enforcement: features at bar t must not depend on any bar > t.
Test method: perturb future bars and assert features at t are unchanged.
Also verifies scalers are fit only on train indices.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config, reset_config
from src.features.feature_pipeline import build_features, fit_scaler, apply_scaler


@pytest.fixture
def bars_100():
    """100 clean bars for lookahead testing."""
    times = pd.date_range("2020-01-02 08:00", periods=100, freq="1min", tz="UTC")
    np.random.seed(7)
    closes = 1800.0 + np.cumsum(np.random.randn(100) * 0.2)
    opens = closes + np.random.randn(100) * 0.1
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(100) * 0.05)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(100) * 0.05)
    return pd.DataFrame({
        "time": times, "open": opens, "high": highs,
        "low": lows, "close": closes,
        "tick_volume": np.ones(100) * 100.0,
        "spread": 0.25, "gap_before_minutes": 0.0,
        "session": "london", "spread_is_synthetic": True,
    })


def test_features_unchanged_after_future_perturbation(bars_100):
    """Perturb bars [t+1 .. N-1] and assert features at bar t are identical."""
    reset_config()
    cfg = load_config("config/config.yaml")

    feat_orig, names = build_features(bars_100.copy(), cfg)
    idx_t = feat_orig.index[50]  # pick bar 50 as the reference bar

    # Perturb ALL future bars (index > idx_t)
    perturbed = bars_100.copy()
    future_mask = perturbed.index > idx_t
    perturbed.loc[future_mask, "close"] += 999.0  # huge perturbation
    perturbed.loc[future_mask, "high"] += 999.0
    perturbed.loc[future_mask, "low"] -= 999.0

    feat_perturbed, _ = build_features(perturbed, cfg)

    # Features at bar t must be identical
    for col in names:
        if col in feat_orig.columns and col in feat_perturbed.columns:
            orig_val = feat_orig.loc[idx_t, col] if idx_t in feat_orig.index else np.nan
            pert_val = feat_perturbed.loc[idx_t, col] if idx_t in feat_perturbed.index else np.nan
            if not (np.isnan(orig_val) and np.isnan(pert_val)):
                assert np.isclose(orig_val, pert_val, equal_nan=True), (
                    f"LOOKAHEAD DETECTED in feature '{col}': "
                    f"orig={orig_val}, perturbed={pert_val}"
                )


def test_scaler_fit_only_on_train(bars_100):
    """Scaler must only be fit on train indices, not val/test."""
    reset_config()
    cfg = load_config("config/config.yaml")

    feat_df, names = build_features(bars_100.copy(), cfg)
    feat_df = feat_df.fillna(0.0)

    # Simulate a train/test split
    n = len(feat_df)
    train_end = int(n * 0.7)
    train_mask = pd.Series([i < train_end for i in range(n)], index=feat_df.index)

    scaler = fit_scaler(feat_df, train_mask)

    # Scaler center_ should match the median of train rows only
    train_rows = feat_df.loc[train_mask]
    expected_center = np.median(train_rows.values, axis=0)

    # Check a few columns
    np.testing.assert_allclose(
        scaler.center_[:5], expected_center[:5], rtol=1e-3,
        err_msg="Scaler center_ does not match train-only median — possible data leakage."
    )
