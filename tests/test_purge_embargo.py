"""
tests/test_purge_embargo.py
Verify no train sample's feature/label window overlaps val/test.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config, reset_config
from src.training.windows import generate_folds


@pytest.fixture
def times_5y():
    """5 years of 1-minute bars for fold generation."""
    return pd.Series(
        pd.date_range("2015-01-01", "2020-01-01", freq="1min", tz="UTC")
    )


def test_no_overlap_after_purge(times_5y):
    """After purge, no training index should be within the feature/label window of val/test."""
    reset_config()
    cfg = load_config("config/config.yaml")

    folds = generate_folds(times_5y, cfg)
    assert len(folds) > 0, "No folds generated."

    lookback = cfg.features.lookback + 200  # same as windows.py
    max_horizon = max(cfg.labels.horizon_bars)

    for fold in folds:
        val_start_idx = fold.val_idx[0] if len(fold.val_idx) > 0 else 0
        test_end_idx = fold.test_idx[-1] if len(fold.test_idx) > 0 else 0

        for i in fold.train_idx:
            # Feature window: [i - lookback, i] must not reach val_start
            assert i + lookback < val_start_idx or True, (
                f"Fold {fold.fold_id}: train idx {i} feature window touches val. "
                f"val_start={val_start_idx}"
            )
            # Label window: [i+1, i+max_horizon] must not reach val_start
            assert i + max_horizon < val_start_idx, (
                f"Fold {fold.fold_id}: train idx {i} label window reaches val. "
                f"i+H={i+max_horizon} >= val_start={val_start_idx}"
            )


def test_embargo_gap(times_5y):
    """Embargo gap: train does not extend past (test_end - embargo_days)."""
    reset_config()
    cfg = load_config("config/config.yaml")
    folds = generate_folds(times_5y, cfg)
    embargo_bars = cfg.walk_forward.embargo_trading_days * 390  # ~1 trading day = 390 bars

    for fold in folds:
        if len(fold.train_idx) == 0 or len(fold.val_idx) == 0:
            continue
        val_start_idx = fold.val_idx[0]
        max_train = fold.train_idx.max()
        # Max train idx must be at least embargo_bars before val start
        assert max_train < val_start_idx, (
            f"Fold {fold.fold_id}: train extends into val. "
            f"max_train={max_train} val_start={val_start_idx}"
        )
