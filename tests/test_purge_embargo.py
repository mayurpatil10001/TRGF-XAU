"""
tests/test_purge_embargo.py
Verify no train sample's feature/label window overlaps val/test.

IMPORTANT: Uses synthetic config override so main_start matches the fixture date range.
The production config has main_start=2026-06-22; using the 2015-2020 fixture
with that main_start would produce 0 folds (all dates are before 2026).
"""
import copy
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config, reset_config, Config
from src.training.windows import generate_folds


@pytest.fixture
def times_5y():
    """5 years of 1-minute bars for fold generation."""
    return pd.Series(
        pd.date_range("2015-01-01", "2020-01-01", freq="1min", tz="UTC")
    )


def _cfg_for_range(start: str, end: str) -> Config:
    """Load config and override main_start/dev_end to match a synthetic date range."""
    reset_config()
    cfg = load_config("config/config.yaml")
    # Override data dates via model_copy (Pydantic v2)
    cfg = cfg.model_copy(update={
        "data": cfg.data.model_copy(update={
            "main_start": start,
            "dev_end": end,
            "holdout_start": end,
        })
    })
    return cfg


def test_no_overlap_after_purge(times_5y):
    """After purge, no training index should be within the feature/label window of val/test."""
    # Use 2015-2020 range matching the fixture
    cfg = _cfg_for_range("2015-01-01", "2020-01-01")

    folds = generate_folds(times_5y, cfg)
    assert len(folds) > 0, (
        "No folds generated. Check that main_start/dev_end span at least "
        f"{cfg.walk_forward.train_months + cfg.walk_forward.val_months + cfg.walk_forward.test_months} months."
    )

    lookback = cfg.features.lookback + 200  # same as windows.py
    max_horizon = max(cfg.labels.horizon_bars)

    for fold in folds:
        val_start_idx = fold.val_idx[0] if len(fold.val_idx) > 0 else 0
        test_end_idx = fold.test_idx[-1] if len(fold.test_idx) > 0 else 0

        for i in fold.train_idx:
            # Label window: [i+1, i+max_horizon] must not reach val_start
            assert i + max_horizon < val_start_idx, (
                f"Fold {fold.fold_id}: train idx {i} label window reaches val. "
                f"i+H={i+max_horizon} >= val_start={val_start_idx}"
            )


def test_embargo_gap(times_5y):
    """Embargo gap: train max idx must be below val start idx."""
    cfg = _cfg_for_range("2015-01-01", "2020-01-01")
    folds = generate_folds(times_5y, cfg)
    assert len(folds) > 0, "No folds generated — check date range."

    for fold in folds:
        if len(fold.train_idx) == 0 or len(fold.val_idx) == 0:
            continue
        val_start_idx = fold.val_idx[0]
        max_train = fold.train_idx.max()
        assert max_train < val_start_idx, (
            f"Fold {fold.fold_id}: train extends into val. "
            f"max_train={max_train} val_start={val_start_idx}"
        )
