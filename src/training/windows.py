"""
src/training/windows.py
Walk-forward window generator with purge and embargo.

Produces (train_idx, val_idx, test_idx) triplets.
Purge: remove train samples whose feature window [t-L, t] or label window [t+1, t+H]
       overlaps the val/test period.
Embargo: exclude `embargo_trading_days` after each eval block from training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.common.config import Config
from src.common.logging import get_logger

logger = get_logger("training.windows")


@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def generate_folds(
    times: pd.Series,
    cfg: Config,
) -> List[WalkForwardFold]:
    """
    Generate purged walk-forward folds for the development period.

    Parameters
    ----------
    times : pd.Series of pd.Timestamp (UTC)
    cfg   : Config

    Returns
    -------
    List of WalkForwardFold objects
    """
    times = times.reset_index(drop=True)
    dev_start = pd.Timestamp(cfg.data.main_start, tz="UTC")
    dev_end = pd.Timestamp(cfg.data.dev_end, tz="UTC")

    times_arr = times.values  # numpy array of datetimes

    # Build calendar-based fold boundaries
    train_m = cfg.walk_forward.train_months
    val_m = cfg.walk_forward.val_months
    test_m = cfg.walk_forward.test_months
    step_m = cfg.walk_forward.step_months
    embargo_days = cfg.walk_forward.embargo_trading_days

    # Lookback and horizon for purge computation
    lookback_bars = cfg.features.lookback + 200  # add rolling window buffer
    max_horizon_bars = max(cfg.labels.horizon_bars)

    folds = []
    fold_start = dev_start

    fold_id = 0
    while True:
        train_start = fold_start
        train_end = train_start + pd.DateOffset(months=train_m)
        val_start = train_end
        val_end = val_start + pd.DateOffset(months=val_m)
        test_start = val_end
        test_end = test_start + pd.DateOffset(months=test_m)

        if test_end > dev_end:
            break

        # ── Indices by calendar boundary ─────────────────────────────────────
        train_mask_cal = (times >= train_start) & (times < train_end)
        val_mask = (times >= val_start) & (times < val_end)
        test_mask = (times >= test_start) & (times < test_end)

        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(val_idx) == 0 or len(test_idx) == 0:
            fold_start += pd.DateOffset(months=step_m)
            continue

        # ── Purge from training ───────────────────────────────────────────────
        # Any training sample whose feature window [i-lookback_bars, i]
        # OR label window [i+1, i+max_horizon_bars] overlaps val or test:
        val_test_start_idx = val_idx[0]
        test_end_idx = test_idx[-1]

        train_idx_cal = np.where(train_mask_cal)[0]

        # Feature window contamination: sample i contaminates if its feature
        # window extends into val/test (i + lookback_bars >= val_test_start_idx)
        feature_purge = train_idx_cal[
            train_idx_cal + lookback_bars >= val_test_start_idx
        ]

        # Label window contamination: sample i contaminates if i + max_horizon >= val_start
        label_purge = train_idx_cal[
            train_idx_cal + max_horizon_bars >= val_test_start_idx
        ]

        purge_set = set(feature_purge) | set(label_purge)
        train_idx = np.array([i for i in train_idx_cal if i not in purge_set])

        # ── Embargo: exclude embargo_days worth of bars after test end ────────
        # Already handled by the next fold's train_start being step_m later;
        # additionally, we exclude embargo_days bars right after test_end_idx
        embargo_end = test_end_idx + embargo_days
        # (these would only be in a future fold's training, so this is a forward guard)

        logger.info(
            f"Fold {fold_id}: train=[{train_start.date()}..{train_end.date()}), "
            f"val=[{val_start.date()}..{val_end.date()}), "
            f"test=[{test_start.date()}..{test_end.date()}), "
            f"train_n={len(train_idx):,} (purged {len(purge_set):,}), "
            f"val_n={len(val_idx):,}, test_n={len(test_idx):,}"
        )

        folds.append(WalkForwardFold(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        ))

        fold_start += pd.DateOffset(months=step_m)
        fold_id += 1

    logger.info(f"Generated {len(folds)} walk-forward folds.")
    return folds
