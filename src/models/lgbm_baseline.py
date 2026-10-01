"""
src/models/lgbm_baseline.py
LightGBM baseline: tabular multiclass (NONE/LONG/SHORT) + binary profitability.
Uses last-bar features + lag aggregates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common.config import Config
from src.common.logging import get_logger

logger = get_logger("models.lgbm")


def _make_tabular(feat_df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Aggregate lag features for tabular models."""
    lag_cols = [c for c in feat_df.columns if c.startswith("r_close_lag")][:lookback]
    other_cols = [c for c in feat_df.columns if c not in lag_cols]
    return feat_df[other_cols + lag_cols]


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
    n_classes: int = 3,
    seed: int = 42,
) -> lgb.Booster:
    """Train multiclass LightGBM with early stopping on validation log-loss."""
    # Class weights
    class_counts = np.bincount(y_train, minlength=n_classes)
    total = class_counts.sum()
    class_weights = total / (n_classes * (class_counts + 1))
    sample_weights = class_weights[y_train]

    train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    params: Dict[str, Any] = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": seed,
        "num_threads": -1,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=cfg.model.early_stop_patience, verbose=False),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=callbacks,
    )
    return model


def predict_lgbm(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """Return class probabilities [N, 3]."""
    return model.predict(X)


def save_lgbm(model: lgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info(f"Saved LightGBM model to {path}")


def load_lgbm(path: Path) -> lgb.Booster:
    model = lgb.Booster(model_file=str(path))
    logger.info(f"Loaded LightGBM model from {path}")
    return model
