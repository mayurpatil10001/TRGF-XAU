"""
src/strategy/meta_labeler.py  — Step 3: Meta-labeling secondary classifier
===========================================================================
De Prado style meta-labeling: a secondary binary classifier that predicts
whether the primary model's directional signal will CLEAR transaction costs.

Architecture:
  Primary model:  LightGBM/MLP/GatedFusion → direction (LONG/SHORT/NONE)
  Meta model:     LightGBM → P(trade is profitable after costs) ∈ [0, 1]

  Entry rule: trade ONLY when primary_prob >= thresh AND meta_prob >= meta_thresh

The meta-model is trained on validation fold signals only (label = 1 if
actual trade cleared costs, 0 if it did not). Features include primary model
probabilities, ATR, spread, session, and existing features.

Usage:
    from src.strategy.meta_labeler import MetaLabeler
    ml = MetaLabeler(cfg)
    ml.fit(X_val, primary_probs_val, actual_pnl_val)
    keep = ml.predict(X_test, primary_probs_test, spread, atr)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def make_meta_features(
    X: np.ndarray,       # [N, F] primary features (already scaled)
    probs: np.ndarray,   # [N, 3] primary model probs [p_none, p_long, p_short]
    spreads: np.ndarray, # [N] spread values
    atrs: np.ndarray,    # [N] ATR values
) -> np.ndarray:
    """
    Build meta-features: primary features + primary probs + derived inputs.
    Returns [N, F + 7] array.
    """
    p_none = probs[:, 0:1]
    p_long = probs[:, 1:2]
    p_short = probs[:, 2:3]
    p_max = np.maximum(p_long, p_short)
    p_confidence = p_max - p_none  # how much signal vs no-trade
    spread_atr_ratio = (spreads / (atrs + 1e-12)).reshape(-1, 1)
    spread_ = spreads.reshape(-1, 1)
    atr_ = atrs.reshape(-1, 1)

    extras = np.concatenate([
        p_none, p_long, p_short, p_max, p_confidence,
        spread_, atr_, spread_atr_ratio
    ], axis=1)
    return np.concatenate([X, extras], axis=1)


class MetaLabeler:
    """
    Wraps a LightGBM binary classifier that filters primary signals.
    """

    def __init__(self, cfg, meta_threshold: float = 0.55):
        self.cfg = cfg
        self.meta_threshold = meta_threshold
        self._model = None

    def fit(
        self,
        X: np.ndarray,           # primary features [N, F]
        probs: np.ndarray,       # primary probs [N, 3]
        spreads: np.ndarray,     # [N] spread at signal bar
        atrs: np.ndarray,        # [N] ATR at signal bar
        net_pnl: np.ndarray,     # [N] actual net PnL from backtest
        min_trades: int = 30,
    ) -> bool:
        """
        Train meta-labeler on validation fold outcomes.
        Returns True if enough samples and model trained successfully.
        """
        if not HAS_LGB:
            return False

        # Meta-label: 1 = trade was profitable after costs, 0 = not
        y_meta = (net_pnl > 0).astype(np.int32)
        n_pos = int(y_meta.sum())
        n_neg = int((y_meta == 0).sum())

        if len(y_meta) < min_trades or n_pos < 5 or n_neg < 5:
            return False  # not enough diversity

        X_meta = make_meta_features(X, probs, spreads, atrs)

        params = {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 16,
            "min_child_samples": 10,
            "scale_pos_weight": n_neg / (n_pos + 1e-9),
            "verbose": -1,
            "seed": self.cfg.seed,
        }
        try:
            self._model = lgb.LGBMClassifier(**params)
            self._model.fit(X_meta, y_meta)
            return True
        except Exception:
            return False

    def predict_proba(
        self,
        X: np.ndarray,
        probs: np.ndarray,
        spreads: np.ndarray,
        atrs: np.ndarray,
    ) -> np.ndarray:
        """Return P(profitable) for each sample. If no model, returns 1.0."""
        if self._model is None:
            return np.ones(len(X))
        X_meta = make_meta_features(X, probs, spreads, atrs)
        return self._model.predict_proba(X_meta)[:, 1]

    def filter_signals(
        self,
        primary_action: np.ndarray,   # [N] strings: "BUY", "SELL", "NO_TRADE"
        X: np.ndarray,
        probs: np.ndarray,
        spreads: np.ndarray,
        atrs: np.ndarray,
    ) -> np.ndarray:
        """
        Apply meta filter: set action to NO_TRADE if meta_prob < threshold.
        Returns filtered action array.
        """
        meta_probs = self.predict_proba(X, probs, spreads, atrs)
        filtered = primary_action.copy()
        filtered[(meta_probs < self.meta_threshold) & (primary_action != "NO_TRADE")] = "NO_TRADE"
        return filtered

    @property
    def is_fitted(self) -> bool:
        return self._model is not None
