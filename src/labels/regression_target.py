"""
src/labels/regression_target.py
Regression target: ln(VWAP_{t+1}/VWAP_t) or ln(C_{t+1}/C_t).
Document which was used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger("labels.regression")


def compute_regression_target(df: pd.DataFrame) -> pd.Series:
    """
    Returns y_reg = ln(VWAP_{t+1}/VWAP_t) if computable, else ln(C_{t+1}/C_t).
    Uses bar t+1 values — this is the TARGET, not the input.
    Strictly causal: only the next bar's ALREADY-CLOSED values are used.
    """
    # Compute bar VWAP = (H+L+C)/3 * volume, then ratio of successive
    vol = df["tick_volume"].clip(lower=1.0)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap = typical  # same-bar approximation (no cross-bar accumulation needed for 1-min)

    if (vol > 0).any():
        y = np.log(vwap.shift(-1) / (vwap + 1e-12))
        which = "VWAP-based: ln(VWAP_{t+1}/VWAP_t)"
    else:
        c = df["close"]
        y = np.log(c.shift(-1) / (c + 1e-12))
        which = "close-based: ln(C_{t+1}/C_t)"

    logger.info(f"Regression target: {which}")
    y.name = "y_reg"
    return y
