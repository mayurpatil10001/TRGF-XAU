"""
src/evaluation/metrics.py
Forecast and trading performance metrics.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import stats


def directional_accuracy(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """Directional accuracy: fraction of correct directional forecasts."""
    pred_dir = np.argmax(y_pred_prob, axis=1)  # 0=NONE,1=LONG,2=SHORT
    # Consider LONG(1) as +1, SHORT(2) as -1, NONE(0) as neutral
    # Accuracy = fraction where pred class matches true class
    return float((pred_dir == y_true).mean())


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Precision, recall, F1 per class."""
    from sklearn.metrics import classification_report
    return classification_report(y_true, y_pred, labels=[0, 1, 2],
                                 target_names=["NONE", "LONG", "SHORT"],
                                 output_dict=True, zero_division=0)


def nll_gaussian(y_true: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> float:
    """Gaussian NLL."""
    return float(np.mean(0.5 * np.log(2 * np.pi * scale ** 2) + ((y_true - loc) / scale) ** 2 / 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def profit_factor(wins: np.ndarray, losses: np.ndarray) -> float:
    """Profit factor = sum(wins) / sum(abs(losses))."""
    w = wins[wins > 0].sum()
    l = np.abs(losses[losses < 0]).sum()
    if l == 0:
        return float("inf") if w > 0 else 1.0
    return float(w / l)


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252 * 390) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma == 0:
        return 0.0
    return float(mu / sigma * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, periods_per_year: int = 252 * 390) -> float:
    """Annualized Sortino ratio."""
    if len(returns) < 2:
        return 0.0
    mu = returns.mean()
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf") if mu > 0 else 0.0
    sigma_d = np.std(downside, ddof=1)
    if sigma_d == 0:
        return 0.0
    return float(mu / sigma_d * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> Dict[str, float]:
    """Maximum drawdown: depth and duration in bars."""
    peak = equity_curve[0]
    max_dd = 0.0
    max_dd_duration = 0
    curr_duration = 0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / (peak + 1e-12)
        max_dd = max(max_dd, dd)
        curr_duration = curr_duration + 1 if v < peak else 0
        max_dd_duration = max(max_dd_duration, curr_duration)
    return {"max_drawdown_pct": float(max_dd * 100), "max_dd_duration_bars": int(max_dd_duration)}


def expectancy_per_trade(pnl_series: np.ndarray) -> float:
    """Mean P/L per trade."""
    if len(pnl_series) == 0:
        return 0.0
    return float(pnl_series.mean())


def breakeven_win_rate(rr_ratio: float) -> float:
    """p* = 1/(1+RR) — the win rate needed to break even."""
    return 1.0 / (1.0 + rr_ratio)
