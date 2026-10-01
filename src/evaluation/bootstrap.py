"""
src/evaluation/bootstrap.py
Stationary block bootstrap for 95% CIs.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def _block_length(series: np.ndarray) -> int:
    """Choose block length from autocorrelation (Andrews 1991 rule of thumb)."""
    n = len(series)
    if n < 10:
        return 1
    # Lag-1 autocorrelation
    mu = series.mean()
    c0 = np.mean((series - mu) ** 2)
    c1 = np.mean((series[:-1] - mu) * (series[1:] - mu))
    rho = c1 / (c0 + 1e-12)
    rho = np.clip(rho, -0.99, 0.99)
    b = max(1, int(np.ceil(1.75 * n ** (1/3) * abs(rho) ** (2/3))))
    return min(b, n // 4)


def stationary_block_bootstrap(
    series: np.ndarray,
    statistic: Callable,
    n_resamples: int = 5000,
    block_length: int | None = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Stationary block bootstrap.

    Returns (stat_point, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    n = len(series)
    b = block_length or _block_length(series)

    stat_point = statistic(series)
    boot_stats = np.empty(n_resamples)

    for i in range(n_resamples):
        # Circular block bootstrap
        indices = []
        while len(indices) < n:
            start = rng.integers(0, n)
            block = [(start + j) % n for j in range(b)]
            indices.extend(block)
        sample = series[np.array(indices[:n])]
        boot_stats[i] = statistic(sample)

    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return float(stat_point), ci_lower, ci_upper
