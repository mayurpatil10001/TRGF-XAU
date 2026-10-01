"""
src/backtest/sizing.py
Position sizing: lots = floor_to_step((equity * kappa) / (SL_dist * contract_size))
Clamped to broker min/max lot.
"""
from __future__ import annotations

import math

from src.common.config import Config


def compute_lots(
    equity: float,
    sl_dist_price: float,
    cfg: Config,
    lot_min: float = 0.01,
    lot_max: float = 100.0,
    lot_step: float = 0.01,
) -> float:
    """
    Compute lot size. Returns 0.0 if the minimum lot risks more than 2*kappa.

    Parameters
    ----------
    equity: current account equity in USD
    sl_dist_price: SL distance in price units (e.g., 0.5)
    cfg: Config
    lot_min, lot_max, lot_step: broker constraints
    """
    kappa = cfg.risk.risk_fraction_kappa
    contract_size = cfg.symbol.contract_size

    if sl_dist_price <= 0:
        return 0.0

    risk_usd = equity * kappa
    raw_lots = risk_usd / (sl_dist_price * contract_size)
    lots = _floor_to_step(raw_lots, lot_step)
    lots = max(lot_min, min(lot_max, lots))

    # Skip if minimum lot risks more than 2*kappa
    min_lot_risk = lot_min * sl_dist_price * contract_size / equity
    if min_lot_risk > 2 * kappa:
        return 0.0

    return lots


def _floor_to_step(value: float, step: float) -> float:
    """Floor value to the nearest multiple of step."""
    return math.floor(value / step) * step
