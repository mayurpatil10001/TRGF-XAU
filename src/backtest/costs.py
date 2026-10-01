"""
src/backtest/costs.py
Cost calculations for the backtester.
"""
from __future__ import annotations

from src.common.config import Config


def ask_price(bid: float, spread: float) -> float:
    return bid + spread


def bid_price(ask: float, spread: float) -> float:
    return ask - spread


def entry_price_long(open_price: float, spread: float, slippage: float) -> float:
    """LONG enters at ASK = open + spread + slippage."""
    return open_price + spread + slippage


def entry_price_short(open_price: float, slippage: float) -> float:
    """SHORT enters at BID = open - slippage."""
    return open_price - slippage


def exit_price_long(bid: float, slippage: float = 0.0) -> float:
    """LONG exits at BID."""
    return bid - slippage


def exit_price_short(ask: float, slippage: float = 0.0) -> float:
    """SHORT exits at ASK."""
    return ask + slippage


def commission_price_units(cfg: Config, lots: float) -> float:
    """Total commission in price units for a round-trip trade of `lots`."""
    return cfg.costs.commission_per_lot_round_trip / cfg.symbol.contract_size * lots


def net_pnl(
    side: str,
    entry: float,
    exit_p: float,
    lots: float,
    cfg: Config,
) -> float:
    """
    Net P/L in account currency (USD).
    PnL = direction*(exit - entry)*lots*contract_size - commission*lots
    """
    direction = 1.0 if side == "BUY" else -1.0
    gross = direction * (exit_p - entry) * lots * cfg.symbol.contract_size
    comm = cfg.costs.commission_per_lot_round_trip * lots
    return gross - comm
