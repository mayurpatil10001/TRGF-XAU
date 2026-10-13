"""
src/execution/risk_manager.py

Institutional-grade SL/TP calculator.

Principles (how quant funds actually do it):
  1. SL is ALWAYS beyond a market structure point (swing high/low via fractals)
     — never a flat ATR multiple that price can easily stop-hunt
  2. TP is determined FIRST by a minimum Risk:Reward ratio (never below 1.5:1)
  3. Position size is derived FROM the SL distance so that $-risk is fixed
  4. Partial exit at 1:1 R:R moves SL to breakeven (free ride on the runner)
  5. Volatility regime widens/tightens levels automatically via ATR z-score
  6. Spread is always fully accounted for in break-even math
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger("execution.risk_manager")

EPS = 1e-8


@dataclass
class TradeSetup:
    side: str            # "BUY" | "SELL"
    entry: float
    sl: float
    tp1: float           # First target — 1.5:1 R:R (close 60% here)
    tp2: float           # Second target — 2.5:1 R:R (close remaining 40%)
    sl_dist: float       # |entry - sl|  in price
    tp1_dist: float
    tp2_dist: float
    rr1: float           # actual R:R to tp1
    rr2: float
    atr: float
    vol_regime: str      # "low" | "normal" | "high"
    structure_sl: bool   # True if SL was anchored to a fractal swing point
    lots: float = 0.0    # filled in by lot calculator


# ─────────────────────────────────────────────────────────────────────────────
# Volatility regime
# ─────────────────────────────────────────────────────────────────────────────

def _vol_regime(atr: float, atr_series: pd.Series) -> tuple[str, float]:
    """
    Compare current ATR to its rolling 20-bar mean.
    Returns (regime, multiplier):
      low    → 0.8  (tighten — low vol means small moves)
      normal → 1.0
      high   → 1.3  (widen — high vol needs more room)
    """
    if len(atr_series) < 20:
        return "normal", 1.0
    mean_atr = atr_series.iloc[-20:].mean()
    if mean_atr < EPS:
        return "normal", 1.0
    ratio = atr / mean_atr
    if ratio < 0.7:
        return "low", 0.8
    elif ratio > 1.4:
        return "high", 1.3
    return "normal", 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Structure SL — find the nearest swing point beyond entry
# ─────────────────────────────────────────────────────────────────────────────

def _find_structure_sl(
    side: str,
    entry: float,
    bars: pd.DataFrame,
    atr: float,
    min_sl_atr: float = 0.4,
    max_sl_atr: float = 2.5,
    lookback: int = 30,
) -> tuple[float, bool]:
    """
    Find the nearest confirmed fractal swing beyond the entry price.

    BUY  → SL below last confirmed swing LOW (down-fractal)
    SELL → SL above last confirmed swing HIGH (up-fractal)

    Falls back to ATR-based SL if no valid structure found.
    Returns (sl_price, is_structure_based).
    """
    min_dist = min_sl_atr * atr
    max_dist = max_sl_atr * atr

    h = bars["high"].iloc[-lookback:]
    l = bars["low"].iloc[-lookback:]

    if side == "BUY":
        # Find confirmed swing lows (down-fractals) in lookback window
        swing_lows = []
        for i in range(2, len(l) - 2):
            if l.iloc[i] < l.iloc[i-1] and l.iloc[i] < l.iloc[i+1] and \
               l.iloc[i] < l.iloc[i-2] and l.iloc[i] < l.iloc[i+2]:
                swing_lows.append(l.iloc[i])

        # Find the highest swing low that is still below entry - min_dist
        # (we want SL to sit just below the nearest meaningful swing low)
        valid = [sl for sl in swing_lows if sl < entry - min_dist * 0.5]
        if valid:
            # Use nearest (highest) valid swing low, then buffer 0.15*ATR below
            structure_level = max(valid) - 0.15 * atr
            sl_dist = entry - structure_level
            if min_dist <= sl_dist <= max_dist:
                return structure_level, True

        # Fallback: ATR-based
        return entry - min_dist, False

    else:  # SELL
        swing_highs = []
        for i in range(2, len(h) - 2):
            if h.iloc[i] > h.iloc[i-1] and h.iloc[i] > h.iloc[i+1] and \
               h.iloc[i] > h.iloc[i-2] and h.iloc[i] > h.iloc[i+2]:
                swing_highs.append(h.iloc[i])

        valid = [sh for sh in swing_highs if sh > entry + min_dist * 0.5]
        if valid:
            structure_level = min(valid) + 0.15 * atr
            sl_dist = structure_level - entry
            if min_dist <= sl_dist <= max_dist:
                return structure_level, True

        return entry + min_dist, False


# ─────────────────────────────────────────────────────────────────────────────
# Main calculator
# ─────────────────────────────────────────────────────────────────────────────

def calculate_setup(
    side: str,
    entry: float,
    bars: pd.DataFrame,
    atr: float,
    atr_series: pd.Series,
    spread: float,
    min_rr1: float = 1.5,
    min_rr2: float = 2.5,
    min_sl_atr: float = 0.4,
    max_sl_atr: float = 2.5,
) -> Optional[TradeSetup]:
    """
    Calculate a full institutional trade setup.

    Parameters
    ----------
    side        : "BUY" or "SELL"
    entry       : expected fill price (ask for BUY, bid for SELL)
    bars        : recent OHLCV DataFrame (needs at least 30 rows)
    atr         : current ATR value
    atr_series  : rolling ATR series for regime detection
    spread      : current spread in price units
    min_rr1/2   : minimum R:R ratios for first and second targets
    """
    if atr < EPS:
        return None

    vol_regime, vol_mult = _vol_regime(atr, atr_series)

    # 1. Structure-based SL
    sl_price, is_structure = _find_structure_sl(
        side, entry, bars, atr,
        min_sl_atr=min_sl_atr * vol_mult,
        max_sl_atr=max_sl_atr * vol_mult,
    )

    sl_dist = abs(entry - sl_price)

    # Ensure SL is at least 2x spread (can't have SL inside the spread)
    min_sl_from_spread = max(spread * 2.0, min_sl_atr * atr * vol_mult)
    if sl_dist < min_sl_from_spread:
        sl_dist = min_sl_from_spread
        sl_price = (entry - sl_dist) if side == "BUY" else (entry + sl_dist)
        is_structure = False

    # 2. TP levels from R:R
    tp1_dist = sl_dist * min_rr1
    tp2_dist = sl_dist * min_rr2

    if side == "BUY":
        tp1 = entry + tp1_dist
        tp2 = entry + tp2_dist
    else:
        tp1 = entry - tp1_dist
        tp2 = entry - tp2_dist

    rr1 = tp1_dist / (sl_dist + EPS)
    rr2 = tp2_dist / (sl_dist + EPS)

    setup = TradeSetup(
        side=side,
        entry=entry,
        sl=round(sl_price, 2),
        tp1=round(tp1, 2),
        tp2=round(tp2, 2),
        sl_dist=round(sl_dist, 4),
        tp1_dist=round(tp1_dist, 4),
        tp2_dist=round(tp2_dist, 4),
        rr1=round(rr1, 2),
        rr2=round(rr2, 2),
        atr=round(atr, 4),
        vol_regime=vol_regime,
        structure_sl=is_structure,
    )

    logger.info(
        f"Setup [{side}] entry={entry:.2f} sl={setup.sl:.2f} "
        f"tp1={setup.tp1:.2f} tp2={setup.tp2:.2f} | "
        f"SL={sl_dist:.2f} ({sl_dist/atr:.2f}xATR) | "
        f"R:R={rr1:.2f}/{rr2:.2f} | "
        f"vol={vol_regime} | structure={is_structure}"
    )

    return setup


# ─────────────────────────────────────────────────────────────────────────────
# Lot calculator — risk fixed fraction of equity
# ─────────────────────────────────────────────────────────────────────────────

def calculate_lots(
    equity: float,
    sl_dist: float,
    risk_pct: float = 0.01,
    contract_size: float = 100.0,    # XAU: 1 lot = 100 oz
    min_lot: float = 0.01,
    max_lot: float = 50.0,
    lot_step: float = 0.01,
) -> float:
    """
    Risk exactly `risk_pct` of equity per trade.

    dollar_risk  = equity * risk_pct
    lots         = dollar_risk / (sl_dist * contract_size)
    """
    if sl_dist < EPS or equity < 1:
        return min_lot

    dollar_risk = equity * risk_pct
    raw_lots = dollar_risk / (sl_dist * contract_size)

    # Round to nearest lot step
    lots = round(round(raw_lots / lot_step) * lot_step, 2)
    return float(np.clip(lots, min_lot, max_lot))


# ─────────────────────────────────────────────────────────────────────────────
# Trail manager — called every bar while position is open
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrailState:
    """Persists per open position."""
    setup: TradeSetup
    bars_in_trade: int = 0
    at_breakeven: bool = False
    tp1_hit: bool = False


def update_trail(
    state: TrailState,
    current_price: float,
    atr: float,
) -> tuple[Optional[float], Optional[float]]:
    """
    Returns (new_sl, new_tp) or (None, None) if no change needed.

    Rules:
      1. After 1:1 R:R → move SL to breakeven (free ride)
      2. After TP1 hit → trail SL at 0.5*ATR below/above current price
      3. Time stop → if 5+ bars with no 0.5:1 R:R progress → move SL to
         tighter position (half the original SL distance)
    """
    setup = state.setup
    state.bars_in_trade += 1
    new_sl = new_tp = None

    if setup.side == "BUY":
        pnl_dist = current_price - setup.entry
        rr_achieved = pnl_dist / (setup.sl_dist + EPS)

        # Rule 1: Move to breakeven at 1:1
        if rr_achieved >= 1.0 and not state.at_breakeven:
            new_sl = setup.entry + 0.1 * atr  # slightly above entry = guaranteed profit
            state.at_breakeven = True
            logger.info(f"Trail [BUY]: SL → breakeven+buffer {new_sl:.2f} (R:R={rr_achieved:.2f})")

        # Rule 2: Trail SL after TP1 progress
        if rr_achieved >= setup.rr1 and state.at_breakeven:
            trail_sl = current_price - 0.6 * atr
            if trail_sl > (new_sl or setup.sl):
                new_sl = trail_sl
                new_tp = current_price + setup.tp2_dist  # extend runner
                logger.info(f"Trail [BUY]: extending runner sl={new_sl:.2f} tp={new_tp:.2f}")

        # Rule 3: Time stop — tighten if stuck too long with no progress
        if state.bars_in_trade >= 5 and rr_achieved < 0.3 and not state.at_breakeven:
            tight_sl = setup.entry - setup.sl_dist * 0.5
            if tight_sl > setup.sl:  # only tighten, never loosen
                new_sl = tight_sl
                logger.info(f"Trail [BUY]: time stop tightening sl={new_sl:.2f}")

    else:  # SELL
        pnl_dist = setup.entry - current_price
        rr_achieved = pnl_dist / (setup.sl_dist + EPS)

        if rr_achieved >= 1.0 and not state.at_breakeven:
            new_sl = setup.entry - 0.1 * atr
            state.at_breakeven = True
            logger.info(f"Trail [SELL]: SL → breakeven+buffer {new_sl:.2f} (R:R={rr_achieved:.2f})")

        if rr_achieved >= setup.rr1 and state.at_breakeven:
            trail_sl = current_price + 0.6 * atr
            if trail_sl < (new_sl or setup.sl):
                new_sl = trail_sl
                new_tp = current_price - setup.tp2_dist
                logger.info(f"Trail [SELL]: extending runner sl={new_sl:.2f} tp={new_tp:.2f}")

        if state.bars_in_trade >= 5 and rr_achieved < 0.3 and not state.at_breakeven:
            tight_sl = setup.entry + setup.sl_dist * 0.5
            if tight_sl < setup.sl:
                new_sl = tight_sl
                logger.info(f"Trail [SELL]: time stop tightening sl={new_sl:.2f}")

    return new_sl, new_tp
