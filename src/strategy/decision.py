"""
src/strategy/decision.py
Shared decision logic used by BOTH the backtester and the live bot.
This is the single source of truth for entry decisions.

Filters in order:
1. Data valid (no NaN, spread within limit)
2. Session allowed
3. Not in blackout window
4. Spread <= max_spread
5. No open position
6. Cooldown satisfied
7. Daily loss limit not hit
8. Consecutive-loss pause not active
9. Probability >= threshold
10. Expected net R >= min_expected_R
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.common.config import Config
from src.common.timeutils import is_in_blackout, get_session


@dataclass
class Decision:
    action: str          # "BUY", "SELL", "NO_TRADE"
    confidence: float    # max probability
    p_long: float
    p_short: float
    p_none: float
    tp_dist: float
    sl_dist: float
    reason: str          # reason for skipping, if NO_TRADE


def make_decision(
    probs: np.ndarray,        # [3] float: p_none, p_long, p_short
    bar_time: pd.Timestamp,
    spread: float,
    atr: float,
    cfg: Config,
    # Risk state (passed from live bot or backtester)
    has_open_position: bool = False,
    cooldown_bars_remaining: int = 0,
    daily_loss_pct: float = 0.0,
    consecutive_losses: int = 0,
    # Optional expected-R (from the model's exp_r head)
    expected_R: float = 0.0,
) -> Decision:
    """
    Return a Decision given model probabilities and current risk state.
    """
    p_none, p_long, p_short = float(probs[0]), float(probs[1]), float(probs[2])

    # Pick best direction
    if p_long >= p_short:
        best_dir = "BUY"
        best_prob = p_long
    else:
        best_dir = "SELL"
        best_prob = p_short

    def skip(reason: str) -> Decision:
        return Decision(
            action="NO_TRADE", confidence=best_prob,
            p_long=p_long, p_short=p_short, p_none=p_none,
            tp_dist=0.0, sl_dist=0.0, reason=reason,
        )

    # ── Filter 1: data sanity ────────────────────────────────────────────────
    if np.isnan(probs).any() or atr <= 0 or spread < 0:
        return skip("invalid_data")

    # ── Filter 2: session ────────────────────────────────────────────────────
    session_map = {k: (v[0], v[1]) for k, v in cfg.sessions.model_dump().items()}
    current_session = get_session(bar_time, session_map)
    if current_session not in cfg.decision.sessions_allowed:
        return skip(f"session_not_allowed:{current_session}")

    # ── Filter 3: blackout ───────────────────────────────────────────────────
    if is_in_blackout(bar_time, cfg.decision.blackout_utc):
        return skip("blackout_window")

    # ── Filter 4: spread ─────────────────────────────────────────────────────
    if spread > cfg.decision.max_spread_price_units:
        return skip(f"spread_too_high:{spread:.3f}")

    # ── Filter 5: open position ──────────────────────────────────────────────
    if has_open_position:
        return skip("open_position_exists")

    # ── Filter 6: cooldown ───────────────────────────────────────────────────
    if cooldown_bars_remaining > 0:
        return skip(f"cooldown:{cooldown_bars_remaining}")

    # ── Filter 7: daily loss limit ───────────────────────────────────────────
    if daily_loss_pct >= cfg.risk.daily_loss_limit_pct:
        return skip(f"daily_loss_limit:{daily_loss_pct:.2f}%")

    # ── Filter 8: consecutive losses ─────────────────────────────────────────
    if consecutive_losses >= cfg.risk.max_consecutive_losses:
        return skip(f"consecutive_losses:{consecutive_losses}")

    # ── Filter 9: probability threshold ─────────────────────────────────────
    # Use the primary threshold from config (or the best validated one)
    threshold = cfg.decision.prob_threshold_grid[0]  # default; walk-forward may override
    if best_prob < threshold:
        return skip(f"prob_below_threshold:{best_prob:.3f}<{threshold:.3f}")

    # ── Filter 10: expected R ────────────────────────────────────────────────
    if expected_R < cfg.decision.min_expected_R:
        return skip(f"expected_R_too_low:{expected_R:.3f}")

    # ── TP / SL distances ────────────────────────────────────────────────────
    from src.labels.triple_barrier import cost_round_trip
    cost_rt = cost_round_trip(cfg)
    k = 0.75  # default; overridden per fold in walk-forward
    m = 0.75
    tp_dist = cost_rt + k * atr
    sl_dist = m * atr

    return Decision(
        action=best_dir,
        confidence=best_prob,
        p_long=p_long,
        p_short=p_short,
        p_none=p_none,
        tp_dist=tp_dist,
        sl_dist=sl_dist,
        reason="OK",
    )
