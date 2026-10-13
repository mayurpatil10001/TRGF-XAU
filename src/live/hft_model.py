"""
src/live/hft_model.py  (v2 — Pullback Exhaustion Scalper)

HFT Signal Logic:
═══════════════════════════════════════════════════════════════
In a H1=BEAR trend:
  - M5=UP (pullback) is the ENTRY OPPORTUNITY, not a blocker
  - Sell when the M5 bounce exhausts: RSI peaks, momentum decelerates
  - SL above recent swing = 1.5pt, TP = 3pt continuation

In a H1=BULL trend:
  - M5=DOWN (pullback) is the entry opportunity
  - Buy when dip exhausts: RSI bottoms, momentum turns up

Signal fires on ANY of these conditions (not requiring all three):
  1. Micro RSI exhaustion  (RSI>65 SELL / RSI<35 BUY)
  2. Momentum deceleration (mom5 turning while accel<0)
  3. Spread normal (not widening = no news spike)

Score-based confidence — minimum 0.45 to trade.
═══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.live.tick_engine import TickFeatures


@dataclass
class HFTSignal:
    action: str       # "BUY" | "SELL" | "FLAT"
    confidence: float # 0.0 - 1.0
    reason: str


def generate_signal(
    features: Optional[TickFeatures],
    h1_trend: str,
    h1_adx: float,
    max_spread_pts: float = 0.60,
    current_spread: float = 0.25,
) -> HFTSignal:
    """Pullback-exhaustion scalping signal."""

    if features is None:
        return HFTSignal("FLAT", 0.0, "warming_up")

    if h1_trend == "NEUTRAL":
        return HFTSignal("FLAT", 0.0, "h1_neutral")

    # Block only on extreme spread (news spike)
    if current_spread > max_spread_pts:
        return HFTSignal("FLAT", 0.0, f"wide_spread:{current_spread:.3f}")

    # Block extreme volatility burst (spike candle)
    if features.vol_burst:
        return HFTSignal("FLAT", 0.0, "vol_burst")

    f = features
    score = 0.0
    reasons = []

    # ── ADX strength always contributes ──────────────────────────────────────
    adx_bonus = min((h1_adx - 20.0) / 40.0, 1.0) * 0.20
    score += adx_bonus

    if h1_trend == "BEAR":
        # ── SELL: pullback exhaustion ─────────────────────────────────────────
        # Condition 1: RSI overbought on micro frame (bounce peaked)
        if f.micro_rsi > 60:
            c = min((f.micro_rsi - 60.0) / 20.0, 1.0) * 0.40
            score += c
            reasons.append(f"rsi={f.micro_rsi:.0f}")

        # Condition 2: momentum decelerating (bounce losing steam)
        if f.accel < 0:
            c = min(abs(f.accel) / 0.03, 1.0) * 0.30
            score += c
            reasons.append(f"accel={f.accel:.4f}")

        # Condition 3: short-term momentum still positive (still in bounce)
        # — this confirms we're at the peak, not mid-crash
        if f.momentum_5 > -0.01:  # flat or slightly up
            score += 0.15
            reasons.append("mom_flat_at_peak")

        if score >= 0.45:
            return HFTSignal("SELL", min(score, 1.0),
                             "BEAR_pullback_peak|" + "|".join(reasons))

    elif h1_trend == "BULL":
        # ── BUY: dip exhaustion ───────────────────────────────────────────────
        if f.micro_rsi < 40:
            c = min((40.0 - f.micro_rsi) / 20.0, 1.0) * 0.40
            score += c
            reasons.append(f"rsi={f.micro_rsi:.0f}")

        if f.accel > 0:
            c = min(f.accel / 0.03, 1.0) * 0.30
            score += c
            reasons.append(f"accel={f.accel:.4f}")

        if f.momentum_5 < 0.01:
            score += 0.15
            reasons.append("mom_flat_at_dip")

        if score >= 0.45:
            return HFTSignal("BUY", min(score, 1.0),
                             "BULL_dip_exhaust|" + "|".join(reasons))

    return HFTSignal("FLAT", score,
                     f"no_signal score={score:.2f} rsi={f.micro_rsi:.1f}")
