"""
src/backtest/engine.py
Event-driven, bar-by-bar backtester.

Key rules (R3):
- LONG enters at ASK (open + spread + slippage), exits at BID
- SHORT enters at BID (open - slippage), exits at ASK
- Costs always on; spread stress reruns
- SL first when both TP and SL hit in the same bar

Shares decision.py with the live bot.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, Config
from src.common.logging import setup_logging, get_logger
from src.backtest.costs import (
    entry_price_long, entry_price_short, commission_price_units, net_pnl
)
from src.backtest.sizing import compute_lots
from src.strategy.decision import make_decision, Decision
from src.evaluation.metrics import (
    profit_factor, sharpe_ratio, sortino_ratio, max_drawdown, expectancy_per_trade
)

logger = get_logger("backtest.engine")


@dataclass
class Position:
    bar_idx: int
    side: str              # "BUY" | "SELL"
    lots: float
    entry_price: float
    tp_price: float
    sl_price: float
    horizon_bars: int
    p_long: float
    p_short: float
    p_none: float
    entry_time: pd.Timestamp


@dataclass
class TradeLog:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    lots: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    cost_usd: float
    net_pnl: float
    R: float
    bars_held: int
    exit_reason: str
    p_long: float
    p_short: float
    p_none: float
    fold_id: int = -1
    session: str = ""
    spread_used: float = 0.0


def run_backtest(
    df: pd.DataFrame,           # cleaned bars with features and labels
    probs: np.ndarray,          # [N, 3] model probabilities
    cfg: Config,
    spread_mult: float = 1.0,   # spread stress multiplier
    horizon_bars: int = 3,
    k: float = 0.75,
    m: float = 0.75,
    prob_threshold: float = 0.55,
    initial_equity: float = 10_000.0,
    fold_id: int = -1,
    feature_atr_col: str = "atr",  # column name for ATR in df
) -> Tuple[List[TradeLog], np.ndarray]:
    """
    Run a single backtest pass and return (trade_log, equity_curve).
    """
    from src.labels.triple_barrier import cost_round_trip

    n = len(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    times = df["time"].values
    sessions = df["session"].values if "session" in df.columns else np.array(["unknown"] * n)
    gaps = df["gap_before_minutes"].values if "gap_before_minutes" in df.columns else np.zeros(n)

    if feature_atr_col in df.columns:
        atrs = df[feature_atr_col].values
    else:
        # Compute ATR inline
        from src.features.feature_pipeline import wilder_atr
        atrs = wilder_atr(df, cfg.features.atr_period).values

    spreads = df["spread"].values * spread_mult if "spread" in df.columns else (
        np.full(n, cfg.costs.spread_price_units * spread_mult)
    )
    slip = cfg.costs.slippage_price_units

    equity = initial_equity
    equity_curve = np.full(n, initial_equity)
    trade_logs: List[TradeLog] = []
    position: Optional[Position] = None

    # Risk state
    daily_loss_start_equity = initial_equity
    current_day = pd.Timestamp(times[0]).date() if len(times) > 0 else None
    consecutive_losses = 0
    cooldown_bars = 0

    # Adjust thresholds
    cost_rt = cost_round_trip(cfg)

    # Override threshold
    _prob_threshold = prob_threshold

    for i in range(n - 1):  # -1 because entry is at bar i+1
        t = pd.Timestamp(times[i])

        # ── Day reset ────────────────────────────────────────────────────────
        day = t.date()
        if day != current_day:
            current_day = day
            daily_loss_start_equity = equity

        daily_loss_pct = max(0.0, (daily_loss_start_equity - equity) / daily_loss_start_equity * 100)

        # ── Manage open position ──────────────────────────────────────────────
        if position is not None:
            bars_held = i - position.bar_idx
            h, l = highs[i], lows[i]
            sp = spreads[i]

            exit_price = None
            exit_reason = None

            if position.side == "BUY":
                tp_hit = h >= position.tp_price
                sl_hit = l <= position.sl_price
            else:
                tp_hit = (l + sp) <= position.tp_price
                sl_hit = (h + sp) >= position.sl_price

            if tp_hit and sl_hit:
                exit_price = position.sl_price
                exit_reason = "SL_first"
            elif tp_hit:
                exit_price = position.tp_price
                exit_reason = "TP"
            elif sl_hit:
                exit_price = position.sl_price
                exit_reason = "SL"
            elif bars_held >= horizon_bars:
                exit_price = closes[i]
                if position.side == "SELL":
                    exit_price += sp
                exit_reason = "time_exit"

            if exit_price is not None:
                if position.side == "SELL":
                    # Short exits at ASK for TP/SL, BID + spread for time exit
                    pass  # already adjusted above

                gross = ((1 if position.side == "BUY" else -1)
                         * (exit_price - position.entry_price)
                         * position.lots * cfg.symbol.contract_size)
                comm = cfg.costs.commission_per_lot_round_trip * position.lots
                net = gross - comm
                sl_dist = abs(position.entry_price - position.sl_price)
                R = net / (sl_dist * position.lots * cfg.symbol.contract_size + 1e-9)

                equity += net
                if net < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                cooldown_bars = cfg.risk.cooldown_bars_after_exit

                trade_logs.append(TradeLog(
                    entry_time=position.entry_time,
                    exit_time=t,
                    side=position.side,
                    lots=position.lots,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    gross_pnl=gross,
                    cost_usd=comm,
                    net_pnl=net,
                    R=R,
                    bars_held=bars_held,
                    exit_reason=exit_reason,
                    p_long=position.p_long,
                    p_short=position.p_short,
                    p_none=position.p_none,
                    fold_id=fold_id,
                    session=str(sessions[i]),
                    spread_used=float(spreads[i]),
                ))
                position = None

        if cooldown_bars > 0:
            cooldown_bars -= 1

        equity_curve[i] = equity

        # ── New entry decision (using bar i's signal, entry at bar i+1) ──────
        if position is None and i + 1 < n:
            atr_i = float(atrs[i]) if not np.isnan(atrs[i]) else 0.0
            prob_i = probs[i]

            decision = make_decision(
                probs=prob_i,
                bar_time=t,
                spread=float(spreads[i]),
                atr=atr_i,
                cfg=cfg,
                has_open_position=False,
                cooldown_bars_remaining=cooldown_bars,
                daily_loss_pct=daily_loss_pct,
                consecutive_losses=consecutive_losses,
            )

            # Override threshold from backtest params
            max_prob = max(decision.p_long, decision.p_short)
            if max_prob < _prob_threshold and decision.action != "NO_TRADE":
                decision.action = "NO_TRADE"
                decision.reason = f"prob_below_backtest_threshold"

            if decision.action in ("BUY", "SELL"):
                # Entry at next bar's open
                sp_next = spreads[i + 1]
                open_next = opens[i + 1]

                if gaps[i + 1] > cfg.data.max_gap_minutes:
                    continue  # skip gapped bar

                if decision.action == "BUY":
                    entry_p = entry_price_long(open_next, sp_next, slip)
                    tp_dist = cost_rt + k * atr_i
                    sl_dist_p = m * atr_i
                    tp_price = entry_p + tp_dist
                    sl_price = entry_p - sl_dist_p
                else:
                    entry_p = entry_price_short(open_next, slip)
                    tp_dist = cost_rt + k * atr_i
                    sl_dist_p = m * atr_i
                    tp_price = entry_p - tp_dist
                    sl_price = entry_p + sl_dist_p

                lots = compute_lots(equity, sl_dist_p, cfg)
                if lots <= 0:
                    continue

                position = Position(
                    bar_idx=i + 1,
                    side=decision.action,
                    lots=lots,
                    entry_price=entry_p,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    horizon_bars=horizon_bars,
                    p_long=decision.p_long,
                    p_short=decision.p_short,
                    p_none=decision.p_none,
                    entry_time=pd.Timestamp(times[i + 1]),
                )

    equity_curve[-1] = equity
    return trade_logs, equity_curve


def trades_to_df(trade_logs: List[TradeLog]) -> pd.DataFrame:
    if not trade_logs:
        return pd.DataFrame()
    return pd.DataFrame([vars(t) for t in trade_logs])


def backtest_stats(trade_logs: List[TradeLog], equity_curve: np.ndarray) -> Dict[str, Any]:
    """Compute all standard backtest statistics."""
    if not trade_logs:
        return {"n_trades": 0, "error": "no_trades"}

    pnl = np.array([t.net_pnl for t in trade_logs])
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    R_vals = np.array([t.R for t in trade_logs])

    n = len(pnl)
    win_rate = float((pnl > 0).mean())
    rr = float(wins.mean() / abs(losses.mean())) if len(wins) > 0 and len(losses) > 0 else 0.0
    p_star = 1.0 / (1.0 + rr) if rr > 0 else 0.5

    dd = max_drawdown(equity_curve)
    bars_held = [t.bars_held for t in trade_logs]

    # Longest losing streak
    streak = cur_streak = 0
    for p in pnl:
        cur_streak = cur_streak + 1 if p < 0 else 0
        streak = max(streak, cur_streak)

    return {
        "n_trades": n,
        "win_rate": round(win_rate, 4),
        "breakeven_win_rate": round(p_star, 4),
        "avg_win_usd": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss_usd": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "expectancy_usd": round(float(pnl.mean()), 4),
        "expectancy_R": round(float(R_vals.mean()), 4),
        "profit_factor": round(profit_factor(pnl, pnl), 4),
        "total_net_pnl": round(float(pnl.sum()), 2),
        "sharpe": round(sharpe_ratio(pnl), 4),
        "sortino": round(sortino_ratio(pnl), 4),
        "max_drawdown_pct": dd["max_drawdown_pct"],
        "max_dd_duration_bars": dd["max_dd_duration_bars"],
        "avg_bars_held": round(float(np.mean(bars_held)), 2),
        "longest_losing_streak": streak,
        "exit_reasons": pd.Series([t.exit_reason for t in trade_logs]).value_counts().to_dict(),
    }


def run_spread_stress(
    df: pd.DataFrame,
    probs: np.ndarray,
    cfg: Config,
    stress_mults: List[float] = None,
    **kwargs,
) -> Dict[float, Dict]:
    """Run backtest at multiple spread multipliers and return stats table."""
    if stress_mults is None:
        stress_mults = cfg.costs.spread_stress
    results = {}
    for mult in stress_mults:
        logs, curve = run_backtest(df, probs, cfg, spread_mult=mult, **kwargs)
        stats = backtest_stats(logs, curve)
        stats["spread_mult"] = mult
        results[mult] = stats
        logger.info(f"Spread {mult:.1f}x: trades={stats['n_trades']}, "
                    f"PF={stats.get('profit_factor', 'N/A')}, "
                    f"net={stats.get('total_net_pnl', 'N/A')}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    logger.info("Backtest engine loaded. Run via run_all.py or walk_forward.py.")


if __name__ == "__main__":
    main()
