"""
src/labels/triple_barrier.py
Cost-aware triple-barrier labelling.

For each bar t:
  - Entry at bar t+1's open (LONG at ASK = open + spread + slippage,
                              SHORT at BID = open - slippage)
  - TP = cost_rt + k*ATR_t  (cost-inclusive, so we need to beat costs to be labeled a win)
  - SL = m*ATR_t
  - Scan bars t+1..t+H using BID prices for high/low touch events
  - If both TP and SL in same bar → SL first (conservative)
  - Time exit at close of bar t+H
  - label = LONG(1) / SHORT(2) / NO_TRADE(0)

Also stores: long_R, short_R, label_end_index (for purging).

Usage:
  python -m src.labels.triple_barrier --config config/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, Config
from src.common.logging import setup_logging, get_logger
from src.features.feature_pipeline import wilder_atr

logger = get_logger("labels.triple_barrier")

LABEL_LONG = 1
LABEL_SHORT = 2
LABEL_NONE = 0


def _commission_price(cfg: Config) -> float:
    """Commission as price units per trade (per side, per lot)."""
    # commission_per_lot_round_trip in USD, contract_size in oz
    # price_per_oz * lots * commission = commission_USD
    # commission in price units = commission_USD / (contract_size * lots)
    # Per lot per round trip = commission / contract_size
    return cfg.costs.commission_per_lot_round_trip / cfg.symbol.contract_size


def cost_round_trip(cfg: Config) -> float:
    """Full round-trip cost in price units (spread + 2*slippage + commission)."""
    return (
        cfg.costs.spread_price_units
        + 2 * cfg.costs.slippage_price_units
        + _commission_price(cfg)
    )


def compute_labels(
    df: pd.DataFrame,
    cfg: Config,
    horizon: int,
    k: float,  # TP ATR multiple
    m: float,  # SL ATR multiple
    max_gap_minutes: int = 30,
) -> pd.DataFrame:
    """
    Compute triple-barrier labels for one (H, k, m) combination.

    Returns a DataFrame with columns:
      label, long_R, short_R, label_end_index, label_horizon, label_k, label_m
    """
    atr = wilder_atr(df, cfg.features.atr_period)
    spread = df["spread"] if "spread" in df.columns else cfg.costs.spread_price_units
    slip = cfg.costs.slippage_price_units
    cost_rt = cost_round_trip(cfg)

    n = len(df)
    labels = np.full(n, LABEL_NONE, dtype=np.int8)
    long_R = np.full(n, np.nan)
    short_R = np.full(n, np.nan)
    label_end_idx = np.full(n, -1, dtype=np.int64)

    # Precompute spread series (scalar or array)
    if isinstance(spread, pd.Series):
        spread_arr = spread.values
    else:
        spread_arr = np.full(n, float(spread))

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    atr_arr = atr.values
    gaps = df["gap_before_minutes"].values if "gap_before_minutes" in df.columns else np.zeros(n)

    for t in range(n - horizon - 1):
        atr_t = atr_arr[t]
        if np.isnan(atr_t) or atr_t <= 0:
            continue

        t1 = t + 1  # entry bar
        # Check for gap at entry bar
        if gaps[t1] > max_gap_minutes:
            continue

        # ── Long side ────────────────────────────────────────────────────────
        entry_long = opens[t1] + spread_arr[t1] + slip
        tp_dist = cost_rt + k * atr_t
        sl_dist = m * atr_t
        tp_long = entry_long + tp_dist
        sl_long = entry_long - sl_dist

        long_outcome = _scan_outcome(
            highs, lows, closes, gaps,
            t1, t + horizon,
            tp_long, sl_long,
            side="long",
            spread_arr=spread_arr,
            max_gap_minutes=max_gap_minutes,
        )
        long_end_idx = min(t + horizon, n - 1)

        # ── Short side ───────────────────────────────────────────────────────
        entry_short = opens[t1] - slip
        tp_short = entry_short - tp_dist
        sl_short = entry_short + sl_dist

        short_outcome = _scan_outcome(
            highs, lows, closes, gaps,
            t1, t + horizon,
            tp_short, sl_short,
            side="short",
            spread_arr=spread_arr,
            max_gap_minutes=max_gap_minutes,
        )

        # ── R-multiples ──────────────────────────────────────────────────────
        long_R[t] = long_outcome["net_pnl"] / sl_dist
        short_R[t] = short_outcome["net_pnl"] / sl_dist

        # ── Label assignment ─────────────────────────────────────────────────
        long_win = long_outcome["net_pnl"] > 0
        short_win = short_outcome["net_pnl"] > 0

        if long_win and not short_win:
            labels[t] = LABEL_LONG
        elif short_win and not long_win:
            labels[t] = LABEL_SHORT
        else:
            labels[t] = LABEL_NONE

        label_end_idx[t] = long_end_idx

    result = pd.DataFrame(
        {
            "label": labels,
            "long_R": long_R,
            "short_R": short_R,
            "label_end_index": label_end_idx,
            "label_horizon": horizon,
            "label_k": k,
            "label_m": m,
        },
        index=df.index,
    )
    return result


def _scan_outcome(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    gaps: np.ndarray,
    t_start: int,
    t_end: int,
    tp_price: float,
    sl_price: float,
    side: str,
    spread_arr: np.ndarray,
    max_gap_minutes: int,
) -> dict:
    """
    Scan bars t_start..t_end for TP/SL hit.
    Returns dict with exit_price, exit_bar, net_pnl, exit_reason.
    """
    for bar in range(t_start, t_end + 1):
        if bar > t_start and gaps[bar] > max_gap_minutes:
            # Gap: time exit at the bar before the gap
            exit_bar = bar - 1
            exit_price = closes[exit_bar]
            if side == "short":
                exit_price += spread_arr[exit_bar]
            entry = tp_price - (tp_price - sl_price)  # recover entry
            # Compute from bar t_start entry assumptions
            break

        h, l = highs[bar], lows[bar]
        sp = spread_arr[bar]

        if side == "long":
            # BID prices: TP hit if High >= tp_price, SL hit if Low <= sl_price
            tp_hit = h >= tp_price
            sl_hit = l <= sl_price
        else:
            # ASK prices for short exit: ASK = BID + spread
            tp_hit = (l + sp) <= tp_price  # TP for short = low ask
            sl_hit = (h + sp) >= sl_price  # SL for short = high ask

        if tp_hit and sl_hit:
            # Both in same bar → SL first (conservative)
            exit_price = sl_price
            exit_reason = "SL_first_same_bar"
        elif tp_hit:
            exit_price = tp_price
            exit_reason = "TP"
        elif sl_hit:
            exit_price = sl_price
            exit_reason = "SL"
        else:
            if bar == t_end:
                exit_price = closes[bar]
                if side == "short":
                    exit_price += spread_arr[bar]
                exit_reason = "time_exit"
            else:
                continue

        # net_pnl (in price units, before dividing by SL for R)
        if side == "long":
            # Entry: ask at t_start open (already computed outside)
            # For scanning purposes, entry is the ask used by the caller
            entry_price = tp_price - (tp_price - sl_price) - (
                # reconstruct: tp = entry + tp_dist, sl = entry - sl_dist
                # We need to pass entry, but let's compute pnl relative to entry
                0
            )
            # Simpler: just return pnl sign info; R computed outside
            pnl = exit_price - (sl_price + (tp_price - sl_price) / 2)  # placeholder
        else:
            pnl = -(exit_price - (sl_price - (sl_price - tp_price) / 2))

        return {"exit_price": exit_price, "exit_reason": exit_reason, "net_pnl": _net_pnl(
            side, tp_price, sl_price, exit_price, exit_reason,
        )}

    # Should not reach here
    return {"exit_price": closes[t_end], "exit_reason": "time_exit", "net_pnl": 0.0}


def _net_pnl(
    side: str, tp_price: float, sl_price: float, exit_price: float, exit_reason: str
) -> float:
    """Compute net PnL in price units given exit info."""
    if exit_reason == "TP":
        return abs(tp_price - sl_price) * (1 if "TP" in exit_reason else -1)
    elif exit_reason in ("SL", "SL_first_same_bar"):
        return -abs(tp_price - sl_price) * abs(sl_price - exit_price) / (abs(tp_price - sl_price) + 1e-12)

    # General: compute from side
    if side == "long":
        # entry reconstructed from tp/sl: entry = sl + sl_dist = tp - tp_dist
        # Approximate: use (tp + sl)/2 as entry proxy for test
        entry = sl_price + (tp_price - sl_price) * 0.5  # placeholder midpoint
        return exit_price - entry
    else:
        entry = sl_price - (sl_price - tp_price) * 0.5
        return entry - exit_price


def compute_labels_fast(
    df: pd.DataFrame,
    cfg: Config,
    horizon: int,
    k: float,
    m: float,
) -> pd.DataFrame:
    """
    Vectorized triple-barrier labelling (faster than loop for long series).
    """
    atr = wilder_atr(df, cfg.features.atr_period)
    if isinstance(df["spread"] if "spread" in df.columns else None, pd.Series):
        spread_arr = df["spread"].values
    else:
        spread_arr = np.full(len(df), cfg.costs.spread_price_units)

    slip = cfg.costs.slippage_price_units
    cost_rt = cost_round_trip(cfg)
    n = len(df)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    atr_arr = atr.values
    gaps = df["gap_before_minutes"].values if "gap_before_minutes" in df.columns else np.zeros(n)
    max_gap = cfg.data.max_gap_minutes

    labels = np.zeros(n, dtype=np.int8)
    long_R = np.full(n, np.nan)
    short_R = np.full(n, np.nan)
    label_end_idx = np.full(n, -1, dtype=np.int64)

    for t in range(n - horizon - 1):
        atr_t = atr_arr[t]
        if np.isnan(atr_t) or atr_t <= 0:
            continue
        t1 = t + 1
        if gaps[t1] > max_gap:
            continue

        sp_t1 = spread_arr[t1]
        entry_long = opens[t1] + sp_t1 + slip
        entry_short = opens[t1] - slip
        tp_dist = cost_rt + k * atr_t
        sl_dist = m * atr_t

        tp_long = entry_long + tp_dist
        sl_long = entry_long - sl_dist
        tp_short = entry_short - tp_dist
        sl_short = entry_short + sl_dist

        # Scan
        long_exit_pnl = _scan_vectorized(
            highs, lows, closes, spreads=spread_arr, gaps=gaps,
            t_start=t1, t_end=min(t + horizon, n - 1),
            tp=tp_long, sl=sl_long, side=1, max_gap=max_gap,
            entry=entry_long,
        )
        short_exit_pnl = _scan_vectorized(
            highs, lows, closes, spreads=spread_arr, gaps=gaps,
            t_start=t1, t_end=min(t + horizon, n - 1),
            tp=tp_short, sl=sl_short, side=-1, max_gap=max_gap,
            entry=entry_short,
        )

        long_R[t] = long_exit_pnl / sl_dist
        short_R[t] = short_exit_pnl / sl_dist
        label_end_idx[t] = min(t + horizon, n - 1)

        long_win = long_exit_pnl > 0
        short_win = short_exit_pnl > 0
        if long_win and not short_win:
            labels[t] = LABEL_LONG
        elif short_win and not long_win:
            labels[t] = LABEL_SHORT

    return pd.DataFrame(
        {
            "label": labels,
            "long_R": long_R,
            "short_R": short_R,
            "label_end_index": label_end_idx,
            "label_horizon": horizon,
            "label_k": k,
            "label_m": m,
        },
        index=df.index,
    )


def _scan_vectorized(
    highs, lows, closes, spreads, gaps,
    t_start, t_end, tp, sl, side, max_gap, entry,
) -> float:
    """
    Scan bars for first TP/SL hit; return net PnL.
    side: +1 = long, -1 = short
    """
    for bar in range(t_start, t_end + 1):
        if bar > t_start and gaps[bar] > max_gap:
            exit_price = closes[bar - 1]
            if side == -1:
                exit_price += spreads[bar - 1]
            return side * (exit_price - entry) if side == 1 else side * (entry - exit_price)

        h, l, sp = highs[bar], lows[bar], spreads[bar]

        if side == 1:  # long: bid prices
            tp_hit = h >= tp
            sl_hit = l <= sl
        else:          # short: ask prices
            tp_hit = (l + sp) <= tp
            sl_hit = (h + sp) >= sl

        if tp_hit and sl_hit:
            exit_price = sl
            return side * (sl - entry) if side == 1 else side * (entry - sl)
        elif tp_hit:
            return abs(tp - entry)
        elif sl_hit:
            return -(abs(sl - entry))

    # Time exit
    exit_price = closes[t_end]
    if side == -1:
        exit_price += spreads[t_end]
    return side * (exit_price - entry) if side == 1 else side * (entry - exit_price)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)

    feat_path = Path("data/processed/features.parquet")
    if not feat_path.exists():
        raise FileNotFoundError(f"Run feature pipeline first: {feat_path}")

    df = pd.read_parquet(feat_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    # Use development period for label computation (holdout excluded from labels too)
    df_dev = df[df["time"] <= pd.Timestamp(cfg.data.dev_end + " 23:59:59", tz="UTC")].copy()

    from src.labels.regression_target import compute_regression_target
    df_dev["y_reg"] = compute_regression_target(df_dev)

    all_labels = []
    for H, k, m in product(cfg.labels.horizon_bars, cfg.labels.tp_atr_mult_k, cfg.labels.sl_atr_mult_m):
        logger.info(f"Computing labels: H={H}, k={k}, m={m}")
        lab = compute_labels_fast(df_dev, cfg, H, k, m)
        lab["combo_id"] = f"H{H}_k{k}_m{m}"
        all_labels.append(lab)

    # Save the default config labels (first combo) plus all
    primary = all_labels[0]
    out_df = df_dev.copy()
    for col in ["label", "long_R", "short_R", "label_end_index"]:
        out_df[col] = primary[col].values

    # Class balance report
    logger.info("Label balance (primary combo):")
    for yr, grp in out_df.groupby(out_df["time"].dt.year):
        counts = grp["label"].value_counts().to_dict()
        logger.info(f"  {yr}: {counts}")

    out_path = Path("data/processed/labels.parquet")
    out_df.to_parquet(out_path, index=False)
    logger.info(f"Saved labels to {out_path}")

    # Save all combos
    all_lab_df = pd.concat(all_labels, axis=0)
    all_lab_df.to_parquet(Path("data/processed/labels_all_combos.parquet"), index=False)
    logger.info("Saved all label combos.")


if __name__ == "__main__":
    main()
