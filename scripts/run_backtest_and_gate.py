"""
scripts/run_backtest_and_gate.py  (v3 — correct API calls)
Runs the event-driven backtester on OOS walk-forward predictions,
then evaluates the 8-criteria Profit Gate.

Usage: python scripts/run_backtest_and_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config, reset_config
from src.common.logging import setup_logging, get_logger
from src.evaluation.profit_gate import evaluate_gate, save_gate_result
from src.backtest.costs import entry_price_long, entry_price_short, net_pnl
from src.backtest.sizing import compute_lots
from src.strategy.decision import make_decision

reset_config()
cfg = load_config("config/config.yaml")
setup_logging(cfg.logging.level, cfg.logging.log_dir)
logger = get_logger("backtest_gate")

H = cfg.labels.horizon_bars[1]       # primary label horizon (bars)
ATR_MULT = 0.75                       # TP/SL multiplier in ATR units


def fuse_probs(preds: dict) -> np.ndarray:
    """Weighted average of available model probability arrays."""
    candidates = [("gated_fusion", 2.0), ("mlp", 1.0), ("lgbm", 1.0)]
    arrays, weights = [], []
    for name, w in candidates:
        arr = preds.get(name)
        if arr is not None:
            arrays.append(arr)
            weights.append(w)
    if not arrays:
        raise ValueError("No model predictions found in OOS file.")
    total_w = sum(weights)
    return sum(a * w for a, w in zip(arrays, weights)) / total_w


def main() -> None:
    # ── Load OOS predictions ──────────────────────────────────────────────────
    pred_path = Path("artifacts/oos_predictions.npy")
    if not pred_path.exists():
        logger.error("No OOS predictions found. Run src.run_all first.")
        sys.exit(1)

    preds = np.load(pred_path, allow_pickle=True).item()
    logger.info(f"OOS predictions loaded: {len(preds['times'])} bars")

    probs_fused = fuse_probs(preds)
    logger.info(f"Fused probs shape: {probs_fused.shape}  "
                f"max={probs_fused.max():.4f}  threshold[0]={cfg.decision.prob_threshold_grid[0]}")

    # ── Load OOS bars ─────────────────────────────────────────────────────────
    df_all = pd.read_parquet("data/processed/labels.parquet")
    df_all["time"] = pd.to_datetime(df_all["time"], utc=True)
    oos_times = pd.to_datetime(preds["times"], utc=True)
    df_oos = df_all[df_all["time"].isin(oos_times)].reset_index(drop=True)
    logger.info(f"OOS bars: {len(df_oos)}")

    # ── Event-driven bar-by-bar backtest ─────────────────────────────────────
    equity = 10_000.0
    trades = []
    in_trade = False
    trade_dir = None
    entry_bar = entry_px = entry_lots = sl_px = tp_px = None

    rows = df_oos.reset_index(drop=True)
    n = min(len(rows), len(probs_fused))

    for i in range(n):
        row = rows.iloc[i]
        prob = probs_fused[i]
        spread = float(row.get("spread", cfg.costs.spread_price_units))
        if pd.isna(spread) or spread <= 0:
            spread = cfg.costs.spread_price_units

        # ── Exit open trade ───────────────────────────────────────────────────
        if in_trade:
            hi, lo, op = float(row["high"]), float(row["low"]), float(row["open"])
            bars_held = i - entry_bar

            pnl = None
            exit_reason = None

            if trade_dir == "long":
                # SL first (R4)
                if lo <= sl_px:
                    pnl = net_pnl("BUY", entry_px, sl_px, entry_lots, cfg)
                    exit_reason = "sl"
                elif hi >= tp_px:
                    pnl = net_pnl("BUY", entry_px, tp_px, entry_lots, cfg)
                    exit_reason = "tp"
                elif bars_held >= H:
                    exit_p = op - spread / 2   # bid approximation
                    pnl = net_pnl("BUY", entry_px, exit_p, entry_lots, cfg)
                    exit_reason = "timeout"
            else:  # short
                if hi >= sl_px:
                    pnl = net_pnl("SELL", entry_px, sl_px, entry_lots, cfg)
                    exit_reason = "sl"
                elif lo <= tp_px:
                    pnl = net_pnl("SELL", entry_px, tp_px, entry_lots, cfg)
                    exit_reason = "tp"
                elif bars_held >= H:
                    exit_p = op + spread / 2   # ask approximation
                    pnl = net_pnl("SELL", entry_px, exit_p, entry_lots, cfg)
                    exit_reason = "timeout"

            if pnl is not None:
                equity += pnl
                trades.append({
                    "time": str(row["time"]),
                    "dir": trade_dir,
                    "entry": entry_px,
                    "exit": sl_px if exit_reason == "sl" else (tp_px if exit_reason == "tp" else exit_p),
                    "lots": entry_lots,
                    "pnl": pnl,
                    "net_pnl": pnl,   # alias expected by profit_gate.py
                    "exit_reason": exit_reason,
                    "fold_id": 0,
                })
                in_trade = False

        # ── Enter new trade ───────────────────────────────────────────────────
        if not in_trade:
            atr = float(row.get("atr", row.get("atr_14", 0.5)))
            if atr <= 0 or pd.isna(atr):
                atr = 0.5

            decision = make_decision(
                probs=prob,
                bar_time=row["time"],
                spread=spread,
                atr=atr,
                cfg=cfg,
                has_open_position=False,
            )

            if decision.action in ("BUY", "SELL"):
                sl_dist = ATR_MULT * atr
                lots = compute_lots(equity, sl_dist, cfg)
                if lots > 0:
                    if decision.action == "BUY":
                        ep = entry_price_long(float(row["open"]), spread, cfg.costs.slippage_price_units)
                        tp_px = ep + ATR_MULT * atr
                        sl_px = ep - ATR_MULT * atr
                        trade_dir = "long"
                    else:
                        ep = entry_price_short(float(row["open"]), cfg.costs.slippage_price_units)
                        tp_px = ep - ATR_MULT * atr
                        sl_px = ep + ATR_MULT * atr
                        trade_dir = "short"
                    entry_px, entry_lots, entry_bar = ep, lots, i
                    in_trade = True

    # ── Save trades ───────────────────────────────────────────────────────────
    trades_df = pd.DataFrame(trades)
    out_path = Path("data/processed/oos_trades.csv")
    trades_df.to_csv(out_path, index=False)
    n_trades = len(trades_df)

    logger.info(f"\nBacktest complete:")
    logger.info(f"  Trades     : {n_trades}")
    logger.info(f"  Final equity: ${equity:,.2f}")

    if n_trades == 0:
        logger.warning("No trades generated — model may not have learned a signal.")
        logger.warning(f"  Max fused prob = {probs_fused.max():.4f}, threshold = {cfg.decision.prob_threshold_grid[0]}")
        save_gate_result({"overall_status": "FAILED", "reason": "zero_trades"})
        return

    gross_pnl = trades_df["pnl"].sum()
    win_rate = (trades_df["pnl"] > 0).mean()
    exit_counts = trades_df["exit_reason"].value_counts().to_dict()

    logger.info(f"  Gross PnL  : ${gross_pnl:,.2f}")
    logger.info(f"  Win rate   : {win_rate:.1%}")
    logger.info(f"  Exits      : {exit_counts}")

    # ── Profit Gate ───────────────────────────────────────────────────────────
    logger.info("\n[M5] Profit Gate evaluation...")
    n_folds = trades_df["fold_id"].nunique() if "fold_id" in trades_df.columns else 1
    result = evaluate_gate(trades_df, n_folds, cfg)
    save_gate_result(result)

    logger.info(f"\n{'='*60}")
    logger.info(f"GATE RESULT: {result['overall_status']}")
    logger.info(f"{'='*60}")
    for k, v in result.items():
        if k != "overall_status":
            logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
