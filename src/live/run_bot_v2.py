"""
src/live/run_bot_v2.py  — ML-Driven Bar Bot (replaces HFT scalper v5)
======================================================================

ARCHITECTURE (fixes all v5 structural problems):
  • Entry: driven by ML model probability ≥ threshold (0.65+), never by noise
  • Single position at a time (no stacking)
  • TP/SL computed from MID price movement with explicit round-trip cost:
      TP_dist >= 3 × round_trip_cost AND R:R gives achievable breakeven win rate
  • Breakeven edge check: refuses to enter if expected edge is below breakeven
  • Spread filter: skip when spread > max_fraction_of_expected_move
  • deviation_points = 6 (6 points = 0.06 price units max slippage per leg)
  • Partial/requote failures: handled with retry + rejection, no market chasing
  • Symmetric early-stop: counter-trend trades get same stops, not wider
  • Cost-aware exit: every trade logged with full cost breakdown
  • Rolling edge monitor: auto-pause if last N trades' expectancy < threshold

HOW IT WORKS:
  1. Wait for new 1-min bar to close
  2. Get live features from last N bars (same pipeline as backtest)
  3. Call LightGBM model → [p_none, p_long, p_short]
  4. If max(p_long, p_short) >= threshold AND edge check passes → enter
  5. Manage position: check TP/SL/time_exit every 5s

PROFIT GATE:
  The gate must PASS before any orders are placed.
  demo_override_gate=false by default (only override for testing, not production).

Usage:
    python -m src.live.run_bot_v2 --config config/config.yaml
"""
from __future__ import annotations

import json
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, Config
from src.common.logging import setup_logging, get_logger
from src.backtest.costs import (
    entry_price_long, entry_price_short, commission_price_units, net_pnl
)
from src.labels.triple_barrier import cost_round_trip

logger = get_logger("live.bot_v2")
_shutdown = False


def _sig(sig, frame):
    global _shutdown
    _shutdown = True


signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)


# ══════════════════════════════════════════════════════════════════════════════
# Edge / breakeven geometry
# ══════════════════════════════════════════════════════════════════════════════

def breakeven_win_rate(tp_dist: float, sl_dist: float) -> float:
    """Breakeven win rate = SL_dist / (TP_dist + SL_dist)."""
    return sl_dist / (tp_dist + sl_dist + 1e-12)


def compute_tp_sl(atr: float, cfg: Config) -> Tuple[float, float]:
    """
    Compute TP and SL distances from MID price, in price units.
    Rules:
      - cost_rt = spread + 2*slippage + commission_per_lot (in price units)
      - TP_dist >= 3 × cost_rt  (so winning trades clearly beat costs)
      - TP from config: tp_atr_mult * ATR (but at least 3x cost)
      - SL from config: sl_atr_mult * ATR
    """
    pm = cfg.position_management
    cost_rt = cost_round_trip(cfg)
    min_tp = 3.0 * cost_rt
    tp_dist = max(pm.tp_atr_mult * atr, min_tp)
    sl_dist = pm.sl_atr_mult * atr
    return tp_dist, sl_dist


def edge_check_passes(
    p_best: float,
    tp_dist: float,
    sl_dist: float,
    cost_rt: float,
    min_expected_R: float,
) -> Tuple[bool, str]:
    """
    Check whether the expected R is above the min_expected_R threshold.
    Expected PnL = p_win*(TP_dist - cost_rt) + p_loss*(-SL_dist - cost_rt)
                 = p_win*TP_dist - p_loss*SL_dist - cost_rt
    where p_win ≈ p_best (model calibrated probability) and p_loss = 1 - p_best.
    We also check breakeven win rate is achievable.
    """
    bkv_wr = breakeven_win_rate(tp_dist, sl_dist)
    if p_best < bkv_wr:
        return False, f"p_best={p_best:.3f} < breakeven_wr={bkv_wr:.3f}"

    expected_pnl = p_best * (tp_dist - cost_rt) - (1 - p_best) * (sl_dist + cost_rt)
    expected_R = expected_pnl / (sl_dist + 1e-12)
    if expected_R < min_expected_R:
        return False, f"expected_R={expected_R:.3f} < min={min_expected_R:.3f}"

    return True, f"OK: expected_R={expected_R:.3f} bkv_wr={bkv_wr:.3f}"


# ══════════════════════════════════════════════════════════════════════════════
# Rolling edge monitor
# ══════════════════════════════════════════════════════════════════════════════

class EdgeMonitor:
    """
    Tracks last N trade PnL values. Auto-pauses if rolling expectancy < threshold.
    """
    def __init__(self, window: int = 20, pause_threshold: float = -0.5):
        self.window = window
        self.pause_threshold = pause_threshold  # USD per trade
        self._pnl: Deque[float] = deque(maxlen=window)
        self._paused = False
        self._pause_reason = ""

    def add_trade(self, pnl_usd: float):
        self._pnl.append(pnl_usd)
        if len(self._pnl) >= self.window // 2:
            mean_pnl = float(np.mean(self._pnl))
            if mean_pnl < self.pause_threshold:
                if not self._paused:
                    logger.warning(
                        f"[EdgeMonitor] PAUSING: rolling expectancy={mean_pnl:.2f} "
                        f"< threshold={self.pause_threshold:.2f} over {len(self._pnl)} trades"
                    )
                self._paused = True
                self._pause_reason = f"rolling_exp={mean_pnl:.2f}"
            elif self._paused and mean_pnl > 0:
                logger.info("[EdgeMonitor] Resuming — rolling expectancy recovered.")
                self._paused = False

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def rolling_expectancy(self) -> float:
        return float(np.mean(self._pnl)) if self._pnl else 0.0

    def status(self) -> str:
        return (f"EdgeMonitor: paused={self._paused} "
                f"rolling_exp={self.rolling_expectancy:.2f} n={len(self._pnl)}")


# ══════════════════════════════════════════════════════════════════════════════
# Trade cost logger
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    entry_time: str
    exit_time: str
    side: str
    lots: float
    entry_price: float
    exit_price: float
    gross_pnl_usd: float
    spread_paid_pu: float   # price units
    commission_usd: float
    slippage_usd: float
    total_cost_usd: float
    net_pnl_usd: float
    exit_reason: str
    bars_held: int
    p_long: float
    p_short: float
    p_none: float
    session: str = ""


def _log_trade(rec: TradeRecord, log_path: Path):
    """Append trade to JSONL cost-audit log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(vars(rec)) + "\n")
    logger.info(
        f"TRADE EXIT | {rec.exit_reason} | {rec.side} | "
        f"net={rec.net_pnl_usd:+.2f} USD | "
        f"gross={rec.gross_pnl_usd:+.2f} | cost={rec.total_cost_usd:.2f} | "
        f"spread={rec.spread_paid_pu:.3f}pu | comm={rec.commission_usd:.2f} | "
        f"bars={rec.bars_held}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main bot
# ══════════════════════════════════════════════════════════════════════════════

def run_bot(cfg: Config) -> None:
    """
    Main bot loop. Connects to MT5, loads the best LightGBM model from
    artifacts/, polls for new bars and manages one position at a time.
    """
    from src.execution.mt5_connector import MT5Connector
    from src.execution.order_manager import OrderManager
    from src.evaluation.profit_gate import load_gate_status, GATE_STATUS_PASSED

    # ── Profit gate check ─────────────────────────────────────────────────────
    gate_status = load_gate_status()
    if gate_status != GATE_STATUS_PASSED:
        if cfg.live.demo_override_gate and cfg.live.mode == "demo":
            logger.warning(
                f"[GATE] Status={gate_status}. demo_override_gate=True — "
                f"proceeding in demo but NO live money allowed."
            )
        else:
            logger.error(
                f"[GATE] FAILED (status={gate_status}). Bot will not start. "
                f"Run the full pipeline and pass the gate first."
            )
            return

    if cfg.live.allow_live_money:
        logger.error("allow_live_money=true is not permitted. Aborting.")
        return

    # ── MT5 connection ────────────────────────────────────────────────────────
    connector = MT5Connector(cfg)
    if not connector.connect():
        logger.error("MT5 connection failed. Aborting.")
        return

    om = OrderManager(cfg, connector)
    edge_monitor = EdgeMonitor(window=20, pause_threshold=-2.0)  # pause if avg loss > $2/trade
    trade_log_path = Path("logs") / "trade_cost_audit.jsonl"

    logger.info("=" * 60)
    logger.info("TEGF-XAU Bar Bot v2 — ML-driven, single-position")
    logger.info(f"Symbol: {cfg.symbol.name} | Mode: {cfg.live.mode}")
    logger.info(f"Profit gate: {gate_status}")
    logger.info("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────────
    model = _load_best_model(cfg)
    if model is None:
        logger.error("No trained model found in artifacts/. Run the pipeline first.")
        return

    # ── State ─────────────────────────────────────────────────────────────────
    position: Optional[Dict] = None   # current open position dict
    last_bar_time: Optional[pd.Timestamp] = None
    consecutive_losses = 0
    daily_loss_start_equity: Optional[float] = None
    current_day: Optional[object] = None
    cooldown_bars = 0
    cost_rt = cost_round_trip(cfg)

    poll_s = cfg.live.poll_interval_seconds
    kill_file = Path(cfg.live.kill_switch_file)

    while not _shutdown:
        if kill_file.exists():
            logger.warning(f"Kill switch detected: {kill_file}. Shutting down.")
            break

        try:
            tick = connector.get_tick()
            if tick is None:
                time.sleep(poll_s)
                continue

            bid, ask, spread = tick["bid"], tick["ask"], tick["spread"]
            now = pd.Timestamp.utcnow()

            # ── Day reset ─────────────────────────────────────────────────────
            today = now.date()
            if today != current_day:
                current_day = today
                account = connector.get_account_info()
                daily_loss_start_equity = account.get("equity", 10000.0) if account else 10000.0
                logger.info(f"New day {today}: equity={daily_loss_start_equity:.2f}")

            # ── Manage open position ──────────────────────────────────────────
            if position is not None:
                position, closed_trade = _manage_position(
                    position, tick, now, om, cfg, cost_rt, spread
                )
                if closed_trade is not None:
                    _log_trade(closed_trade, trade_log_path)
                    edge_monitor.add_trade(closed_trade.net_pnl_usd)
                    if closed_trade.net_pnl_usd < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    cooldown_bars = cfg.risk.cooldown_bars_after_exit
                    position = None

            # ── Wait for new bar ──────────────────────────────────────────────
            bars = connector.get_bars(cfg.symbol.name, "M1", count=max(cfg.features.lookback + 20, 30))
            if bars is None or len(bars) < cfg.features.lookback + 5:
                logger.warning(
                    f"[BAR WAIT] get_bars returned {len(bars) if bars is not None else 'None'} bars "
                    f"(need >={cfg.features.lookback + 5}). "
                    f"Check MT5 Market Watch has {cfg.symbol.name} subscribed."
                )
                time.sleep(poll_s)
                continue

            latest_bar_time = pd.Timestamp(bars[-1]["time"], unit="s", tz="UTC")
            if latest_bar_time == last_bar_time:
                time.sleep(poll_s)
                continue
            last_bar_time = latest_bar_time

            # ── Guard: no entry if active position, cooldown, risk limits ─────
            if position is not None:
                time.sleep(poll_s)
                continue
            if cooldown_bars > 0:
                cooldown_bars -= 1
                time.sleep(poll_s)
                continue
            if consecutive_losses >= cfg.risk.max_consecutive_losses:
                logger.warning(f"Consecutive losses={consecutive_losses}. Paused.")
                time.sleep(poll_s * 5)
                continue
            if edge_monitor.is_paused:
                logger.warning(f"EdgeMonitor paused. {edge_monitor.status()}")
                time.sleep(poll_s * 5)
                continue

            # ── Daily loss limit ──────────────────────────────────────────────
            if daily_loss_start_equity is not None:
                account = connector.get_account_info()
                equity = account.get("equity", daily_loss_start_equity) if account else daily_loss_start_equity
                daily_loss_pct = max(0.0, (daily_loss_start_equity - equity) / daily_loss_start_equity * 100)
                if daily_loss_pct >= cfg.risk.daily_loss_limit_pct:
                    logger.warning(f"Daily loss limit {daily_loss_pct:.1f}% hit. Paused until midnight.")
                    time.sleep(poll_s * 10)
                    continue

            # ── Spread filter ─────────────────────────────────────────────────
            if spread > cfg.decision.max_spread_price_units:
                time.sleep(poll_s)
                continue

            # ── Get features and model prediction ─────────────────────────────
            df_bars = pd.DataFrame(bars)
            probs, atr = _predict(model, df_bars, cfg)
            if probs is None or atr <= 0:
                time.sleep(poll_s)
                continue

            p_none, p_long, p_short = float(probs[0]), float(probs[1]), float(probs[2])
            if p_long >= p_short:
                best_dir = "BUY"
                p_best = p_long
            else:
                best_dir = "SELL"
                p_best = p_short

            # ── Probability threshold ─────────────────────────────────────────
            threshold = cfg.decision.prob_threshold_grid[0]
            if p_best < threshold:
                time.sleep(poll_s)
                continue

            # ── TP/SL geometry and edge check ─────────────────────────────────
            tp_dist, sl_dist = compute_tp_sl(atr, cfg)
            ok, reason = edge_check_passes(p_best, tp_dist, sl_dist, cost_rt, cfg.decision.min_expected_R)
            if not ok:
                logger.debug(f"Edge check failed: {reason}")
                time.sleep(poll_s)
                continue

            # ── Session filter ────────────────────────────────────────────────
            from src.common.timeutils import get_session, is_in_blackout
            session_map = {k: (v[0], v[1]) for k, v in cfg.sessions.model_dump().items()}
            current_session = get_session(now, session_map)
            if current_session not in cfg.decision.sessions_allowed:
                time.sleep(poll_s)
                continue
            if is_in_blackout(now, cfg.decision.blackout_utc):
                time.sleep(poll_s)
                continue

            # ── Enter position ────────────────────────────────────────────────
            account = connector.get_account_info()
            equity = account.get("equity", 10000.0) if account else 10000.0
            lots = _compute_lots(equity, sl_dist, cfg)
            if lots <= 0:
                time.sleep(poll_s)
                continue

            if best_dir == "BUY":
                entry_p = ask + cfg.costs.slippage_price_units
                tp_price = entry_p + tp_dist
                sl_price = entry_p - sl_dist
            else:
                entry_p = bid - cfg.costs.slippage_price_units
                tp_price = entry_p - tp_dist
                sl_price = entry_p + sl_dist

            ticket = om.send_order(
                side=best_dir,
                volume=lots,
                price=entry_p,
                sl=sl_price,
                tp=tp_price,
                comment="BotV2",
            )
            if ticket is not None:
                position = {
                    "ticket": ticket,
                    "side": best_dir,
                    "lots": lots,
                    "entry_price": entry_p,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "entry_time": now,
                    "bars_held": 0,
                    "tp_dist": tp_dist,
                    "sl_dist": sl_dist,
                    "p_long": p_long,
                    "p_short": p_short,
                    "p_none": p_none,
                    "session": current_session,
                    "spread_at_entry": spread,
                }
                logger.info(
                    f"ENTRY | {best_dir} {lots}L @ {entry_p:.3f} | "
                    f"TP={tp_price:.3f} SL={sl_price:.3f} | "
                    f"p={p_best:.3f}>{threshold} | {reason} | #{ticket}"
                )

        except Exception as e:
            logger.error(f"Bot loop error: {e}", exc_info=True)

        time.sleep(poll_s)

    logger.info("Bot shutdown complete.")
    connector.disconnect()


def _manage_position(
    position: Dict,
    tick: Dict,
    now: pd.Timestamp,
    om,
    cfg: Config,
    cost_rt: float,
    spread: float,
) -> Tuple[Optional[Dict], Optional[TradeRecord]]:
    """
    Check open position for TP/SL/time exit.
    Returns (new_position, closed_trade_record).
    """
    bid, ask = tick["bid"], tick["ask"]
    side = position["side"]
    tp_price = position["tp_price"]
    sl_price = position["sl_price"]
    entry_price = position["entry_price"]
    entry_time = position["entry_time"]
    lots = position["lots"]

    # Determine current exit price (mark-to-market)
    if side == "BUY":
        current_exit = bid   # BUY exits at bid
        tp_hit = bid >= tp_price
        sl_hit = bid <= sl_price
    else:
        current_exit = ask   # SELL exits at ask
        tp_hit = ask <= tp_price
        sl_hit = ask >= sl_price

    # Time exit: after max_horizon from entry
    bars_held = position.get("bars_held", 0) + 1
    position["bars_held"] = bars_held
    max_horizon = max(cfg.labels.horizon_bars)  # use max configured horizon
    time_exit = bars_held >= max_horizon

    exit_reason = None
    exit_price = None

    if tp_hit and sl_hit:
        exit_price = sl_price   # SL wins (conservative)
        exit_reason = "SL_first"
    elif tp_hit:
        exit_price = tp_price
        exit_reason = "TP"
    elif sl_hit:
        exit_price = sl_price
        exit_reason = "SL"
    elif time_exit:
        exit_price = current_exit
        exit_reason = "time_exit"

    if exit_reason is None:
        return position, None  # still open

    # Execute close
    closed = om.close_position({"ticket": position["ticket"]})
    if closed is None:
        return position, None  # close failed, keep managing

    # Compute cost breakdown
    direction = 1.0 if side == "BUY" else -1.0
    gross_pnl = direction * (exit_price - entry_price) * lots * cfg.symbol.contract_size
    comm_usd = cfg.costs.commission_per_lot_round_trip * lots
    slippage_usd = 2 * cfg.costs.slippage_price_units * lots * cfg.symbol.contract_size
    spread_paid_pu = position.get("spread_at_entry", cfg.costs.spread_price_units)
    spread_cost_usd = spread_paid_pu * lots * cfg.symbol.contract_size
    total_cost = comm_usd + slippage_usd + spread_cost_usd
    net = gross_pnl - total_cost

    rec = TradeRecord(
        entry_time=str(entry_time),
        exit_time=str(now),
        side=side,
        lots=lots,
        entry_price=entry_price,
        exit_price=exit_price,
        gross_pnl_usd=round(gross_pnl, 4),
        spread_paid_pu=round(spread_paid_pu, 4),
        commission_usd=round(comm_usd, 4),
        slippage_usd=round(slippage_usd, 4),
        total_cost_usd=round(total_cost, 4),
        net_pnl_usd=round(net, 4),
        exit_reason=exit_reason,
        bars_held=bars_held,
        p_long=position.get("p_long", 0.0),
        p_short=position.get("p_short", 0.0),
        p_none=position.get("p_none", 0.0),
        session=position.get("session", ""),
    )
    return None, rec


def _compute_lots(equity: float, sl_dist: float, cfg: Config) -> float:
    """Kelly-fraction lot sizing based on risk per trade."""
    kappa = cfg.risk.risk_fraction_kappa
    risk_usd = equity * kappa
    dollar_per_lot_per_pu = cfg.symbol.contract_size
    lots = risk_usd / (sl_dist * dollar_per_lot_per_pu + 1e-9)
    return max(0.01, min(round(lots, 2), 5.0))


def _load_best_model(cfg: Config):
    """Load the most recent LightGBM model from artifacts."""
    import glob
    try:
        import lightgbm as lgb
        paths = sorted(glob.glob("artifacts/fold_*/lgbm.txt"))
        if not paths:
            logger.error("No LightGBM model found in artifacts/fold_*/lgbm.txt")
            return None
        model_path = paths[-1]   # use latest fold model
        model = lgb.Booster(model_file=model_path)
        logger.info(f"Loaded model: {model_path}")
        return model
    except ImportError:
        logger.error("lightgbm not installed")
        return None
    except Exception as e:
        logger.error(f"Model load error: {e}")
        return None


def _predict(model, df_bars: pd.DataFrame, cfg: Config):
    """
    Run feature engineering on latest bars and return (probs[3], atr).
    Returns (None, 0) on error.
    """
    try:
        import json as _json
        import joblib

        # Load feature names
        feat_names_path = Path("artifacts/feature_names.json")
        if not feat_names_path.exists():
            logger.error("artifacts/feature_names.json not found. Run pipeline first.")
            return None, 0.0

        with open(feat_names_path) as f:
            feature_names = _json.load(f)

        # Normalise bar columns
        df = df_bars.copy()
        for col in ["time", "open", "high", "low", "close", "tick_volume"]:
            if col not in df.columns:
                if col == "tick_volume":
                    df[col] = 0.0
                else:
                    return None, 0.0

        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.sort_values("time").reset_index(drop=True)

        from src.features.feature_pipeline import build_features, wilder_atr
        feat_df, _ = build_features(df, cfg)
        atr_series = wilder_atr(df, cfg.features.atr_period)
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

        if feat_df.empty:
            return None, 0.0

        # Try to load scaler from latest fold
        scaler_paths = sorted(Path("artifacts").glob("fold_*/scaler.joblib"))
        if scaler_paths:
            scaler = joblib.load(scaler_paths[-1])
            from src.features.feature_pipeline import apply_scaler
            feat_df = apply_scaler(feat_df, scaler)

        # Get latest bar features
        row = feat_df[feature_names].iloc[[-1]].values.astype(np.float32)
        if np.isnan(row).any():
            return None, atr

        probs = model.predict(row)[0]  # LightGBM: returns [N, n_classes]
        return probs, atr

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=False)
        return None, 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TEGF-XAU ML Bar Bot v2")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)

    logger.info("Starting ML Bar Bot v2...")
    run_bot(cfg)


if __name__ == "__main__":
    main()
