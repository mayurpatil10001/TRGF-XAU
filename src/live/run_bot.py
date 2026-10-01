"""
src/live/run_bot.py
Live trading bot main loop.

- Sleeps to next minute boundary + 1-2s
- Fetches last CLOSED bar(s), appends to rolling buffer
- Builds features with the FROZEN pipeline and scaler
- Runs inference -> decision.py -> order_manager
- Manages open positions (SL/TP/time-exit)
- Enforces all safety rules
- Graceful SIGINT: closes any open position before exit

Usage:
    python -m src.live.run_bot --config config/config.yaml
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, Config
from src.common.logging import setup_logging, get_logger
from src.common.seed import set_global_seed
from src.common.timeutils import next_minute_boundary
from src.evaluation.profit_gate import load_gate_status, GATE_STATUS_FAILED, GATE_STATUS_PROVISIONAL
from src.execution.mt5_connector import MT5Connector
from src.execution.order_manager import OrderManager
from src.execution.safety import SafetyManager
from src.live.inference import LiveInferenceEngine
from src.live.state_store import StateStore
from src.strategy.decision import make_decision
from src.backtest.sizing import compute_lots
from src.labels.triple_barrier import cost_round_trip

logger = get_logger("live.bot")

_shutdown_requested = False


def _handle_sigint(sig, frame):
    global _shutdown_requested
    logger.info("SIGINT received — requesting graceful shutdown.")
    _shutdown_requested = True


def _confirm_live_money() -> bool:
    phrase = input(
        "\n⚠ LIVE MONEY MODE REQUESTED.\n"
        "Type exactly 'I ACCEPT ALL RISK' to continue: "
    ).strip()
    return phrase == "I ACCEPT ALL RISK"


def run_bot(cfg: Config) -> None:
    global _shutdown_requested
    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Safety: live money requires confirmation ─────────────────────────────
    if cfg.live.allow_live_money:
        if not _confirm_live_money():
            logger.info("Live money confirmation rejected. Exiting.")
            return

    # ── Profit gate check ────────────────────────────────────────────────────
    gate_status = load_gate_status()
    signal_only = False

    if gate_status == GATE_STATUS_FAILED:
        # Demo override: allow demo trading even with failed gate (never live money)
        if cfg.live.demo_override_gate and not cfg.live.allow_live_money:
            logger.warning(
                "PROFIT GATE: FAILED — but demo_override_gate=true. "
                "Executing on DEMO account only. Live money is still blocked."
            )
        else:
            logger.warning(
                "PROFIT GATE: FAILED. Running in SIGNAL-ONLY mode. No orders will be placed."
            )
            signal_only = True
    elif gate_status == GATE_STATUS_PROVISIONAL:
        if cfg.live.mode != "demo":
            logger.warning("PROVISIONAL gate: restricting to DEMO mode only.")
            signal_only = True
        else:
            logger.info("PROVISIONAL gate: DEMO trading permitted.")
    else:
        logger.info(f"Profit gate: {gate_status}")

    bot_status = "SIGNAL_ONLY" if signal_only else "RUNNING"

    # ── Initialize ───────────────────────────────────────────────────────────
    state_store = StateStore()
    connector = MT5Connector(cfg)
    order_mgr = OrderManager(cfg, connector)
    safety = SafetyManager(cfg)
    inference = LiveInferenceEngine(cfg)

    if not connector.initialize():
        logger.error("MT5 init failed.")
        return
    if not connector.login():
        logger.error("MT5 login failed.")
        return
    if not connector.resolve_symbol():
        logger.error("Symbol resolution failed.")
        return

    # Rolling bar buffer (keep lookback + buffer)
    buffer_size = cfg.features.lookback + 250
    bar_buffer: deque = deque(maxlen=buffer_size)

    # Seed initial buffer
    initial_bars = connector.get_rates(n_bars=buffer_size)
    if initial_bars is not None:
        for _, row in initial_bars.iterrows():
            bar_buffer.append(row.to_dict())

    acc = connector.get_account_info()
    if acc:
        safety.set_daily_start(float(acc.get("equity", 10_000.0)))
    equity = float(acc.get("equity", 10_000.0)) if acc else 10_000.0

    open_position = None  # dict of current open position info
    entry_bar_count = 0

    logger.info(f"Bot started [{bot_status}]. Symbol={connector.symbol}, Gate={gate_status}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while not _shutdown_requested:
        # Check kill switch
        if safety.check_kill_switch():
            logger.critical("Kill switch active — closing positions and stopping.")
            if open_position:
                positions = connector.get_positions()
                for pos in positions:
                    order_mgr.close_position(pos)
            break

        # Sleep until next minute + 1.5s
        wait = next_minute_boundary() + 1.5
        logger.debug(f"Sleeping {wait:.1f}s for next bar...")
        time.sleep(wait)

        # Fetch new bars
        new_bars = connector.get_rates(n_bars=5)
        if new_bars is None or len(new_bars) == 0:
            logger.warning("No bars fetched; retrying next cycle.")
            continue

        last_bar = new_bars.iloc[-1]
        bar_time_str = str(last_bar["time"])

        # Idempotency
        if safety.is_duplicate_bar(bar_time_str):
            logger.debug(f"Duplicate bar {bar_time_str} — skipping.")
            continue

        # Staleness check
        if not safety.is_data_fresh(last_bar["time"], connector):
            logger.warning("Stale data — skipping cycle.")
            continue

        # Append to buffer
        for _, row in new_bars.iterrows():
            bar_buffer.append(row.to_dict())

        # Build DataFrame from buffer
        buf_df = pd.DataFrame(list(bar_buffer))
        buf_df["time"] = pd.to_datetime(buf_df["time"], utc=True)
        if "spread" not in buf_df.columns:
            buf_df["spread"] = cfg.costs.spread_price_units
        buf_df["gap_before_minutes"] = 0.0
        buf_df["session"] = "unknown"
        buf_df["tick_volume"] = buf_df.get("tick_volume", buf_df.get("real_volume", 1.0))

        # Account update
        acc = connector.get_account_info()
        if acc:
            equity = float(acc.get("equity", equity))

        # ── Manage open position ──────────────────────────────────────────────
        if open_position:
            entry_bar_count += 1
            positions = connector.get_positions()
            if not positions:
                # Position was closed externally (SL/TP hit)
                open_position = None
                entry_bar_count = 0
                safety.record_trade_result(-1.0)  # conservative
            elif entry_bar_count >= cfg.labels.horizon_bars[-1] + 2:
                # Hard failsafe: close any position older than H+2 bars
                logger.warning("Hard failsafe: closing position past max horizon+2.")
                for pos in positions:
                    result = order_mgr.close_position(pos)
                    if result:
                        state_store.log_order(
                            ticket=pos["ticket"], side="CLOSE",
                            lots=pos["volume"], price=0.0, sl=0.0, tp=0.0,
                            retcode=result.get("retcode", -1), comment="failsafe_close",
                        )
                open_position = None
                entry_bar_count = 0

        safety.decrement_cooldown()

        # ── Inference ────────────────────────────────────────────────────────
        try:
            probs, feat_hash = inference.predict(buf_df)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            continue

        tick = connector.get_tick()
        spread = (tick["spread"] if tick else cfg.costs.spread_price_units)
        bar_time_ts = last_bar["time"]
        if hasattr(bar_time_ts, "to_pydatetime"):
            bar_time_ts = pd.Timestamp(bar_time_ts)

        # ── ATR (from buffer) ─────────────────────────────────────────────────
        from src.features.feature_pipeline import wilder_atr
        try:
            atr_val = float(wilder_atr(buf_df, cfg.features.atr_period).iloc[-1])
        except Exception:
            atr_val = cfg.costs.spread_price_units * 10

        decision = make_decision(
            probs=probs,
            bar_time=bar_time_ts,
            spread=spread,
            atr=atr_val,
            cfg=cfg,
            has_open_position=open_position is not None,
            cooldown_bars_remaining=safety.cooldown_bars,
            daily_loss_pct=safety.daily_loss_pct(equity),
            consecutive_losses=safety.consecutive_losses,
        )

        # Log decision
        state_store.log_decision(
            bar_time=bar_time_str,
            features_hash=feat_hash,
            p_long=round(decision.p_long, 4),
            p_short=round(decision.p_short, 4),
            p_none=round(decision.p_none, 4),
            action=decision.action,
            reason=decision.reason,
            spread=round(spread, 4),
            model_version=inference.model_version,
        )

        logger.info(
            f"Bar {bar_time_str} | p_L={decision.p_long:.3f} p_S={decision.p_short:.3f} "
            f"action={decision.action} reason={decision.reason}"
        )

        # ── Place order ───────────────────────────────────────────────────────
        if decision.action in ("BUY", "SELL") and not signal_only and open_position is None:
            lots = compute_lots(equity, decision.sl_dist, cfg)
            if lots > 0 and tick:
                price = tick["ask"] if decision.action == "BUY" else tick["bid"]
                if decision.action == "BUY":
                    sl = price - decision.sl_dist
                    tp = price + decision.tp_dist
                else:
                    sl = price + decision.sl_dist
                    tp = price - decision.tp_dist

                result = order_mgr.send_order(
                    side=decision.action,
                    volume=lots,
                    price=price,
                    sl=sl,
                    tp=tp,
                )
                if result and result.get("retcode") == 10009:
                    open_position = {"ticket": result.get("order"), "side": decision.action,
                                     "lots": lots, "entry_price": price}
                    entry_bar_count = 0
                    state_store.log_order(
                        ticket=result.get("order", 0),
                        side=decision.action,
                        lots=lots,
                        price=price,
                        sl=sl,
                        tp=tp,
                        retcode=result.get("retcode", -1),
                        comment="tegf_xau",
                    )
                else:
                    logger.warning(f"Order not placed: {result}")

        # Small sleep to avoid busy loop
        time.sleep(0.1)

    # ── Graceful shutdown ────────────────────────────────────────────────────
    logger.info("Shutdown: ensuring no orphan positions...")
    if open_position:
        positions = connector.get_positions()
        for pos in positions:
            order_mgr.close_position(pos)
    connector.shutdown()
    logger.info("Bot stopped cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_global_seed(cfg.seed)
    run_bot(cfg)


if __name__ == "__main__":
    main()
