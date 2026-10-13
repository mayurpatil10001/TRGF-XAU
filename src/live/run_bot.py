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
from src.execution.risk_manager import calculate_setup, calculate_lots, update_trail, TrailState
from src.live.inference import LiveInferenceEngine
from src.live.state_store import StateStore
from src.strategy.decision import make_decision
from src.labels.triple_barrier import cost_round_trip
from src.live.trend_filter import compute_trend

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

    open_positions: list = []     # list of active position dicts
    trail_states: dict = {}       # ticket -> TrailState
    atr_series = None             # rolling ATR series for vol-regime detection
    htf_h1_bars = None            # H1 bars for trend filter
    htf_m5_bars = None            # M5 bars for trend filter
    htf_refresh_counter = 5       # start at 5 so first bar triggers immediate HTF fetch
    last_bar_time_str = None      # track last processed bar for sub-minute mode

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

        # Semi-HFT: poll every 30 seconds
        poll_interval = getattr(cfg.live, 'poll_interval_seconds', 30)
        time.sleep(poll_interval)

        # Fetch new bars
        new_bars = connector.get_rates(n_bars=5)
        if new_bars is None or len(new_bars) == 0:
            logger.warning("No bars fetched; retrying next cycle.")
            continue

        last_bar = new_bars.iloc[-1]
        bar_time_str = str(last_bar["time"])

        # Sub-minute: only process each bar once, but still manage positions every poll
        is_new_bar = (bar_time_str != last_bar_time_str)
        if not is_new_bar:
            # Still manage open positions (trailing) even on repeated polls
            pass  # fall through to position management below
        else:
            last_bar_time_str = bar_time_str

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

        # -- Manage open positions (multi-position semi-HFT) --
        live_positions = connector.get_positions()
        live_tickets = {p['ticket'] for p in live_positions}

        # Check for externally closed positions (TP/SL hit)
        for pos_info in list(open_positions):
            if pos_info['ticket'] not in live_tickets:
                safety.record_trade_result(-1.0)
                logger.info("Position " + str(pos_info['ticket']) + " closed. bars=" + str(pos_info.get('bars', 0)))
                trail_states.pop(pos_info['ticket'], None)
                open_positions.remove(pos_info)

        # Update trailing for all open positions every poll
        tick_now = connector.get_tick()
        for pos_info in list(open_positions):
            t = pos_info['ticket']
            if is_new_bar:
                pos_info['bars'] = pos_info.get('bars', 0) + 1
            ts = trail_states.get(t)
            live_pos = next((p for p in live_positions if p['ticket'] == t), None)
            if ts and tick_now and live_pos:
                side = pos_info['side']
                cp = tick_now['bid'] if side == 'BUY' else tick_now['ask']
                new_sl, new_tp = update_trail(ts, cp, atr_val if 'atr_val' in dir() else 1.0)
                if new_sl is not None or new_tp is not None:
                    order_mgr.modify_position(
                        ticket=t,
                        sl=new_sl if new_sl is not None else live_pos.get('sl', ts.setup.sl),
                        tp=new_tp if new_tp is not None else live_pos.get('tp', ts.setup.tp1),
                    )
            # Hard time-stop
            max_bars = cfg.labels.horizon_bars[-1] + 5
            if pos_info.get('bars', 0) >= max_bars and live_pos:
                logger.warning("Hard failsafe: closing ticket " + str(t) + " after " + str(pos_info.get('bars', 0)) + " bars")
                order_mgr.close_position(live_pos)
                trail_states.pop(t, None)
                if pos_info in open_positions:
                    open_positions.remove(pos_info)


        safety.decrement_cooldown()

        # -- Refresh HTF bars every 5 bars (not every bar -- saves API calls) --
        htf_refresh_counter += 1
        if htf_refresh_counter >= 5:
            htf_refresh_counter = 0
            htf_h1_bars = connector.get_htf_rates("H1", n_bars=50)
            htf_m5_bars = connector.get_htf_rates("M5", n_bars=30)

        # -- Compute multi-timeframe trend state --
        trend = compute_trend(htf_h1_bars, htf_m5_bars, adx_threshold=20.0, require_m5_align=True)

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
            atr_series = wilder_atr(buf_df, cfg.features.atr_period)
            atr_val = float(atr_series.iloc[-1])
        except Exception:
            atr_val = cfg.costs.spread_price_units * 10
            atr_series = None

        decision = make_decision(
            probs=probs,
            bar_time=bar_time_ts,
            spread=spread,
            atr=atr_val,
            cfg=cfg,
            has_open_position=len(open_positions) > 0,
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
        # -- HTF Trend Gate --
        from dataclasses import replace as _dc_replace
        if decision.action == "BUY" and not trend.allows_buy:
            logger.info("Trend gate BLOCKED BUY | " + trend.reason)
            decision = _dc_replace(decision, action="NO_TRADE", reason="trend_gate:" + trend.reason)
        elif decision.action == "SELL" and not trend.allows_sell:
            logger.info("Trend gate BLOCKED SELL | " + trend.reason)
            decision = _dc_replace(decision, action="NO_TRADE", reason="trend_gate:" + trend.reason)

        max_pos = getattr(cfg.risk, 'max_open_positions', 3)
        can_open = len(open_positions) < max_pos and is_new_bar
        if decision.action in ("BUY", "SELL") and not signal_only and can_open:
            tick_at_order = connector.get_tick()
            if tick_at_order:
                price = tick_at_order["ask"] if decision.action == "BUY" else tick_at_order["bid"]
                live_spread = tick_at_order.get("spread", spread)
                tp1_rr = getattr(cfg.position_management, 'tp1_rr', 0.8)
                tp2_rr = getattr(cfg.position_management, 'tp2_rr', 1.5)
                setup = calculate_setup(
                    side=decision.action, entry=price, bars=buf_df,
                    atr=atr_val,
                    atr_series=atr_series if atr_series is not None else pd.Series([atr_val]),
                    spread=live_spread, min_rr1=tp1_rr, min_rr2=tp2_rr,
                    min_sl_atr=0.8,
                    max_sl_atr=2.5,
                )
                if setup is None:
                    logger.warning("Risk setup could not be calculated -- skipping order.")
                else:
                    lots = calculate_lots(
                        equity=equity, sl_dist=setup.sl_dist,
                        risk_pct=cfg.risk.risk_fraction_kappa,
                        contract_size=100.0, min_lot=0.01, max_lot=10.0,  # cap at 10L for DEMO safety
                    )
                    if lots >= 0.01:
                        result = order_mgr.send_order(
                            side=decision.action, volume=lots,
                            price=price, sl=setup.sl, tp=setup.tp1,
                        )
                        if result and result.get("retcode") == 10009:
                            new_pos = {
                                "ticket": result.get("order"),
                                "side": decision.action,
                                "lots": lots,
                                "entry_price": price,
                                "bars": 0,
                            }
                            open_positions.append(new_pos)
                            trail_states[result.get("order")] = TrailState(setup=setup)
                            state_store.log_order(
                                ticket=result.get("order", 0), side=decision.action,
                                lots=lots, price=price, sl=setup.sl, tp=setup.tp1,
                                retcode=result.get("retcode", -1),
                                comment=f"rr={setup.rr1}/{setup.rr2} vol={setup.vol_regime} struct={'Y' if setup.structure_sl else 'N'}",
                            )
                            logger.info(
                                f"[OK ORDER] {decision.action} {lots}L @ {price:.2f} | "
                                f"SL={setup.sl:.2f} ({setup.sl_dist/atr_val:.2f}xATR) | "
                                f"TP1={setup.tp1:.2f} TP2={setup.tp2:.2f} | "
                                f"R:R={setup.rr1:.1f}/{setup.rr2:.1f} regime={setup.vol_regime}"
                            )
                        else:
                            logger.warning(f"Order not placed: {result}")
        # Small sleep to avoid busy loop
        time.sleep(0.1)

    # ── Graceful shutdown ────────────────────────────────────────────────────
    logger.info("Shutdown: ensuring no orphan positions...")
    if open_positions:
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
