"""
src/live/run_hft_bot.py  — HFT SCALPER v5 (Momentum-Following)
════════════════════════════════════════════════════════════════

STRATEGY: Trade WITH current tick momentum, filtered by H1 regime.

ENTRY LOGIC:
  1. Compute tick momentum from last 5 bid prices
  2. If ≥3/5 ticks falling  → SELL  (price moving down NOW)
  3. If ≥3/5 ticks rising   → BUY   (price moving up NOW)
  4. H1 regime filter:
       - H1=BEAR: allow SELL freely, allow BUY with half size only
       - H1=BULL: allow BUY freely, allow SELL with half size only
       - H1=NEUTRAL: only trade if ADX > 25 (some trend present)

EXIT LOGIC (ultra-fast, local P&L):
  1. Compute P&L = entry - current_bid (SELL) or current_ask - entry (BUY)
  2. If profit ≥ 0.3pt → close IMMEDIATELY (book it, don't wait)
  3. If loss > 0.6pt   → close manually (don't wait for 0.8pt SL)
  4. If age > 20s      → close (free the slot)

PARAMETERS:
  Poll      = 0.3s
  ProfitLock= 0.3pt
  EarlyStop = 0.6pt  (exit before SL hits)
  SL_MT5    = 0.8pt  (safety net only)
  MaxPos    = 5
  Cooldown  = 2s
  MaxAge    = 20s
"""
from __future__ import annotations

import time
import signal
import collections
from pathlib import Path
from typing import Deque

from src.common.config import load_config
from src.common.logging import setup_logging, get_logger
from src.execution.mt5_connector import MT5Connector
from src.execution.order_manager import OrderManager
from src.live.trend_filter import compute_trend, TrendState

logger = get_logger("live.hft_bot")
_shutdown = False


def _sig(sig, frame):
    global _shutdown
    _shutdown = True


signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)

# ══════════════════════════════════════════════════════════════════════════════
POLL_S         = 0.3
PROFIT_PT      = 0.4    # Real profit threshold — must exceed spread (~0.25pt)
                        # SELL exits at ASK, BUY exits at BID, so spread is a cost
EARLY_STOP_PT  = 0.7    # Cut at -0.7pt only (give trades more room)
SL_MT5_PT      = 0.8    # MT5 hard SL safety
TP_SAFETY_PT   = 8.0
MAX_POS        = 5
COOLDOWN_S     = 2.0
RISK_PCT       = 0.001
HTF_S          = 20     # Refresh trend every 20s
MAX_SPREAD_PT  = 0.35
MAX_AGE_S      = 15     # Free slot after 15s (faster cycling)
TICK_BUF       = 5
TICK_MIN       = 2      # ≥2/5 ticks confirming (more frequent entries)
# ══════════════════════════════════════════════════════════════════════════════


class TickDir:
    def __init__(self, n=5):
        self.q: Deque[float] = collections.deque(maxlen=n)
        self.n = n

    def push(self, bid: float):
        self.q.append(bid)

    def ready(self) -> bool:
        return len(self.q) >= self.n

    def falls(self) -> int:
        p = list(self.q)
        return sum(1 for i in range(1, len(p)) if p[i] < p[i-1])

    def rises(self) -> int:
        p = list(self.q)
        return sum(1 for i in range(1, len(p)) if p[i] > p[i-1])


def _lots(equity: float, half: bool = False) -> float:
    risk = equity * RISK_PCT * (0.5 if half else 1.0)
    lot  = risk / (SL_MT5_PT * 100.0)
    return max(0.01, min(round(lot, 2), 2.0))


def _place(order_mgr, connector, side: str,
           price: float, lots: float) -> dict | None:
    sl = round(price + SL_MT5_PT, 2) if side == "SELL" else round(price - SL_MT5_PT, 2)
    tp = round(price - TP_SAFETY_PT, 2) if side == "SELL" else round(price + TP_SAFETY_PT, 2)
    r  = order_mgr.send_order(side=side, volume=lots, price=price,
                              sl=sl, tp=tp, comment="HFT")
    if r and r.get("retcode") == 10009:
        t = r.get("order")
        logger.info(f">>> {side} {lots}L @ {price:.3f}  SL={sl}  #{t}")
        return {"ticket": t, "side": side, "entry": price,
                "open_ts": time.time()}
    logger.warning(f"Rejected: {r}")
    return None


def _close_pos(order_mgr, connector, pos: dict, reason: str,
               unreal: float, n_profit: int, n_sl: int, n_time: int):
    live = next((p for p in connector.get_positions()
                 if p["ticket"] == pos["ticket"]), None)
    if live:
        order_mgr.close_position(live)
        if reason == "PROFIT":
            logger.info(f"✅ +{unreal:.3f}pt  #{pos['ticket']}"
                        f"  age={time.time()-pos['open_ts']:.0f}s"
                        f"  P={n_profit} SL={n_sl} T={n_time}")
        elif reason == "EARLY_STOP":
            logger.warning(f"🛑 EARLY_STOP {unreal:+.3f}pt  #{pos['ticket']}"
                           f"  P={n_profit} SL={n_sl} T={n_time}")
        else:
            logger.info(f"⏱ TIME {unreal:+.3f}pt  #{pos['ticket']}"
                        f"  P={n_profit} SL={n_sl} T={n_time}")
    return live is not None


def run_hft_bot():
    cfg = load_config()
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir)

    logger.info("═" * 55)
    logger.info("  HFT SCALPER v5 — Momentum Following")
    logger.info(f"  Poll={POLL_S}s  Profit={PROFIT_PT}pt(spread-adj)  EarlyStop={EARLY_STOP_PT}pt")
    logger.info(f"  MaxPos={MAX_POS}  Cooldown={COOLDOWN_S}s  MaxAge={MAX_AGE_S}s")
    logger.info(f"  Entry: {TICK_MIN}/{TICK_BUF} ticks confirm direction")
    logger.info("═" * 55)

    kill_file = Path(cfg.live.kill_switch_file)
    connector = MT5Connector(cfg)
    if not connector.initialize() or not connector.login() or not connector.resolve_symbol():
        logger.error("MT5 init failed.")
        return

    order_mgr = OrderManager(cfg, connector)
    logger.info(f"Connected: {connector.symbol}")

    open_positions: list[dict] = []
    trend: TrendState | None   = None
    td    = TickDir(TICK_BUF)
    last_htf   = 0.0
    last_entry = 0.0
    equity     = 10_000.0
    n_enter = n_profit = n_sl = n_time = n_early = 0

    try:
        while not _shutdown:
            if kill_file.exists():
                break
            t0 = time.time()

            # Equity
            acc = connector.get_account_info()
            if acc:
                equity = float(acc.get("equity", equity))

            # Trend refresh
            now = time.time()
            if now - last_htf >= HTF_S:
                h1 = connector.get_htf_rates("H1", n_bars=50)
                m5 = connector.get_htf_rates("M5", n_bars=20)
                trend = compute_trend(h1, m5, adx_threshold=20.0,
                                      require_m5_align=False)
                last_htf = now
                logger.info(
                    f"H1={trend.h1_trend} ADX={trend.h1_adx:.1f} "
                    f"M5={trend.m5_slope}"
                )

            if trend is None:
                time.sleep(POLL_S)
                continue

            # Live tick
            tick = connector.get_tick()
            if not tick:
                time.sleep(POLL_S)
                continue

            bid    = tick["bid"]
            ask    = tick["ask"]
            spread = ask - bid
            td.push(bid)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # FAST EXIT — local P&L, runs every 0.3s
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            for pos in list(open_positions):
                side = pos["side"]
                age  = time.time() - pos["open_ts"]
                # Spread-adjusted P&L:
                # SELL: entered at bid, exits by buying back at ASK → cost = ask
                # BUY:  entered at ask, exits by selling at BID     → revenue = bid
                unr = (pos["entry"] - ask) if side == "SELL" else (bid - pos["entry"])

                if unr >= PROFIT_PT:
                    if _close_pos(order_mgr, connector, pos,
                                  "PROFIT", unr, n_profit+1, n_sl, n_time):
                        n_profit += 1
                    open_positions.remove(pos)

                elif unr <= -EARLY_STOP_PT:
                    # Only early-stop trend-aligned trades; counter-trend
                    # trades get the full SL room since they're harder to time
                    is_counter = (
                        (pos["side"] == "BUY"  and trend and trend.h1_trend == "BEAR")
                        or (pos["side"] == "SELL" and trend and trend.h1_trend == "BULL")
                    )
                    if not is_counter:
                        if _close_pos(order_mgr, connector, pos,
                                      "EARLY_STOP", unr, n_profit, n_sl, n_early+1):
                            n_early += 1
                        open_positions.remove(pos)

                elif age >= MAX_AGE_S:
                    if _close_pos(order_mgr, connector, pos,
                                  "TIME", unr, n_profit, n_sl, n_time+1):
                        n_time += 1
                    open_positions.remove(pos)

            # Detect MT5 SL hits
            if open_positions:
                live_tkts = {p["ticket"] for p in connector.get_positions()}
                for pos in list(open_positions):
                    if pos["ticket"] not in live_tkts:
                        n_sl += 1
                        logger.warning(
                            f"MT5-SL #{pos['ticket']}"
                            f"  age={time.time()-pos['open_ts']:.0f}s"
                            f"  P={n_profit} SL={n_sl}"
                        )
                        open_positions.remove(pos)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # PRECISION ENTRY — tick momentum + H1 regime
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if (spread <= MAX_SPREAD_PT
                    and td.ready()
                    and len(open_positions) < MAX_POS
                    and (time.time() - last_entry) >= COOLDOWN_S):

                falls = td.falls()
                rises = td.rises()
                side_to_trade = None
                half_size     = False

                if falls >= TICK_MIN:
                    # Ticks falling → SELL momentum
                    if trend.h1_trend in ("BEAR", "NEUTRAL"):
                        side_to_trade = "SELL"
                        half_size = (trend.h1_trend == "NEUTRAL")
                    # Don't SELL in strong BULL

                elif rises >= TICK_MIN:
                    # Ticks rising → BUY momentum
                    if trend.h1_trend in ("BULL", "NEUTRAL"):
                        side_to_trade = "BUY"
                        half_size = (trend.h1_trend == "NEUTRAL")
                    elif trend.h1_trend == "BEAR":
                        # Counter-trend bounce — BUY with half size, exit fast
                        side_to_trade = "BUY"
                        half_size = True

                else:
                    # No clear tick direction — use M5 slope as fallback
                    if trend.m5_slope == "UP" and trend.h1_trend != "BEAR":
                        side_to_trade = "BUY"
                        half_size = True
                    elif trend.m5_slope == "DOWN" and trend.h1_trend != "BULL":
                        side_to_trade = "SELL"
                        half_size = True

                if side_to_trade:
                    price = bid if side_to_trade == "SELL" else ask
                    lots  = _lots(equity, half=half_size)
                    pos   = _place(order_mgr, connector,
                                   side_to_trade, price, lots)
                    if pos:
                        open_positions.append(pos)
                        last_entry = time.time()
                        n_enter += 1

            # Sleep
            time.sleep(max(0.0, POLL_S - (time.time() - t0)))

    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Closing all on exit...")
        for lp in connector.get_positions():
            order_mgr.close_position(lp)
        connector.shutdown()
        wr = (n_profit / n_enter * 100) if n_enter else 0
        logger.info(
            f"DONE  entered={n_enter}  profit={n_profit}  "
            f"early_stop={n_early}  sl={n_sl}  time={n_time}  "
            f"win_rate={wr:.1f}%"
        )


def main():
    run_hft_bot()


if __name__ == "__main__":
    main()
