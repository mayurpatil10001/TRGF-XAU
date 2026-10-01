"""
src/api/mt5_poller.py
Asyncio background task: polls MT5 every 1 second and updates shared state.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from src.api.schemas import (
    AccountInfo, BotStatus, LiveSnapshot, MarketSnapshot,
    OpenPosition, ProfitGateStatus
)
from src.common.config import Config
from src.common.logging import get_logger
from src.evaluation.profit_gate import load_gate_status

logger = get_logger("api.mt5_poller")

# Global mutable snapshot (updated by the poller, read by WebSocket handler)
_current_snapshot: LiveSnapshot = LiveSnapshot()
_snapshot_seq: int = 0
_mt5_connected: bool = False


def get_snapshot() -> LiveSnapshot:
    return _current_snapshot


async def poll_mt5(cfg: Config) -> None:
    """
    Asyncio task: poll MT5 every 1s and update _current_snapshot.
    Pushes only on change (or unconditionally every second).
    """
    global _current_snapshot, _snapshot_seq, _mt5_connected

    try:
        import MetaTrader5 as mt5
        MT5_OK = True
    except ImportError:
        MT5_OK = False

    # Initialize MT5 in this API process (separate from the live bot process)
    if MT5_OK:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        _login = int(os.getenv("MT5_LOGIN", "0"))
        _password = os.getenv("MT5_PASSWORD", "")
        _server = os.getenv("MT5_SERVER", "")
        if not mt5.initialize():
            logger.warning("MT5 initialize() failed in API poller — balance will show $0")
            MT5_OK = False
        elif _login and not mt5.login(_login, password=_password, server=_server):
            logger.warning(f"MT5 login failed in API poller: {mt5.last_error()}")

    gate_status = load_gate_status()
    realized_total: float = 0.0
    bot_start_time = datetime.now(timezone.utc)

    while True:
        try:
            snapshot = LiveSnapshot()
            snapshot.ts = datetime.now(timezone.utc).isoformat()
            _snapshot_seq += 1
            snapshot.seq = _snapshot_seq
            snapshot.profit_gate = ProfitGateStatus(status=gate_status, updated="")

            if MT5_OK and mt5.terminal_info() is not None:
                _mt5_connected = True
                snapshot.mt5_connected = True

                acc = mt5.account_info()
                if acc:
                    snapshot.account = AccountInfo(
                        login=acc.login,
                        server=acc.server,
                        mode="DEMO" if acc.trade_mode == 0 else "LIVE",
                        currency=acc.currency,
                    )
                    snapshot.balance = round(acc.balance, 2)
                    snapshot.equity = round(acc.equity, 2)
                    snapshot.margin = round(acc.margin, 2)
                    snapshot.free_margin = round(acc.margin_free, 2)
                    snapshot.floating_pnl = round(acc.profit, 2)

                # Live positions filtered by magic number
                positions = mt5.positions_get(symbol=cfg.symbol.name) or []
                open_pos = []
                for p in positions:
                    if p.magic == cfg.live.magic_number:
                        open_pos.append(OpenPosition(
                            ticket=p.ticket,
                            side="BUY" if p.type == 0 else "SELL",
                            lots=p.volume,
                            entry=p.price_open,
                            price=p.price_current,
                            sl=p.sl,
                            tp=p.tp,
                            pnl=round(p.profit, 2),
                            bars_held=0,  # updated by bot state
                        ))
                snapshot.open_positions = open_pos

                # Tick
                tick = mt5.symbol_info_tick(cfg.symbol.name)
                if tick:
                    snapshot.market = MarketSnapshot(
                        bid=tick.bid,
                        ask=tick.ask,
                        spread=round(tick.ask - tick.bid, 4),
                    )

                # Realized P/L today
                day_start = datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                deals = mt5.history_deals_get(day_start, datetime.now(timezone.utc))
                if deals:
                    realized_today = sum(
                        d.profit for d in deals
                        if d.magic == cfg.live.magic_number
                    )
                    snapshot.realized_pnl_today = round(realized_today, 2)

                # Total realized since bot start
                all_deals = mt5.history_deals_get(bot_start_time, datetime.now(timezone.utc))
                if all_deals:
                    realized_total = sum(
                        d.profit for d in all_deals
                        if d.magic == cfg.live.magic_number
                    )
                snapshot.realized_pnl_total = round(realized_total, 2)

            else:
                _mt5_connected = False
                snapshot.mt5_connected = False

            snapshot.bot = BotStatus(status="RUNNING")
            _current_snapshot = snapshot

        except Exception as e:
            logger.warning(f"MT5 poll error: {e}")
            _mt5_connected = False

        await asyncio.sleep(1.0)
