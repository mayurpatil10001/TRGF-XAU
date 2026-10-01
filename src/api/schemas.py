"""
src/api/schemas.py
Pydantic schemas for WebSocket and REST payloads.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class AccountInfo(BaseModel):
    login: int = 0
    server: str = ""
    mode: str = "DEMO"
    currency: str = "USD"


class OpenPosition(BaseModel):
    ticket: int = 0
    side: str = "BUY"
    lots: float = 0.0
    entry: float = 0.0
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    pnl: float = 0.0
    bars_held: int = 0


class MarketSnapshot(BaseModel):
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0


class LastSignal(BaseModel):
    bar_time: str = ""
    action: str = "NO_TRADE"
    p_long: float = 0.0
    p_short: float = 0.0
    p_none: float = 0.0
    threshold: float = 0.0
    reason: str = ""


class BotStatus(BaseModel):
    status: str = "RUNNING"
    daily_loss_used_pct: float = 0.0
    consecutive_losses: int = 0


class ProfitGateStatus(BaseModel):
    status: str = "UNKNOWN"
    updated: str = ""


class LiveSnapshot(BaseModel):
    ts: str = ""
    seq: int = 0
    mt5_connected: bool = False
    account: AccountInfo = AccountInfo()
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    floating_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    realized_pnl_total: float = 0.0
    open_positions: List[OpenPosition] = []
    market: MarketSnapshot = MarketSnapshot()
    last_signal: LastSignal = LastSignal()
    bot: BotStatus = BotStatus()
    profit_gate: ProfitGateStatus = ProfitGateStatus()
