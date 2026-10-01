"""
src/execution/safety.py
Safety subsystem: kill switch, daily loss limit, consecutive-loss pause,
spread cap, stale-data guard, graceful shutdown.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from src.common.config import Config
from src.common.logging import get_logger

logger = get_logger("execution.safety")


class SafetyManager:
    """Tracks and enforces all safety constraints."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.kill_switch_active = False
        self.paused = False
        self.daily_start_equity: float = 0.0
        self.consecutive_losses: int = 0
        self.cooldown_bars: int = 0
        self.session_loss_count: int = 0
        self.last_processed_bar_time: Optional[str] = None  # idempotency

    def check_kill_switch(self) -> bool:
        """True if the kill switch file exists."""
        kill_file = Path(self.cfg.live.kill_switch_file)
        if kill_file.exists():
            logger.critical(f"KILL SWITCH ACTIVATED: {kill_file} exists.")
            self.kill_switch_active = True
        return self.kill_switch_active

    def set_daily_start(self, equity: float) -> None:
        self.daily_start_equity = equity

    def daily_loss_pct(self, current_equity: float) -> float:
        if self.daily_start_equity <= 0:
            return 0.0
        return max(0.0, (self.daily_start_equity - current_equity) / self.daily_start_equity * 100)

    def is_daily_loss_hit(self, current_equity: float) -> bool:
        pct = self.daily_loss_pct(current_equity)
        if pct >= self.cfg.risk.daily_loss_limit_pct:
            logger.warning(f"Daily loss limit hit: {pct:.2f}%")
            return True
        return False

    def record_trade_result(self, pnl: float) -> None:
        if pnl < 0:
            self.consecutive_losses += 1
            self.session_loss_count += 1
            logger.info(f"Loss #{self.consecutive_losses} consecutive, {self.session_loss_count} today.")
        else:
            self.consecutive_losses = 0
        self.cooldown_bars = self.cfg.risk.cooldown_bars_after_exit

    def is_consecutive_loss_pause(self) -> bool:
        if self.consecutive_losses >= self.cfg.risk.max_consecutive_losses:
            logger.warning(
                f"Consecutive loss pause: {self.consecutive_losses} losses. "
                "Pausing for rest of session."
            )
            return True
        return False

    def is_spread_ok(self, spread: float) -> bool:
        return spread <= self.cfg.decision.max_spread_price_units

    def is_data_fresh(self, last_bar_time, connector) -> bool:
        return connector.check_staleness(last_bar_time)

    def is_duplicate_bar(self, bar_time_str: str) -> bool:
        """Idempotency: skip if we already processed this bar."""
        if bar_time_str == self.last_processed_bar_time:
            return True
        self.last_processed_bar_time = bar_time_str
        return False

    def decrement_cooldown(self) -> None:
        if self.cooldown_bars > 0:
            self.cooldown_bars -= 1

    def reset_session(self) -> None:
        """Call at session boundary to reset session-scoped state."""
        self.session_loss_count = 0
        self.consecutive_losses = 0
