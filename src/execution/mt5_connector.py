"""
src/execution/mt5_connector.py
MetaTrader5 connection management.

- Reads credentials from .env (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH)
- Verifies account is DEMO unless allow_live_money is explicitly set
- Resolves symbol with suffix variants
- Reads live spread, contract info from MT5
- bar data: copy_rates_from_pos with position 1 (last CLOSED bar; never bar 0)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.common.config import Config
from src.common.logging import get_logger

logger = get_logger("execution.mt5_connector")

# Try importing MT5; gracefully degrade on non-Windows or missing installation
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not available — running in MOCK mode.")


SYMBOL_SUFFIXES = ["", "m", ".a", ".c", "_", "pro", "ecn"]
TRADE_MODE_DEMO = 0  # mt5.ACCOUNT_TRADE_MODE_DEMO


class MT5Connector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._initialized = False
        self._symbol: Optional[str] = None
        self._symbol_info = None

    def initialize(self) -> bool:
        """Load .env credentials and initialize MT5 terminal."""
        load_dotenv()
        path = os.getenv("MT5_PATH", "")
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available; skipping initialization.")
            return False
        if path:
            ok = mt5.initialize(path=path)
        else:
            ok = mt5.initialize()
        if not ok:
            err = mt5.last_error()
            logger.error(f"MT5 initialize failed: {err}")
            return False
        self._initialized = True
        logger.info("MT5 initialized.")
        return True

    def login(self) -> bool:
        """Log in using .env credentials."""
        if not MT5_AVAILABLE or not self._initialized:
            return False
        load_dotenv()
        login = int(os.getenv("MT5_LOGIN", "0"))
        password = os.getenv("MT5_PASSWORD", "")
        server = os.getenv("MT5_SERVER", "")
        ok = mt5.login(login, password=password, server=server)
        if not ok:
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            return False

        # Safety check: must be DEMO unless explicitly permitted
        acc = mt5.account_info()
        if acc is None:
            logger.error("Cannot read account info after login.")
            return False

        if acc.trade_mode != TRADE_MODE_DEMO and not self.cfg.live.allow_live_money:
            logger.critical(
                "SAFETY BLOCK: MT5 account is NOT a demo account and "
                "allow_live_money=false in config. Refusing to connect."
            )
            mt5.shutdown()
            return False

        mode_str = "DEMO" if acc.trade_mode == TRADE_MODE_DEMO else "LIVE"
        logger.info(f"Logged in: #{acc.login} @ {acc.server} [{mode_str}] "
                    f"balance={acc.balance:.2f} {acc.currency}")
        return True

    def resolve_symbol(self) -> Optional[str]:
        """Find the correct symbol name (try suffixes)."""
        if not MT5_AVAILABLE:
            return self.cfg.symbol.name
        base = self.cfg.symbol.name
        for sfx in SYMBOL_SUFFIXES:
            candidate = base + sfx
            info = mt5.symbol_info(candidate)
            if info is not None:
                mt5.symbol_select(candidate, True)
                self._symbol = candidate
                self._symbol_info = info
                logger.info(f"Resolved symbol: {candidate}")
                return candidate
        logger.error(f"Could not resolve symbol: {base}")
        return None

    @property
    def symbol(self) -> str:
        return self._symbol or self.cfg.symbol.name

    @property
    def symbol_info(self):
        return self._symbol_info

    def get_rates(self, n_bars: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch last n_bars CLOSED bars (position 1 = skip the forming bar 0).
        Returns DataFrame with OHLCV, or None on failure.
        """
        if not MT5_AVAILABLE:
            return None
        import MetaTrader5 as _mt5
        rates = _mt5.copy_rates_from_pos(self.symbol, _mt5.TIMEFRAME_M1, 1, n_bars)
        if rates is None or len(rates) == 0:
            logger.warning(f"copy_rates_from_pos returned empty: {_mt5.last_error()}")
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "tick_volume", "spread": "spread"})
        return df

    def get_tick(self) -> Optional[dict]:
        """Return current bid/ask/spread from symbol_info_tick."""
        if not MT5_AVAILABLE:
            return None
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None
        return {"bid": tick.bid, "ask": tick.ask, "spread": tick.ask - tick.bid,
                "time": pd.Timestamp(tick.time, unit="s", tz="UTC")}

    def check_staleness(self, last_bar_time: pd.Timestamp) -> bool:
        """True if data is fresh enough."""
        now = pd.Timestamp.now(tz="UTC")
        age = (now - last_bar_time).total_seconds()
        if age > self.cfg.live.max_data_staleness_seconds:
            logger.warning(f"Stale data: last bar is {age:.0f}s old.")
            return False
        return True

    def get_account_info(self) -> Optional[dict]:
        if not MT5_AVAILABLE:
            return None
        acc = mt5.account_info()
        if acc is None:
            return None
        return acc._asdict()

    def get_positions(self) -> List[dict]:
        if not MT5_AVAILABLE:
            return []
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
        return [p._asdict() for p in positions
                if p.magic == self.cfg.live.magic_number]

    def shutdown(self) -> None:
        if MT5_AVAILABLE and self._initialized:
            mt5.shutdown()
            logger.info("MT5 shutdown.")
