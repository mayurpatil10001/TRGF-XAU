"""
src/execution/order_manager.py
Order placement, management, and SL/TP attachment.

Rules:
- order_check() before order_send()
- Retry only on requote/price-changed (max 2 retries, fresh price each)
- Log full retcode + comment
- No infinite retries on rejection
- Filling mode read from symbol_info
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import pandas as pd

from src.common.config import Config
from src.common.logging import get_logger

logger = get_logger("execution.order_manager")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

# Retcodes that warrant a retry
RETRIABLE_RETCODES = {10004, 10006}  # REQUOTE, PRICE_CHANGED


class OrderManager:
    def __init__(self, cfg: Config, connector):
        self.cfg = cfg
        self.connector = connector

    def _get_filling_mode(self) -> int:
        """Determine the filling mode from symbol info.
        filling_mode bits: FOK=1, IOC=2; use ORDER_FILLING_* for request type field.
        """
        if not MT5_AVAILABLE:
            return 0
        info = self.connector.symbol_info
        if info is None:
            return mt5.ORDER_FILLING_IOC
        filling = info.filling_mode
        # Bit 0 = FOK supported, Bit 1 = IOC supported
        if filling & 1:   # FOK
            return mt5.ORDER_FILLING_FOK
        if filling & 2:   # IOC
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def _round_price(self, price: float) -> float:
        if not MT5_AVAILABLE:
            return round(price, 2)
        info = self.connector.symbol_info
        if info is None:
            return round(price, 2)
        return round(price, info.digits)

    def _check_stops_level(self, entry: float, sl: float, tp: float) -> Tuple[float, float]:
        """Ensure SL/TP respect minimum distance from price."""
        if not MT5_AVAILABLE:
            return sl, tp
        info = self.connector.symbol_info
        if info is None:
            return sl, tp
        min_dist = info.trade_stops_level * info.point
        if abs(entry - sl) < min_dist:
            sl = entry - min_dist if entry > sl else entry + min_dist
        if abs(entry - tp) < min_dist:
            tp = entry + min_dist if tp > entry else entry - min_dist
        return self._round_price(sl), self._round_price(tp)

    def send_order(
        self,
        side: str,         # "BUY" | "SELL"
        volume: float,
        price: float,
        sl: float,
        tp: float,
        comment: str = "tegf_xau",
    ) -> Optional[dict]:
        """
        Send a market order with SL and TP.
        Returns order result dict or None on failure.
        """
        if not MT5_AVAILABLE:
            logger.warning("[MOCK] order_send called (MT5 not available)")
            return {"retcode": 10009, "order": 0, "comment": "mock_success"}

        sl, tp = self._check_stops_level(price, sl, tp)
        order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.connector.symbol,
            "volume": volume,
            "type": order_type,
            "price": self._round_price(price),
            "sl": self._round_price(sl),
            "tp": self._round_price(tp),
            "deviation": self.cfg.live.deviation_points,
            "magic": self.cfg.live.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(),
        }

        # Pre-check
        check = mt5.order_check(request)
        if check is None or check.retcode != 0:
            logger.warning(f"order_check failed: {check}")
            return None

        # Send with retry on requote/price-changed
        for attempt in range(3):
            result = mt5.order_send(request)
            if result is None:
                logger.error(f"order_send returned None: {mt5.last_error()}")
                break
            retcode = result.retcode
            comment_str = result.comment
            logger.info(f"order_send attempt {attempt+1}: retcode={retcode} comment={comment_str}")

            if retcode == 10009:  # DONE
                logger.info(f"Order placed: ticket={result.order} side={side} lots={volume}")
                return result._asdict()
            elif retcode in RETRIABLE_RETCODES:
                tick = self.connector.get_tick()
                if tick:
                    price = tick["ask"] if side == "BUY" else tick["bid"]
                    request["price"] = self._round_price(price)
                time.sleep(0.2)
            else:
                logger.error(f"Order rejected: retcode={retcode} comment={comment_str}")
                break

        return None

    def close_position(self, position: dict) -> Optional[dict]:
        """Close an open position by ticket."""
        if not MT5_AVAILABLE:
            return {"retcode": 10009, "comment": "mock_close"}

        ticket = position["ticket"]
        side = position["type"]  # 0=BUY, 1=SELL
        volume = position["volume"]
        symbol = position["symbol"]

        close_type = mt5.ORDER_TYPE_SELL if side == 0 else mt5.ORDER_TYPE_BUY
        tick = self.connector.get_tick()
        price = (tick["bid"] if side == 0 else tick["ask"]) if tick else 0.0

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "position": ticket,
            "price": self._round_price(price),
            "deviation": self.cfg.live.deviation_points,
            "magic": self.cfg.live.magic_number,
            "comment": "tegf_xau_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(),
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Close position failed: {mt5.last_error()}")
            return None
        logger.info(f"Position closed: ticket={ticket} retcode={result.retcode}")
        return result._asdict()
