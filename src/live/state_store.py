"""
src/live/state_store.py
SQLite state store for audit trail and idempotency.

Tables:
  decisions(ts, bar_time, features_hash, p_long, p_short, p_none, action, reason, spread, model_version)
  orders(ts, ticket, side, lots, price, sl, tp, retcode, comment)
  trades(ticket, open_time, close_time, side, lots, open_price, close_price, profit, commission, swap, exit_reason)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column, Float, Integer, String, Text, create_engine, DateTime
)
from sqlalchemy.orm import DeclarativeBase, Session

from src.common.logging import get_logger

logger = get_logger("live.state_store")


class Base(DeclarativeBase):
    pass


class DecisionRecord(Base):
    __tablename__ = "decisions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(String, nullable=False)
    bar_time = Column(String, nullable=False)
    features_hash = Column(String)
    p_long = Column(Float)
    p_short = Column(Float)
    p_none = Column(Float)
    action = Column(String)
    reason = Column(String)
    spread = Column(Float)
    model_version = Column(String)


class OrderRecord(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(String, nullable=False)
    ticket = Column(Integer)
    side = Column(String)
    lots = Column(Float)
    price = Column(Float)
    sl = Column(Float)
    tp = Column(Float)
    retcode = Column(Integer)
    comment = Column(String)


class TradeRecord(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket = Column(Integer)
    open_time = Column(String)
    close_time = Column(String)
    side = Column(String)
    lots = Column(Float)
    open_price = Column(Float)
    close_price = Column(Float)
    profit = Column(Float)
    commission = Column(Float)
    swap = Column(Float)
    exit_reason = Column(String)


class StateStore:
    def __init__(self, db_path: str = "data/bot_state.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        logger.info(f"State store initialized: {db_path}")

    def log_decision(self, **kwargs) -> None:
        with Session(self.engine) as s:
            s.add(DecisionRecord(
                ts=datetime.now(timezone.utc).isoformat(), **kwargs
            ))
            s.commit()

    def log_order(self, **kwargs) -> None:
        with Session(self.engine) as s:
            s.add(OrderRecord(
                ts=datetime.now(timezone.utc).isoformat(), **kwargs
            ))
            s.commit()

    def log_trade(self, **kwargs) -> None:
        with Session(self.engine) as s:
            s.add(TradeRecord(**kwargs))
            s.commit()

    def get_last_bar_time(self) -> Optional[str]:
        with Session(self.engine) as s:
            row = s.query(DecisionRecord).order_by(DecisionRecord.id.desc()).first()
            return row.bar_time if row else None
