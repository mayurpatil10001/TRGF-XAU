"""
src/api/db.py
API database helpers for trade history and equity curve persistence.
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import Column, Float, Integer, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session

DB_PATH = "data/api.db"


class Base(DeclarativeBase):
    pass


class EquityPoint(Base):
    __tablename__ = "equity_curve"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(String, nullable=False)
    equity = Column(Float)
    balance = Column(Float)
    realized_pnl = Column(Float)


def get_engine():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Base.metadata.create_all(engine, checkfirst=True)
    return engine


_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def append_equity_point(ts: str, equity: float, balance: float, realized_pnl: float) -> None:
    with Session(engine()) as s:
        s.add(EquityPoint(ts=ts, equity=equity, balance=balance, realized_pnl=realized_pnl))
        s.commit()


def get_equity_curve(limit: int = 10_000) -> list:
    with Session(engine()) as s:
        rows = s.query(EquityPoint).order_by(EquityPoint.id.desc()).limit(limit).all()
        return [{"ts": r.ts, "equity": r.equity, "balance": r.balance,
                 "realized_pnl": r.realized_pnl} for r in reversed(rows)]
