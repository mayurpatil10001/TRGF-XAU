"""
src/api/main.py
FastAPI application: REST endpoints + WebSocket + MT5 poller background task.

Endpoints:
  GET  /api/health
  GET  /api/trades
  GET  /api/equity_curve
  GET  /api/metrics
  GET  /api/backtest_summary
  GET  /api/profit_gate
  POST /api/kill_switch      (requires X-API-Token header)
  POST /api/pause            (requires X-API-Token header)
  POST /api/resume
  WS   /ws/live

Run with:
  uvicorn src.api.main:app --port 8000 --reload
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, WebSocket
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.db import append_equity_point, get_equity_curve
from src.api.mt5_poller import get_snapshot, poll_mt5
from src.api.schemas import LiveSnapshot
from src.api.ws import ws_endpoint
from src.common.config import load_config
from src.common.logging import setup_logging

cfg = load_config()
setup_logging(cfg.logging.level, cfg.logging.log_dir)

API_TOKEN = os.getenv("API_SECRET_TOKEN", "changeme")
DASHBOARD_ORIGIN = os.getenv("DASHBOARD_ORIGIN", "http://localhost:5173")

_paused = False
_killed = False


async def _equity_logger():
    """Persist equity curve points every 10 seconds."""
    import asyncio
    while True:
        snap = get_snapshot()
        if snap.equity > 0:
            append_equity_point(
                ts=snap.ts,
                equity=snap.equity,
                balance=snap.balance,
                realized_pnl=snap.realized_pnl_total,
            )
        await asyncio.sleep(10.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    t1 = asyncio.create_task(poll_mt5(cfg))
    t2 = asyncio.create_task(_equity_logger())
    yield
    t1.cancel()
    t2.cancel()


app = FastAPI(title="TEGF-XAU API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[DASHBOARD_ORIGIN, "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_token(x_api_token: Optional[str] = Header(None)):
    if x_api_token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API token")


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    snap = get_snapshot()
    return {
        "status": "ok",
        "mt5_connected": snap.mt5_connected,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/profit_gate")
def profit_gate():
    path = Path("reports/profit_gate.json")
    if not path.exists():
        return {"status": "NOT_EVALUATED"}
    with open(path) as f:
        return json.load(f)


@app.get("/api/backtest_summary")
def backtest_summary():
    path = Path("reports/backtest_summary.json")
    if not path.exists():
        return {"error": "not_available"}
    with open(path) as f:
        return json.load(f)


@app.get("/api/equity_curve")
def equity_curve(limit: int = 5000):
    return get_equity_curve(limit=limit)


@app.get("/api/metrics")
def live_metrics():
    snap = get_snapshot()
    return {
        "equity": snap.equity,
        "balance": snap.balance,
        "floating_pnl": snap.floating_pnl,
        "realized_pnl_today": snap.realized_pnl_today,
        "realized_pnl_total": snap.realized_pnl_total,
        "n_open_positions": len(snap.open_positions),
        "market": snap.market.model_dump(),
        "bot_status": snap.bot.status,
        "gate_status": snap.profit_gate.status,
    }


@app.get("/api/trades")
def trades(
    from_ts: Optional[str] = None,
    to_ts: Optional[str] = None,
    side: Optional[str] = None,
):
    trades_path = Path("data/processed/oos_trades.csv")
    if not trades_path.exists():
        return []
    import pandas as pd
    df = pd.read_csv(trades_path)
    if from_ts:
        df = df[df["entry_time"] >= from_ts]
    if to_ts:
        df = df[df["entry_time"] <= to_ts]
    if side:
        df = df[df["side"] == side.upper()]
    return df.to_dict(orient="records")


@app.post("/api/kill_switch")
def kill_switch(x_api_token: Optional[str] = Header(None)):
    _require_token(x_api_token)
    global _killed
    _killed = True
    # Create kill file
    Path(cfg.live.kill_switch_file).touch()
    return {"status": "kill_switch_activated"}


@app.post("/api/pause")
def pause(x_api_token: Optional[str] = Header(None)):
    _require_token(x_api_token)
    global _paused
    _paused = True
    return {"status": "paused"}


@app.post("/api/resume")
def resume(x_api_token: Optional[str] = Header(None)):
    _require_token(x_api_token)
    global _paused
    _paused = False
    return {"status": "resumed"}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await ws_endpoint(websocket)
