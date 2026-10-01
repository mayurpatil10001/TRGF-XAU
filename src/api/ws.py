"""
src/api/ws.py
WebSocket handler: pushes LiveSnapshot to all connected clients every second.
Stale detection: client should timeout if no message in > 5s.
"""
from __future__ import annotations

import asyncio
import json
from typing import Set

from fastapi import WebSocket, WebSocketDisconnect

from src.api.mt5_poller import get_snapshot
from src.common.logging import get_logger

logger = get_logger("api.ws")

_connections: Set[WebSocket] = set()


async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _connections.add(websocket)
    logger.info(f"WebSocket client connected. Total: {len(_connections)}")
    try:
        while True:
            snapshot = get_snapshot()
            payload = snapshot.model_dump_json()
            await websocket.send_text(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
    finally:
        _connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(_connections)}")
