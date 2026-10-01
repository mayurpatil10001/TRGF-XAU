"""
tests/test_api_schema.py
Verify WebSocket payload matches the Pydantic schema.
"""
import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.schemas import LiveSnapshot


def test_default_snapshot_serializes():
    """Default snapshot serializes/deserializes correctly."""
    snap = LiveSnapshot()
    raw = snap.model_dump_json()
    data = json.loads(raw)

    # Required top-level keys
    required_keys = [
        "ts", "seq", "mt5_connected", "account", "balance", "equity",
        "margin", "free_margin", "floating_pnl", "realized_pnl_today",
        "realized_pnl_total", "open_positions", "market", "last_signal",
        "bot", "profit_gate",
    ]
    for key in required_keys:
        assert key in data, f"Missing key in WebSocket schema: {key}"


def test_snapshot_round_trip():
    """A snapshot serializes and deserializes without loss."""
    snap = LiveSnapshot(
        seq=42,
        balance=10500.0,
        equity=10450.0,
        floating_pnl=-50.0,
        realized_pnl_today=200.0,
    )
    raw = snap.model_dump_json()
    snap2 = LiveSnapshot.model_validate_json(raw)
    assert snap2.seq == 42
    assert snap2.balance == 10500.0
    assert snap2.floating_pnl == -50.0


def test_open_positions_list():
    """open_positions must be a list."""
    snap = LiveSnapshot()
    data = json.loads(snap.model_dump_json())
    assert isinstance(data["open_positions"], list)


def test_profit_gate_field():
    """profit_gate must have status and updated fields."""
    snap = LiveSnapshot()
    data = json.loads(snap.model_dump_json())
    assert "status" in data["profit_gate"]
    assert "updated" in data["profit_gate"]
