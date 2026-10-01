"""
tests/test_profit_gate.py
Verify the gate cannot be bypassed and correctly PASSES/FAILS synthetic trade sets.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config, reset_config
from src.evaluation.profit_gate import (
    evaluate_gate, GATE_STATUS_PASSED, GATE_STATUS_FAILED
)


@pytest.fixture
def cfg():
    reset_config()
    return load_config("config/config.yaml")


def _make_trades(pnl_list, n_folds=4):
    n = len(pnl_list)
    fold_ids = np.array([i % n_folds for i in range(n)])
    return pd.DataFrame({
        "net_pnl": pnl_list,
        "fold_id": fold_ids,
    })


class TestProfitGate:
    def test_gate_fails_on_negative_pnl(self, cfg):
        """500 trades all losing → FAILED."""
        trades = _make_trades([-0.1] * 500)
        result = evaluate_gate(trades, 4, cfg, shuffle_sanity_passed=True)
        assert result["overall_status"] == GATE_STATUS_FAILED

    def test_gate_passes_on_clear_edge(self, cfg):
        """500 trades with large, consistent wins → PASSED."""
        # Win 60% at +2R, lose 40% at -1R → strongly profitable
        pnl = [2.0 if i % 10 < 6 else -1.0 for i in range(500)]
        trades = _make_trades(pnl, n_folds=5)
        result = evaluate_gate(trades, 5, cfg, shuffle_sanity_passed=True)
        assert result["overall_status"] == GATE_STATUS_PASSED

    def test_gate_fails_when_not_enough_trades(self, cfg):
        """Too few trades → min_oos_trades criterion fails → FAILED or PROVISIONAL."""
        trades = _make_trades([1.0] * 50)  # only 50 trades, need 300
        result = evaluate_gate(trades, 1, cfg, shuffle_sanity_passed=True)
        # Must not be PASSED with < min_oos_trades
        assert result["overall_status"] != GATE_STATUS_PASSED

    def test_gate_fails_on_shuffle_sanity(self, cfg):
        """If shuffle sanity fails, gate must FAIL regardless of other metrics."""
        pnl = [2.0 if i % 10 < 6 else -1.0 for i in range(500)]
        trades = _make_trades(pnl)
        result = evaluate_gate(trades, 4, cfg, shuffle_sanity_passed=False)
        assert result["overall_status"] == GATE_STATUS_FAILED
        # Find the shuffle criterion
        shuffle_c = next(c for c in result["criteria"] if c["id"] == 8)
        assert not shuffle_c["pass"]

    def test_gate_fails_below_stress_spread(self, cfg):
        """Net positive at 1x but negative at 1.5x → FAILED."""
        # Thin edge: +0.05 per trade, but stress subtracts 0.5*(spread) additional cost
        # Use very thin margin
        thin_pnl = [0.01] * 600
        trades = pd.DataFrame({
            "net_pnl": thin_pnl,
            "net_pnl_stress": [-0.05] * 600,  # negative at stress spread
            "fold_id": [i % 5 for i in range(600)],
        })
        result = evaluate_gate(trades, 5, cfg, shuffle_sanity_passed=True)
        stress_c = next(c for c in result["criteria"] if c["id"] == 4)
        assert not stress_c["pass"]
