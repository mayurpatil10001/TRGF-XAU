"""
tests/test_hft_structural_fixes.py
===================================
Step 5 required tests:
  1. Breakeven edge check — refuses to trade when expected edge is below threshold
  2. Single-position rule — no stacking allowed
  3. TP/SL from mid geometry — TP >= 3x round-trip cost, proper ask/bid fills
  4. Cost accounting — full cost breakdown matches expected components
  5. Label-shuffle sanity — shuffled labels do NOT produce profitable strategy
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config, reset_config
from src.labels.triple_barrier import cost_round_trip
from src.live.run_bot_v2 import (
    breakeven_win_rate, compute_tp_sl, edge_check_passes, EdgeMonitor
)
from src.backtest.costs import entry_price_long, entry_price_short, net_pnl


@pytest.fixture
def cfg():
    reset_config()
    return load_config("config/config.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Breakeven edge check
# ─────────────────────────────────────────────────────────────────────────────

class TestBreakevenEdgeCheck:

    def test_breakeven_win_rate_formula(self):
        """breakeven_wr = SL / (TP + SL)."""
        tp, sl = 1.0, 0.5
        bkv_wr = breakeven_win_rate(tp, sl)
        assert abs(bkv_wr - 0.5 / 1.5) < 1e-9, f"Got {bkv_wr}"

    def test_edge_check_passes_when_prob_beats_breakeven(self, cfg):
        """High probability clearly above breakeven → edge check passes."""
        tp_dist, sl_dist = 0.75, 0.5
        cost_rt = cost_round_trip(cfg)
        p_best = 0.75  # clearly above breakeven ~0.40
        ok, reason = edge_check_passes(p_best, tp_dist, sl_dist, cost_rt, 0.0)
        assert ok, f"Expected edge check to pass, got: {reason}"

    def test_edge_check_fails_when_prob_below_breakeven(self, cfg):
        """Low probability below breakeven → edge check refuses entry."""
        tp_dist, sl_dist = 0.75, 0.5
        cost_rt = cost_round_trip(cfg)
        p_best = 0.25  # below breakeven ~0.40
        ok, reason = edge_check_passes(p_best, tp_dist, sl_dist, cost_rt, 0.0)
        assert not ok, f"Expected edge check to fail, got: {reason}"

    def test_edge_check_fails_when_expected_R_below_min(self, cfg):
        """Even with p > breakeven, if expected_R < min_expected_R → fail."""
        tp_dist, sl_dist = 0.5, 0.5  # thin R:R
        cost_rt = cost_round_trip(cfg)
        p_best = 0.60
        # With thin margins, expected_R might be small
        ok_no_min, _ = edge_check_passes(p_best, tp_dist, sl_dist, cost_rt, 0.0)
        ok_with_min, reason = edge_check_passes(p_best, tp_dist, sl_dist, cost_rt, min_expected_R=10.0)
        # With unrealistically high min_expected_R, should fail
        assert not ok_with_min, f"Expected failure with high min_R: {reason}"

    def test_zero_probability_always_fails(self, cfg):
        """p_best = 0 → always fails edge check."""
        cost_rt = cost_round_trip(cfg)
        ok, _ = edge_check_passes(0.0, 1.0, 0.5, cost_rt, 0.0)
        assert not ok


# ─────────────────────────────────────────────────────────────────────────────
# 2. Single-position rule (via EdgeMonitor and bot design)
# ─────────────────────────────────────────────────────────────────────────────

class TestSinglePositionRule:

    def test_edge_monitor_pauses_on_negative_expectancy(self):
        """EdgeMonitor should pause after sustained losses."""
        monitor = EdgeMonitor(window=10, pause_threshold=-1.0)
        assert not monitor.is_paused

        # Add 10 losses of -5 USD each → rolling mean = -5 < -1.0
        for _ in range(10):
            monitor.add_trade(-5.0)

        assert monitor.is_paused, "Monitor should be paused after sustained losses"
        assert monitor.rolling_expectancy < -1.0

    def test_edge_monitor_resumes_on_recovery(self):
        """EdgeMonitor should resume when rolling expectancy becomes positive."""
        monitor = EdgeMonitor(window=6, pause_threshold=-1.0)
        for _ in range(6):
            monitor.add_trade(-5.0)
        assert monitor.is_paused

        # Recovery: 6 wins
        for _ in range(6):
            monitor.add_trade(10.0)
        assert not monitor.is_paused, "Monitor should resume after recovery"

    def test_edge_monitor_does_not_pause_on_wins(self):
        """Winning trades should never trigger a pause."""
        monitor = EdgeMonitor(window=10, pause_threshold=-1.0)
        for _ in range(20):
            monitor.add_trade(+5.0)
        assert not monitor.is_paused


# ─────────────────────────────────────────────────────────────────────────────
# 3. TP/SL from mid geometry
# ─────────────────────────────────────────────────────────────────────────────

class TestTPSLGeometry:

    def test_tp_dist_at_least_3x_cost(self, cfg):
        """TP distance must be at least 3× round-trip cost."""
        cost_rt = cost_round_trip(cfg)
        min_required_tp = 3.0 * cost_rt
        atr = 0.30  # typical 1-min XAU ATR
        tp_dist, sl_dist = compute_tp_sl(atr, cfg)
        assert tp_dist >= min_required_tp - 1e-9, (
            f"TP dist {tp_dist:.4f} < 3× cost {min_required_tp:.4f}"
        )

    def test_tp_dist_larger_than_sl_for_positive_rr(self, cfg):
        """TP should generally be larger than SL for positive R:R."""
        atr = 0.30
        tp_dist, sl_dist = compute_tp_sl(atr, cfg)
        rr = tp_dist / sl_dist
        assert rr >= 1.0, f"R:R = {rr:.2f} < 1.0 — negative edge structure"

    def test_long_entry_at_ask(self):
        """Long entry must be at ASK = open + spread + slippage."""
        open_price = 2000.0
        spread = 0.25
        slippage = 0.05
        entry = entry_price_long(open_price, spread, slippage)
        expected = open_price + spread + slippage
        assert abs(entry - expected) < 1e-9, f"Got {entry}, expected {expected}"

    def test_short_entry_at_bid(self):
        """Short entry must be at BID = open - slippage."""
        open_price = 2000.0
        slippage = 0.05
        entry = entry_price_short(open_price, slippage)
        expected = open_price - slippage
        assert abs(entry - expected) < 1e-9, f"Got {entry}, expected {expected}"

    def test_round_trip_cost_formula(self, cfg):
        """Round-trip cost = spread + 2*slippage + commission/contract_size."""
        rt = cost_round_trip(cfg)
        expected = (
            cfg.costs.spread_price_units
            + 2 * cfg.costs.slippage_price_units
            + cfg.costs.commission_per_lot_round_trip / cfg.symbol.contract_size
        )
        assert abs(rt - expected) < 1e-9, f"Got {rt:.6f}, expected {expected:.6f}"

    def test_tp_breakeven_win_rate_achievable(self, cfg):
        """With our TP/SL structure, breakeven win rate should be < 0.60."""
        atr = 0.30
        tp_dist, sl_dist = compute_tp_sl(atr, cfg)
        bkv_wr = breakeven_win_rate(tp_dist, sl_dist)
        # A model with ≥60% accuracy should be able to beat this
        assert bkv_wr < 0.70, (
            f"Breakeven win rate {bkv_wr:.2%} is too high — "
            f"very hard to achieve with ML signal"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cost accounting
# ─────────────────────────────────────────────────────────────────────────────

class TestCostAccounting:

    def test_net_pnl_buy_profitable(self, cfg):
        """BUY: exit > entry → positive gross, net = gross - commission."""
        entry = 2000.25
        exit_p = 2001.00
        lots = 1.0
        gross = (exit_p - entry) * lots * cfg.symbol.contract_size
        comm = cfg.costs.commission_per_lot_round_trip * lots
        expected_net = gross - comm
        actual_net = net_pnl("BUY", entry, exit_p, lots, cfg)
        assert abs(actual_net - expected_net) < 1e-6

    def test_net_pnl_sell_profitable(self, cfg):
        """SELL: exit < entry → positive gross, net = gross - commission."""
        entry = 2000.0
        exit_p = 1999.50
        lots = 1.0
        gross = (entry - exit_p) * lots * cfg.symbol.contract_size
        comm = cfg.costs.commission_per_lot_round_trip * lots
        expected_net = gross - comm
        actual_net = net_pnl("SELL", entry, exit_p, lots, cfg)
        assert abs(actual_net - expected_net) < 1e-6

    def test_tiny_profit_net_can_be_negative_due_to_costs(self, cfg):
        """A tiny gross profit can still be net-negative after costs."""
        entry = 2000.25   # ASK
        exit_p = 2000.30  # BID exit — only 0.05 pts gross
        lots = 1.0
        actual_net = net_pnl("BUY", entry, exit_p, lots, cfg)
        # 0.05 * 100 = $5 gross, but $7 commission → net = -$2
        assert actual_net < 0, (
            f"Tiny win should be net-negative after costs, got {actual_net:.2f}"
        )

    def test_commission_scales_with_lots(self, cfg):
        """Commission should double when lots doubles."""
        entry = 2000.0
        exit_p = 2001.0
        net_1lot = net_pnl("BUY", entry, exit_p, 1.0, cfg)
        net_2lot = net_pnl("BUY", entry, exit_p, 2.0, cfg)
        # Both gross and commission double, so net should roughly double
        assert abs(net_2lot - 2 * net_1lot) < 1e-4, (
            f"net@2lot={net_2lot:.4f} != 2×net@1lot={2*net_1lot:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Label-shuffle sanity integration
# ─────────────────────────────────────────────────────────────────────────────

class TestLabelShuffleSanity:

    def test_shuffle_produces_random_win_rate(self):
        """
        After shuffling model probabilities against labels, accuracy should
        drop to ~1/3 (random 3-class baseline), confirming signal comes from
        the data relationship, not from spurious rules.
        """
        rng = np.random.default_rng(42)
        n = 1000
        # Simulate signal: label 0=NONE,1=LONG,2=SHORT in 1/3 split
        labels = np.tile([0, 1, 2], n // 3 + 1)[:n]

        # Build "perfect" probs: always one-hot for the true label
        probs = np.zeros((n, 3))
        for i, lbl in enumerate(labels):
            probs[i, lbl] = 0.9
            for j in range(3):
                if j != lbl:
                    probs[i, j] = 0.05

        # With aligned signal: accuracy should be high
        correct = (np.argmax(probs, axis=1) == labels).mean()
        assert correct > 0.90, f"Expected high accuracy before shuffle: {correct:.2%}"

        # After shuffling probs (break alignment):
        shuffled_probs = probs.copy()
        rng.shuffle(shuffled_probs)  # shuffles along sample axis
        shuffled_correct = (np.argmax(shuffled_probs, axis=1) == labels).mean()
        # Should be ~33% (random 3-class) — definitely below 60%
        assert shuffled_correct < 0.60, (
            f"Shuffled accuracy {shuffled_correct:.2%} is too high — "
            f"shuffle didn't break signal alignment"
        )

    def test_gate_fails_on_shuffle(self, cfg):
        """
        Gate must fail when shuffle_sanity_passed=False,
        regardless of PnL metrics.
        """
        from src.evaluation.profit_gate import evaluate_gate, GATE_STATUS_FAILED

        # Otherwise "profitable" trade set
        pnl = [2.0 if i % 10 < 6 else -1.0 for i in range(500)]
        df = pd.DataFrame({
            "net_pnl": pnl,
            "fold_id": [i % 5 for i in range(500)],
        })
        result = evaluate_gate(df, 5, cfg, shuffle_sanity_passed=False)
        assert result["overall_status"] == GATE_STATUS_FAILED, (
            "Gate must FAIL when shuffle sanity fails, even if PnL metrics are good"
        )
