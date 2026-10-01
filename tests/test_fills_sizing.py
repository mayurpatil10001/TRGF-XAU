"""
tests/test_fills_sizing.py
Tests for fill prices, spread stress, and position sizing.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config, reset_config
from src.backtest.costs import entry_price_long, entry_price_short, net_pnl
from src.backtest.sizing import compute_lots


@pytest.fixture
def cfg():
    reset_config()
    return load_config("config/config.yaml")


class TestFills:
    def test_buy_uses_ask(self, cfg):
        """Long entry = BID + spread + slippage = ASK + slippage."""
        bid = 1850.0
        spread = cfg.costs.spread_price_units
        slip = cfg.costs.slippage_price_units
        entry = entry_price_long(bid, spread, slip)
        assert entry == pytest.approx(bid + spread + slip, abs=1e-6)

    def test_sell_uses_bid(self, cfg):
        """Short entry = BID - slippage (below mid)."""
        open_p = 1850.0
        slip = cfg.costs.slippage_price_units
        entry = entry_price_short(open_p, slip)
        assert entry == pytest.approx(open_p - slip, abs=1e-6)

    def test_commission_deducted(self, cfg):
        """Net PnL = Gross - Commission."""
        # 0.01 lot BUY: +1.0 price move
        pnl = net_pnl("BUY", 1800.0, 1801.0, 0.01, cfg)
        gross = 1.0 * 0.01 * cfg.symbol.contract_size   # = 1.0
        comm = cfg.costs.commission_per_lot_round_trip * 0.01  # = 0.07
        assert pnl == pytest.approx(gross - comm, abs=1e-4)


class TestSpreadStress:
    def test_doubling_spread_never_increases_pnl(self, cfg):
        """A trade with 2x spread should never have higher net PnL than 1x spread."""
        # Build a simple scenario: 1 lot long, +1.0 move
        # cost = spread + 2*slip + commission/contract_size
        # At 2x spread: cost is higher → net PnL must be lower
        from src.labels.triple_barrier import cost_round_trip
        cost_1x = cost_round_trip(cfg)

        # Manually compute: if we scale spread by 2
        import copy
        cfg2 = cfg.model_copy(deep=True)
        cfg2.costs.spread_price_units = cfg.costs.spread_price_units * 2.0
        from src.labels.triple_barrier import cost_round_trip as crt2
        cost_2x = crt2(cfg2)

        assert cost_2x > cost_1x, "2x spread must increase total cost."


class TestSizing:
    def test_lots_respect_step(self, cfg):
        """Computed lots must be a multiple of lot_step."""
        lots = compute_lots(10_000.0, 0.5, cfg, lot_step=0.01)
        remainder = round(lots / 0.01) * 0.01 - lots
        assert abs(remainder) < 1e-9, f"Lots {lots} not multiple of step 0.01"

    def test_lots_clamped_to_max(self, cfg):
        """Lots must not exceed lot_max."""
        lots = compute_lots(1_000_000.0, 0.01, cfg, lot_max=10.0, lot_step=0.01)
        assert lots <= 10.0

    def test_skip_when_min_lot_too_risky(self, cfg):
        """Return 0 when even minimum lot risks > 2*kappa of tiny equity."""
        lots = compute_lots(10.0, 10.0, cfg, lot_min=1.0)
        assert lots == 0.0, "Should return 0 when min lot is too risky."

    def test_risk_fraction_respected(self, cfg):
        """Risk per trade <= kappa * equity."""
        equity = 10_000.0
        sl_dist = 0.5
        lots = compute_lots(equity, sl_dist, cfg)
        risk_usd = lots * sl_dist * cfg.symbol.contract_size
        assert risk_usd <= equity * cfg.risk.risk_fraction_kappa * 1.01  # 1% tolerance
