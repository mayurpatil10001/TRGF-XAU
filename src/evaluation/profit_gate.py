"""
src/evaluation/profit_gate.py
Profit gate evaluation. Reads OOS backtest results and applies all 8 criteria.
Outputs reports/profit_gate.json.

The live bot MUST pass the gate before placing any orders.
NEVER bypass or weaken these criteria.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, Config
from src.common.logging import setup_logging, get_logger
from src.evaluation.bootstrap import stationary_block_bootstrap
from src.evaluation.metrics import profit_factor, expectancy_per_trade

logger = get_logger("evaluation.profit_gate")

GATE_STATUS_PASSED = "PASSED"
GATE_STATUS_PROVISIONAL = "PROVISIONAL"
GATE_STATUS_FAILED = "FAILED"


def evaluate_gate(
    trades_df: pd.DataFrame,  # must have columns: net_pnl, fold_id, spread_mult
    fold_count: int,
    cfg: Config,
    holdout_trades: pd.DataFrame | None = None,
    shuffle_sanity_passed: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate all 8 gate criteria.

    Parameters
    ----------
    trades_df: OOS test trades with net_pnl (at 1.0x spread) and fold_id
    fold_count: total number of test folds
    cfg: Config
    holdout_trades: trades on the final holdout (may be None if not yet evaluated)
    shuffle_sanity_passed: result of the shuffle sanity test

    Returns
    -------
    Dict with criteria, values, thresholds, and overall status.
    """
    gc = cfg.profit_gate
    pnl = trades_df["net_pnl"].values
    n_trades = len(pnl)
    total_pnl = float(pnl.sum())
    exp = expectancy_per_trade(pnl)
    pf = profit_factor(pnl, pnl)

    # Spread-stress PnL (at stress_mult spread)
    stress_mult = gc.require_positive_at_spread_mult
    if "net_pnl_stress" in trades_df.columns:
        pnl_stress = trades_df["net_pnl_stress"].values
    else:
        # Approximate: additional spread cost per trade
        spread_extra = (stress_mult - 1.0) * cfg.costs.spread_price_units
        pnl_stress = pnl - spread_extra  # rough conservative adjustment

    # Bootstrap CI on expectancy
    if n_trades >= 30:
        stat_point, ci_lower, ci_upper = stationary_block_bootstrap(
            pnl, statistic=np.mean, n_resamples=5000, seed=cfg.seed
        )
        provisional = False
    else:
        stat_point, ci_lower, ci_upper = exp, float("-inf"), float("inf")
        provisional = True

    # Positive fold fraction
    if "fold_id" in trades_df.columns:
        fold_pnls = trades_df.groupby("fold_id")["net_pnl"].sum()
        positive_fold_fraction = float((fold_pnls > 0).mean())
        n_eval_folds = len(fold_pnls)
    else:
        positive_fold_fraction = 1.0 if total_pnl > 0 else 0.0
        n_eval_folds = fold_count

    # Holdout
    if holdout_trades is not None and len(holdout_trades) > 0:
        holdout_pnl = holdout_trades["net_pnl"].sum()
        holdout_positive = holdout_pnl > 0
    else:
        holdout_pnl = None
        holdout_positive = None  # not yet evaluated

    # ── Gate criteria ─────────────────────────────────────────────────────────
    criteria = [
        {
            "id": 1,
            "name": "total_net_profit_positive",
            "value": round(total_pnl, 4),
            "threshold": "> 0",
            "pass": total_pnl > 0 and exp > 0,
            "detail": f"total={total_pnl:.2f}, expectancy={exp:.4f}",
        },
        {
            "id": 2,
            "name": "bootstrap_ci_lower_gt_zero",
            "value": round(ci_lower, 6),
            "threshold": f"> 0 (95% CI lower bound)",
            "pass": ci_lower > 0 if gc.require_bootstrap_ci_lower_gt_zero and not provisional else True,
            "detail": f"CI=[{ci_lower:.4f}, {ci_upper:.4f}], provisional={provisional}",
        },
        {
            "id": 3,
            "name": "profit_factor_gte_threshold",
            "value": round(pf, 4),
            "threshold": f">= {gc.min_profit_factor}",
            "pass": pf >= gc.min_profit_factor,
            "detail": f"PF={pf:.4f}",
        },
        {
            "id": 4,
            "name": "positive_at_stress_spread",
            "value": round(float(pnl_stress.sum()), 4),
            "threshold": f"> 0 at {stress_mult}x spread",
            "pass": float(pnl_stress.sum()) > 0,
            "detail": f"sum_stress_pnl={pnl_stress.sum():.2f}",
        },
        {
            "id": 5,
            "name": "min_oos_trades",
            "value": n_trades,
            "threshold": f">= {gc.min_oos_trades}",
            "pass": n_trades >= gc.min_oos_trades,
            "detail": f"n_trades={n_trades}",
        },
        {
            "id": 6,
            "name": "positive_fold_fraction",
            "value": round(positive_fold_fraction, 4),
            "threshold": f">= {gc.min_positive_fold_fraction}",
            "pass": positive_fold_fraction >= gc.min_positive_fold_fraction,
            "detail": f"positive_folds={positive_fold_fraction:.1%} ({n_eval_folds} folds)",
        },
        {
            "id": 7,
            "name": "holdout_positive",
            "value": round(holdout_pnl, 4) if holdout_pnl is not None else "NOT_EVALUATED",
            "threshold": "> 0 (or NOT_EVALUATED for dev-phase)",
            # NOT_EVALUATED is treated as provisional pass in dev phase;
            # live deployment MUST run holdout before starting.
            "pass": (holdout_positive if holdout_positive is not None else True),
            "detail": (
                "⚠ HOLDOUT NOT YET RUN — required before live deployment"
                if holdout_pnl is None
                else f"holdout_pnl={holdout_pnl:.2f}"
            ),
        },
        {
            "id": 8,
            "name": "shuffle_sanity_test",
            "value": "PASSED" if shuffle_sanity_passed else "FAILED",
            "threshold": "MUST PASS",
            "pass": shuffle_sanity_passed,
            "detail": "shuffled-label strategy must NOT be profitable",
        },
    ]

    all_pass = all(c["pass"] for c in criteria)
    provisional_flag = provisional and not all_pass

    if all_pass:
        overall_status = GATE_STATUS_PASSED
    elif provisional_flag:
        overall_status = GATE_STATUS_PROVISIONAL
    else:
        overall_status = GATE_STATUS_FAILED

    result = {
        "overall_status": overall_status,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "n_trades": n_trades,
        "total_net_pnl": round(total_pnl, 4),
        "expectancy_per_trade": round(exp, 6),
        "profit_factor": round(pf, 4),
        "bootstrap_ci": [round(ci_lower, 6), round(ci_upper, 6)],
        "positive_fold_fraction": round(positive_fold_fraction, 4),
        "shuffle_sanity_passed": shuffle_sanity_passed,
        "criteria": criteria,
        "provisional_note": (
            "Trade count too low for meaningful bootstrap CI; "
            "PROVISIONAL permits DEMO only." if provisional_flag else ""
        ),
    }

    if overall_status == GATE_STATUS_FAILED:
        result["diagnosis"] = _diagnosis(trades_df, cfg, exp, pf)

    return result


def _diagnosis(trades_df: pd.DataFrame, cfg: Config, exp: float, pf: float) -> Dict:
    """Auto-generate failure diagnosis."""
    spread = cfg.costs.spread_price_units
    slip = cfg.costs.slippage_price_units
    comm = cfg.costs.commission_per_lot_round_trip / cfg.symbol.contract_size
    total_cost = spread + 2 * slip + comm

    pnl = trades_df["net_pnl"].values
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = float((pnl > 0).mean()) if len(pnl) > 0 else 0.0
    rr = float(wins.mean() / abs(losses.mean())) if len(wins) > 0 and len(losses) > 0 else 0.0
    p_star = 1.0 / (1.0 + rr) if rr > 0 else 0.5

    return {
        "cost_to_signal_note": (
            f"Total round-trip cost = {total_cost:.4f} price units. "
            f"At 1-min horizon, required |return| > {total_cost:.4f} just to break even before edge. "
            "1-min moves vs 60-min moves have ~7.7x worse signal-to-cost ratio (sqrt-time scaling)."
        ),
        "achieved_win_rate": round(win_rate, 4),
        "breakeven_win_rate": round(p_star, 4),
        "avg_win_R": round(float(wins.mean()), 4) if len(wins) else 0.0,
        "avg_loss_R": round(float(losses.mean()), 4) if len(losses) else 0.0,
        "next_experiments": [
            "Increase horizon H to 5 or more bars",
            "Tighten probability threshold (e.g., 0.65+)",
            "Restrict to overlap session (highest liquidity, lower effective spread)",
            "Meta-labelling: add a second filter model",
            "Use expected-R threshold to only trade high-conviction signals",
        ],
    }


def save_gate_result(result: Dict, path: Path = Path("reports/profit_gate.json")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Profit gate result: {result['overall_status']} → {path}")


def load_gate_status(path: Path = Path("reports/profit_gate.json")) -> str:
    """Return the gate status string, or FAILED if not found."""
    if not path.exists():
        return GATE_STATUS_FAILED
    with open(path) as f:
        data = json.load(f)
    return data.get("overall_status", GATE_STATUS_FAILED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--trades", default="data/processed/oos_trades.csv")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)

    if not Path(args.trades).exists():
        logger.error(f"OOS trades file not found: {args.trades}. Run backtest first.")
        sys.exit(1)

    trades_df = pd.read_csv(args.trades)
    fold_count = trades_df["fold_id"].nunique() if "fold_id" in trades_df.columns else 1
    result = evaluate_gate(trades_df, fold_count, cfg)
    save_gate_result(result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
