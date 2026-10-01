"""
scripts/compare_timing_true_vs_proxy.py
Validates how well the OHLC timing proxy matches true tick-level timing.
Requires MT5 connection and tick data availability.

Run: python scripts/compare_timing_true_vs_proxy.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config
from src.common.logging import setup_logging, get_logger
from src.features.timing import timing_proxy, compute_true_timing
from src.execution.mt5_connector import MT5Connector

logger = get_logger("scripts.compare_timing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--bars", type=int, default=200, help="Number of recent bars to compare")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)

    connector = MT5Connector(cfg)
    if not connector.initialize() or not connector.login():
        logger.error("MT5 connection failed. Cannot compute true timing.")
        return
    if not connector.resolve_symbol():
        return

    bars = connector.get_rates(n_bars=args.bars)
    if bars is None:
        logger.error("No bars returned.")
        return

    bars["time"] = pd.to_datetime(bars["time"], utc=True)
    bars["spread"] = cfg.costs.spread_price_units
    bars["gap_before_minutes"] = 0.0
    bars["tick_volume"] = bars.get("tick_volume", 1.0)
    bars["session"] = "unknown"

    # Compute proxy
    proxy_df = timing_proxy(bars)
    proxy_high_first = proxy_df["timing_high_first_proxy"].values

    # Compute true from ticks
    true_df = compute_true_timing(connector.symbol, bars["time"], mt5_module=None)
    if true_df.empty:
        logger.warning("No tick data available for true timing comparison.")
        connector.shutdown()
        return

    # Merge
    merged = bars[["time"]].merge(true_df, left_on="time", right_on="bar_time", how="inner")
    if len(merged) == 0:
        logger.warning("No overlap between bars and tick data.")
        return

    proxy_subset = proxy_high_first[bars["time"].isin(merged["bar_time"])]
    true_vals = merged["true_high_first"].values.astype(float)
    proxy_vals = proxy_subset[:len(true_vals)]

    agreement = float((proxy_vals == true_vals).mean())
    logger.info(f"\n{'='*50}")
    logger.info(f"Timing Proxy vs True Tick Timing Comparison")
    logger.info(f"Bars analyzed: {len(true_vals)}")
    logger.info(f"Agreement rate: {agreement:.1%}")
    if agreement > 0.65:
        logger.info("✓ Proxy has useful signal (>65% agreement).")
    else:
        logger.warning("⚠ Proxy agreement is low (<65%). Timing features may not add value.")
    logger.info(f"{'='*50}")

    # Save results
    Path("reports").mkdir(exist_ok=True)
    with open("reports/timing_proxy_validation.txt", "w") as f:
        f.write(f"Bars analyzed: {len(true_vals)}\n")
        f.write(f"Agreement rate: {agreement:.4f}\n")
    connector.shutdown()


if __name__ == "__main__":
    main()
