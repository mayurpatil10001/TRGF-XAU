"""
src/data/qc_report.py
Data quality report: generates reports/data_qc.md and QC plots.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.common.config import Config
from src.common.logging import get_logger

logger = get_logger("data.qc_report")


def _safe_kurtosis(series: pd.Series) -> float:
    try:
        from scipy import stats
        return float(stats.kurtosis(series.dropna(), fisher=True))
    except Exception:
        return float("nan")


def generate_qc_report(df: pd.DataFrame, cfg: Config) -> None:
    """Generate the Markdown QC report and basic plots."""
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Data QC Report — TEGF-XAU\n",
        f"**Generated from**: `{cfg.data.path}`\n",
        f"**Total clean bars**: {len(df):,}\n",
        f"**First bar (UTC)**: {df['time'].iloc[0]}\n",
        f"**Last bar (UTC)**: {df['time'].iloc[-1]}\n",
    ]

    # ── Spread note ──────────────────────────────────────────────────────────
    if df.get("spread_is_synthetic", pd.Series([False])).any():
        lines += [
            "\n> **⚠ SPREAD NOTE**: The raw file has no spread column. "
            "Ask price is modelled as `BID + SPREAD_CONSTANT` where "
            f"`SPREAD_CONSTANT = {cfg.costs.spread_price_units}` price units "
            "(configured in `costs.spread_price_units`). "
            "Spread in 2001–2015 was structurally wider than today; "
            "applying a constant modern spread to old data is conservative but imprecise. "
            "This is flagged in all P/L calculations.\n"
        ]

    # ── Bars per year ────────────────────────────────────────────────────────
    df["year"] = df["time"].dt.year
    bars_per_year = df.groupby("year").size()
    lines += ["\n## Bars per Year\n", "| Year | Bars | Missing-min Rate |\n", "|------|------|------------------|\n"]
    # Total expected 1-min bars in a trading year (approx 252*24*60 but FX runs 5.5 days/week)
    expected_per_year = 252 * 24 * 60  # rough upper bound
    for year, count in bars_per_year.items():
        missing_rate = 1.0 - count / expected_per_year
        missing_rate = max(0.0, missing_rate)
        lines.append(f"| {year} | {count:,} | {missing_rate:.1%} |\n")

    # ── Large gaps ───────────────────────────────────────────────────────────
    large_gaps = df[df["gap_before_minutes"] > cfg.data.max_gap_minutes][
        ["time", "gap_before_minutes"]
    ].sort_values("gap_before_minutes", ascending=False).head(20)
    lines += ["\n## Largest Gaps (top 20, excluding weekends)\n",
              "| Time | Gap (min) |\n", "|------|----------|\n"]
    for _, row in large_gaps.iterrows():
        lines.append(f"| {row['time']} | {row['gap_before_minutes']:.0f} |\n")

    # ── Return distribution and kurtosis ────────────────────────────────────
    df["r_close"] = np.log(df["close"] / df["close"].shift(1))
    lines += ["\n## Return Distribution by Year\n",
              "| Year | Mean(r) | Std(r) | Kurtosis(r) |\n",
              "|------|---------|--------|-------------|\n"]
    for year, grp in df.groupby("year"):
        r = grp["r_close"].dropna()
        lines.append(f"| {year} | {r.mean():.6f} | {r.std():.6f} | {_safe_kurtosis(r):.2f} |\n")

    # ── Top 20 extreme 1-min returns ────────────────────────────────────────
    _df_ext = df.copy()
    _df_ext["_abs_r"] = _df_ext["r_close"].abs()
    extremes = _df_ext.nlargest(20, "_abs_r", keep="all")[
        ["time", "open", "high", "low", "close", "r_close"]
    ]
    lines += ["\n## Top 20 Extreme 1-min Returns\n",
              "| Time | Open | High | Low | Close | r_close |\n",
              "|------|------|------|-----|-------|--------|\n"]
    for _, row in extremes.iterrows():
        lines.append(
            f"| {row['time']} | {row['open']:.2f} | {row['high']:.2f} | "
            f"{row['low']:.2f} | {row['close']:.2f} | {row['r_close']:.6f} |\n"
        )

    # ── Spread distribution (if real) ────────────────────────────────────────
    if "spread" in df.columns and not df.get("spread_is_synthetic", pd.Series([True])).all():
        spread_stats = df.groupby("year")["spread"].describe()
        lines += ["\n## Spread Distribution by Year (price units)\n",
                  spread_stats.to_markdown(), "\n"]
    else:
        lines += [f"\n## Spread\nSynthetic constant spread = {cfg.costs.spread_price_units} units.\n"]

    # ── Plots ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Bars per year
        fig, ax = plt.subplots(figsize=(10, 4))
        bars_per_year.plot(kind="bar", ax=ax)
        ax.set_title("Bars per Year")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(str(plots_dir / "bars_per_year.png"), dpi=100)
        plt.close(fig)

        # Return distribution
        fig, ax = plt.subplots(figsize=(10, 4))
        df["r_close"].dropna().clip(-0.02, 0.02).hist(bins=200, ax=ax, density=True)
        ax.set_title("1-min Log-Return Distribution")
        ax.set_xlabel("log-return")
        fig.tight_layout()
        fig.savefig(str(plots_dir / "return_dist.png"), dpi=100)
        plt.close(fig)

        lines += [
            "\n## Plots\n",
            "![Bars per year](plots/bars_per_year.png)\n",
            "![Return distribution](plots/return_dist.png)\n",
        ]
        logger.info("QC plots saved.")
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")

    # ── Write report ─────────────────────────────────────────────────────────
    report_path = out_dir / "data_qc.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info(f"QC report written to {report_path}")
