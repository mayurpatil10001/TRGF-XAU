"""
scripts/cost_breakdown.py  — Step 1: Cost & Trade Breakdown Analyser
=====================================================================
Reads a trade journal CSV (output by backtest engine / live bot) and
reports per-trade and aggregate statistics, broken down by exit reason
and by session.

Usage:
    python scripts/cost_breakdown.py --trades data/processed/oos_trades.csv
    python scripts/cost_breakdown.py --trades data/processed/oos_trades.csv --spread-mult 1.5

Output columns expected in input CSV:
    entry_time, exit_time, side, lots, entry_price, exit_price,
    gross_pnl, cost_usd, net_pnl, R, bars_held, exit_reason,
    p_long, p_short, p_none, fold_id, session, spread_used

The script also writes reports/cost_breakdown_report.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config


def _spread_paid_price_units(row, cfg):
    """Estimate spread paid per trade in price units (1 way)."""
    sp = row.get("spread_used", cfg.costs.spread_price_units)
    return float(sp) if not pd.isna(sp) else cfg.costs.spread_price_units


def _commission_usd(row, cfg):
    """Commission in USD for the round trip."""
    lots = float(row.get("lots", 1.0))
    return cfg.costs.commission_per_lot_round_trip * lots


def _slippage_usd(row, cfg):
    """Estimated slippage in USD (2 legs × slippage_price_units × lots × contract_size)."""
    lots = float(row.get("lots", 1.0))
    return 2 * cfg.costs.slippage_price_units * lots * cfg.symbol.contract_size


def _holding_minutes(row):
    try:
        e = pd.to_datetime(row["entry_time"])
        x = pd.to_datetime(row["exit_time"])
        return (x - e).total_seconds() / 60.0
    except Exception:
        return float("nan")


def analyse(trades_df: pd.DataFrame, cfg, spread_mult: float = 1.0) -> dict:
    df = trades_df.copy()

    if df.empty:
        return {"error": "empty trade journal"}

    # ── Normalise column names (handles older backtest CSV format) ────────────
    rename = {
        "dir": "side", "entry": "entry_price", "exit": "exit_price",
        "pnl": "gross_pnl",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "side" in df.columns:
        df["side"] = df["side"].str.upper().replace({"LONG": "BUY", "SHORT": "SELL"})
    else:
        df["side"] = "BUY"
    if "entry_price" not in df.columns and "entry" in df.columns:
        df["entry_price"] = df["entry"]
    if "exit_price" not in df.columns and "exit" in df.columns:
        df["exit_price"] = df["exit"]

    # ── Per-trade enrichment ──────────────────────────────────────────────────
    df["spread_paid_pu"] = df.apply(lambda r: _spread_paid_price_units(r, cfg) * spread_mult, axis=1)
    df["commission_usd"] = df.apply(lambda r: _commission_usd(r, cfg), axis=1)
    df["slippage_usd"] = df.apply(lambda r: _slippage_usd(r, cfg), axis=1)
    df["total_cost_usd"] = df["commission_usd"] + df["slippage_usd"] + (
        df["spread_paid_pu"] * df.get("lots", 1.0) * cfg.symbol.contract_size
    )
    df["holding_min"] = df.apply(_holding_minutes, axis=1)

    # Gross PnL from entry/exit if not present
    if "gross_pnl" not in df.columns:
        dir_ = df["side"].map({"BUY": 1, "SELL": -1}).fillna(1)
        df["gross_pnl"] = dir_ * (df["exit_price"] - df["entry_price"]) * df["lots"] * cfg.symbol.contract_size

    if "net_pnl" not in df.columns:
        df["net_pnl"] = df["gross_pnl"] - df["total_cost_usd"]

    pnl = df["net_pnl"].values
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    n = len(pnl)
    win_rate = float((pnl > 0).mean()) if n > 0 else 0.0
    rr = abs(wins.mean() / losses.mean()) if len(wins) > 0 and len(losses) > 0 else 0.0
    breakeven_wr = 1.0 / (1.0 + rr) if rr > 0 else 0.5
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and abs(losses.sum()) > 0 else float("inf")
    exp = float(pnl.mean()) if n > 0 else 0.0

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "n_trades": n,
        "total_net_pnl_usd": round(float(pnl.sum()), 2),
        "expectancy_per_trade_usd": round(exp, 4),
        "profit_factor": round(pf, 4),
        "win_rate": round(win_rate, 4),
        "breakeven_win_rate": round(breakeven_wr, 4),
        "avg_win_usd": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss_usd": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "avg_gross_pnl_usd": round(float(df["gross_pnl"].mean()), 4),
        "avg_commission_usd": round(float(df["commission_usd"].mean()), 4),
        "avg_slippage_usd": round(float(df["slippage_usd"].mean()), 4),
        "avg_spread_paid_pu": round(float(df["spread_paid_pu"].mean()), 4),
        "avg_total_cost_usd": round(float(df["total_cost_usd"].mean()), 4),
        "avg_holding_min": round(float(df["holding_min"].mean()), 2),
        "spread_mult_applied": spread_mult,
    }

    # ── By exit reason ────────────────────────────────────────────────────────
    by_exit = {}
    if "exit_reason" in df.columns:
        for reason, grp in df.groupby("exit_reason"):
            g_pnl = grp["net_pnl"].values
            by_exit[reason] = {
                "n": len(g_pnl),
                "pct": round(len(g_pnl) / n * 100, 1),
                "total_pnl": round(float(g_pnl.sum()), 2),
                "avg_pnl": round(float(g_pnl.mean()), 4),
                "win_rate": round(float((g_pnl > 0).mean()), 4),
                "avg_hold_min": round(float(grp["holding_min"].mean()), 2),
            }

    # ── By session ────────────────────────────────────────────────────────────
    by_session = {}
    if "session" in df.columns:
        for sess, grp in df.groupby("session"):
            g_pnl = grp["net_pnl"].values
            by_session[sess] = {
                "n": len(g_pnl),
                "total_pnl": round(float(g_pnl.sum()), 2),
                "avg_pnl": round(float(g_pnl.mean()), 4),
                "win_rate": round(float((g_pnl > 0).mean()), 4),
            }

    # ── By fold ───────────────────────────────────────────────────────────────
    by_fold = {}
    if "fold_id" in df.columns:
        for fid, grp in df.groupby("fold_id"):
            g_pnl = grp["net_pnl"].values
            g_wins = g_pnl[g_pnl > 0]
            g_loss = g_pnl[g_pnl < 0]
            g_pf = abs(g_wins.sum() / g_loss.sum()) if len(g_loss) > 0 and abs(g_loss.sum()) > 0 else float("inf")
            by_fold[int(fid)] = {
                "n": len(g_pnl),
                "total_pnl": round(float(g_pnl.sum()), 2),
                "profit_factor": round(g_pf, 4),
                "win_rate": round(float((g_pnl > 0).mean()), 4),
            }

    # ── Bootstrap CI on expectancy ────────────────────────────────────────────
    if n >= 30:
        rng = np.random.default_rng(42)
        boot_means = np.array([rng.choice(pnl, size=n, replace=True).mean() for _ in range(5000)])
        ci_lower, ci_upper = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    else:
        ci_lower, ci_upper = float("-inf"), float("inf")

    summary["bootstrap_95ci_lower"] = round(ci_lower, 4)
    summary["bootstrap_95ci_upper"] = round(ci_upper, 4)

    return {
        "summary": summary,
        "by_exit_reason": by_exit,
        "by_session": by_session,
        "by_fold": by_fold,
        "enriched_df": df,
    }


def build_markdown(result: dict) -> str:
    s = result["summary"]
    lines = [
        "# Cost Breakdown Report — TEGF-XAU",
        "",
        "## Aggregate Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Trades | {s['n_trades']} |",
        f"| Total net PnL | ${s['total_net_pnl_usd']:,.2f} |",
        f"| Expectancy / trade | ${s['expectancy_per_trade_usd']:.4f} |",
        f"| Profit Factor | {s['profit_factor']:.4f} |",
        f"| Win Rate | {s['win_rate']:.1%} |",
        f"| Breakeven Win Rate | {s['breakeven_win_rate']:.1%} |",
        f"| Avg Win | ${s['avg_win_usd']:.2f} |",
        f"| Avg Loss | ${s['avg_loss_usd']:.2f} |",
        f"| Avg Gross PnL | ${s['avg_gross_pnl_usd']:.4f} |",
        f"| Avg Commission | ${s['avg_commission_usd']:.4f} |",
        f"| Avg Slippage | ${s['avg_slippage_usd']:.4f} |",
        f"| Avg Spread Cost (price units) | {s['avg_spread_paid_pu']:.4f} |",
        f"| Avg Total Cost/trade | ${s['avg_total_cost_usd']:.4f} |",
        f"| Avg Holding (min) | {s['avg_holding_min']:.2f} |",
        f"| Bootstrap 95% CI | [{s['bootstrap_95ci_lower']:.4f}, {s['bootstrap_95ci_upper']:.4f}] |",
        f"| Spread multiplier | {s['spread_mult_applied']:.1f}x |",
        "",
        "## By Exit Reason",
        "",
        "| Reason | N | % | Total PnL | Avg PnL | Win Rate | Avg Hold (min) |",
        "|--------|---|---|-----------|---------|----------|----------------|",
    ]
    for reason, v in result.get("by_exit_reason", {}).items():
        lines.append(
            f"| {reason} | {v['n']} | {v['pct']:.1f}% | ${v['total_pnl']:,.2f} "
            f"| ${v['avg_pnl']:.4f} | {v['win_rate']:.1%} | {v['avg_hold_min']:.1f} |"
        )

    lines += [
        "",
        "## By Session",
        "",
        "| Session | N | Total PnL | Avg PnL | Win Rate |",
        "|---------|---|-----------|---------|----------|",
    ]
    for sess, v in result.get("by_session", {}).items():
        lines.append(
            f"| {sess} | {v['n']} | ${v['total_pnl']:,.2f} "
            f"| ${v['avg_pnl']:.4f} | {v['win_rate']:.1%} |"
        )

    lines += [
        "",
        "## By Fold",
        "",
        "| Fold | N | Total PnL | Profit Factor | Win Rate |",
        "|------|---|-----------|---------------|----------|",
    ]
    for fid, v in result.get("by_fold", {}).items():
        pf_str = f"{v['profit_factor']:.4f}" if v['profit_factor'] < 1e6 else "∞"
        lines.append(
            f"| {fid} | {v['n']} | ${v['total_pnl']:,.2f} | {pf_str} | {v['win_rate']:.1%} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Cost breakdown analyser")
    parser.add_argument("--trades", default="data/processed/oos_trades.csv")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--spread-mult", type=float, default=1.0)
    parser.add_argument("--out", default="reports/cost_breakdown_report.md")
    args = parser.parse_args()

    cfg = load_config(args.config)
    trades_path = Path(args.trades)
    if not trades_path.exists():
        print(f"ERROR: Trades file not found: {trades_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(trades_path)
    print(f"Loaded {len(df)} trades from {trades_path}")

    result = analyse(df, cfg, spread_mult=args.spread_mult)
    md = build_markdown(result)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"\n{md}")
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
