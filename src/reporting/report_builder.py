"""
src/reporting/report_builder.py
Generates the final Markdown report and presentation summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, Config
from src.common.logging import setup_logging, get_logger

logger = get_logger("reporting.builder")

LIMITATIONS = """
## Known Limitations

1. **Bar data vs tick data**: Features are computed from 1-minute OHLC. The timing proxy
   (Group E) approximates intra-bar high/low order from bar structure. Agreement rate with
   true tick-level timing is validated in `scripts/compare_timing_true_vs_proxy.py`.

2. **Spread assumptions**: Pre-2015 data uses a synthetic constant spread which may
   underestimate (or overestimate) historical transaction costs. Results before 2015
   should be interpreted as regime-sensitivity analysis only.

3. **Slippage model**: Constant slippage is a simplification. Live fills on gold in
   thin conditions may be significantly higher. Verified in the live-vs-backtest drift panel.

4. **1-minute horizon cost penalty**: The cost-to-signal ratio at 1-minute is ~7.7×
   worse than hourly forecasting (sqrt-time scaling). This is the primary challenge;
   longer holds or highly selective trading are the honest levers for profitability.

5. **Multiple-testing caveat**: The bounded search tried multiple (H, k, m, threshold,
   session) combinations on validation folds. The Deflated Sharpe Ratio (DSR) corrects
   for this. The number of trials is logged in `reports/trials.csv`.

6. **Non-stationarity**: Gold microstructure has changed materially since 2001
   (electronic trading, HFT, ETF growth). Models trained on 2015–2022 may not
   generalize to post-2024 conditions.

7. **Broker differences**: Spread, commission, and execution quality vary by broker.
   All results assume the configured constant spread unless a real spread column exists.
"""


def build_final_report(cfg: Config) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load existing sub-reports
    gate_data = {}
    gate_path = reports_dir / "profit_gate.json"
    if gate_path.exists():
        with open(gate_path) as f:
            gate_data = json.load(f)

    holdout_data = {}
    holdout_path = reports_dir / "holdout_result.json"
    if holdout_path.exists():
        with open(holdout_path) as f:
            holdout_data = json.load(f)

    lines = [
        "# TEGF-XAU Final Report\n",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n",
        f"**Config hash**: `{cfg.config_hash()}`\n\n",
        "---\n\n",
        "## 1. Data Quality\n",
        "See [`reports/data_qc.md`](data_qc.md) for the full QC report.\n\n",
        "## 2. Feature Engineering\n",
        "Features from Groups A–G (returns, bar shape, volume, volatility, "
        "timing proxy, session/time, trend/mean-reversion). "
        "All features are strictly causal — no lookahead. "
        "Scaler fit on training fold only per walk-forward split.\n\n",
        "## 3. Model Comparison\n",
        "See walk-forward fold artifacts in `artifacts/fold_*/`. "
        "Comparison table (directional accuracy, NLL, calibration) generated per fold.\n\n",
        "## 4. Profit Gate\n",
    ]

    if gate_data:
        status = gate_data.get("overall_status", "UNKNOWN")
        lines.append(f"**Status: {status}**\n\n")
        lines.append("| Criterion | Value | Threshold | Pass |\n")
        lines.append("|-----------|-------|-----------|------|\n")
        for c in gate_data.get("criteria", []):
            pass_str = "✅" if c.get("pass") else "❌"
            lines.append(f"| {c['name']} | {c['value']} | {c['threshold']} | {pass_str} |\n")

        if status == "FAILED":
            diag = gate_data.get("diagnosis", {})
            lines.append(f"\n### Diagnosis\n")
            lines.append(f"- {diag.get('cost_to_signal_note', '')}\n")
            lines.append(f"- Achieved win rate: {diag.get('achieved_win_rate', 'N/A')}\n")
            lines.append(f"- Breakeven win rate: {diag.get('breakeven_win_rate', 'N/A')}\n")
            lines.append("\n**Next experiments:**\n")
            for exp in diag.get("next_experiments", []):
                lines.append(f"- {exp}\n")
    else:
        lines.append("_Gate not yet evaluated._\n\n")

    lines += [
        "\n## 5. Holdout Result\n",
        f"```json\n{json.dumps(holdout_data, indent=2)}\n```\n\n" if holdout_data
        else "_Holdout not yet evaluated._\n\n",
        "## 6. Spread Stress Table\n",
        "See `reports/spread_stress.json`.\n\n",
        LIMITATIONS,
        "\n## Honest Conclusion\n",
        "A rigorous null result — where no reliable net-of-cost edge is found "
        "at the 1-minute horizon — is a valid and valuable scientific outcome. "
        "The pipeline correctly identifies when costs exceed the signal, and "
        "provides a clear diagnosis for where edge, if any, may exist.\n",
    ]

    report_path = reports_dir / "final_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info(f"Final report written to {report_path}")

    # 1-page presentation summary
    summary_lines = [
        "# TEGF-XAU — Presentation Summary\n\n",
        "## Project\n",
        "Timing-Enhanced Gated Fusion for 1-Minute XAU/USD Forecasting.\n\n",
        "## Key Results\n",
        f"- **Profit Gate**: {gate_data.get('overall_status', 'NOT EVALUATED')}\n",
        f"- **OOS Trades**: {gate_data.get('n_trades', 'N/A')}\n",
        f"- **Total Net PnL**: {gate_data.get('total_net_pnl', 'N/A')}\n",
        f"- **Profit Factor**: {gate_data.get('profit_factor', 'N/A')}\n",
        f"- **Bootstrap 95% CI**: {gate_data.get('bootstrap_ci', 'N/A')}\n\n",
        "## Honest Caveats\n",
        "- 1-min cost-to-signal ratio is ~7.7× harder than hourly trading\n",
        "- Spread model is synthetic for pre-2015 data\n",
        "- Multiple comparisons corrected via DSR\n",
        "- A null result is reported honestly and completely\n",
        "- Live edge requires continuous monitoring of live-vs-backtest drift\n",
    ]
    summary_path = reports_dir / "presentation_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(summary_lines)
    logger.info(f"Presentation summary written to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    build_final_report(cfg)


if __name__ == "__main__":
    main()
