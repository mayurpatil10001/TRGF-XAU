"""
src/run_all.py
Master pipeline runner.
Runs: data -> features -> labels -> walk_forward -> backtest -> gate -> report

Usage:
    python -m src.run_all --config config/config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows to avoid cp1252 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config
from src.common.logging import setup_logging, get_logger
from src.common.seed import set_global_seed

logger = get_logger("run_all")


def main() -> None:
    parser = argparse.ArgumentParser(description="TEGF-XAU full pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--holdout", action="store_true", help="Also run holdout evaluation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    set_global_seed(cfg.seed)

    logger.info("=" * 60)
    logger.info("TEGF-XAU FULL PIPELINE")
    logger.info("=" * 60)

    # M1: Data
    logger.info("\n[M1] Data loading and cleaning...")
    from src.data.loader import load_and_clean
    from src.data.qc_report import generate_qc_report
    import pandas as pd
    df = load_and_clean(cfg)
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "bars.parquet", index=False)
    generate_qc_report(df, cfg)
    logger.info(f"[M1] OK - {len(df):,} clean bars saved.")

    # M2: Features
    logger.info("\n[M2] Feature engineering...")
    import json
    from src.features.feature_pipeline import build_features

    df_bars = pd.read_parquet("data/processed/bars.parquet")
    df_bars["time"] = pd.to_datetime(df_bars["time"], utc=True)
    feat_df, feature_names = build_features(df_bars, cfg)

    # Combine into single DataFrame (avoid fragmentation)
    out_feat = pd.concat(
        [df_bars.loc[feat_df.index].reset_index(drop=True),
         feat_df.reset_index(drop=True)],
        axis=1
    )
    # Remove duplicate columns (e.g. if any feature has same name as OHLCV)
    out_feat = out_feat.loc[:, ~out_feat.columns.duplicated()]
    out_feat.to_parquet("data/processed/features.parquet", index=False)

    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"[M2] OK - {len(feature_names)} features built.")

    # M2b: Labels
    logger.info("\n[M2b] Label computation...")
    from src.labels.triple_barrier import compute_labels_fast
    from src.labels.regression_target import compute_regression_target

    df_feat = pd.read_parquet("data/processed/features.parquet")
    df_feat["time"] = pd.to_datetime(df_feat["time"], utc=True)

    dev_end_ts = pd.Timestamp(cfg.data.dev_end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    df_dev = df_feat[df_feat["time"] <= dev_end_ts].copy().reset_index(drop=True)
    df_dev["y_reg"] = compute_regression_target(df_dev)

    primary_H = cfg.labels.horizon_bars[1]   # H=3
    primary_k = cfg.labels.tp_atr_mult_k[1]  # k=0.75
    primary_m = cfg.labels.sl_atr_mult_m[1]  # m=0.75

    lab = compute_labels_fast(df_dev, cfg, primary_H, primary_k, primary_m)
    for col in ["label", "long_R", "short_R", "label_end_index"]:
        if col in lab.columns:
            df_dev[col] = lab[col].values
    df_dev.to_parquet("data/processed/labels.parquet", index=False)
    logger.info(f"[M2b] OK - Labels computed (H={primary_H}, k={primary_k}, m={primary_m}).")

    if args.skip_train:
        logger.info("[M3+] Skipping training (--skip-train).")
    else:
        # M3: Walk-forward training
        logger.info("\n[M3] Walk-forward training...")
        from src.training.walk_forward import run_walk_forward
        oos = run_walk_forward(cfg)
        logger.info(f"[M3] OK - Walk-forward complete. OOS samples: {len(oos.get('times', []))}")

    # M5: Profit gate
    oos_trades_path = Path("data/processed/oos_trades.csv")
    if oos_trades_path.exists():
        logger.info("\n[M5] Profit gate evaluation...")
        from src.evaluation.profit_gate import evaluate_gate, save_gate_result
        trades_df = pd.read_csv(oos_trades_path)
        n_folds = trades_df["fold_id"].nunique() if "fold_id" in trades_df.columns else 1
        result = evaluate_gate(trades_df, n_folds, cfg)
        save_gate_result(result)
        logger.info(f"[M5] Gate result: {result['overall_status']}")
    else:
        logger.info("[M5] No OOS trades file found; skipping gate.")

    if args.holdout:
        logger.info("\n[M5-Holdout] Running holdout evaluation - ONE TIME ONLY...")
        logger.info("[M5-Holdout] Holdout evaluation requires manual execution.")

    # M8: Final report
    logger.info("\n[M8] Generating final report...")
    from src.reporting.report_builder import build_final_report
    build_final_report(cfg)
    logger.info("[M8] Report written to reports/final_report.md")

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
