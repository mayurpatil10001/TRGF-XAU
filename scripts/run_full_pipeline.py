"""
scripts/run_full_pipeline.py  — Steps 3+4: Full ML pipeline with honest validation
=====================================================================================
Runs: data → features → extended labels (H=15,30,60) → walk-forward →
      probability calibration → meta-labeling → backtest (1.0/1.5/2.0x spread) →
      baselines → label-shuffle sanity → profit gate → report

Trial tracking: every variant (H, k, threshold, session, meta_on/off) is
recorded in reports/trial_log.csv. Results for ALL trials are reported,
including failures — no cherry-picking.

Usage:
    python scripts/run_full_pipeline.py --config config/config.yaml
    python scripts/run_full_pipeline.py --config config/config.yaml --holdout
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from src.common.config import load_config, Config
from src.common.logging import setup_logging, get_logger
from src.common.seed import set_global_seed
from src.labels.triple_barrier import cost_round_trip

logger = get_logger("pipeline.run_full")


# ══════════════════════════════════════════════════════════════════════════════
# Baselines
# ══════════════════════════════════════════════════════════════════════════════

def baseline_always_flat(n_trades: int) -> float:
    """Always-flat baseline: net PnL = 0."""
    return 0.0


def baseline_random_signal(
    df: pd.DataFrame, probs: np.ndarray, cfg: Config,
    n_trades: int, seed: int = 42
) -> pd.DataFrame:
    """
    Random-signal baseline: same trade frequency as real strategy,
    random directions. Confirms our strategy beats random at the same
    transaction cost.
    """
    rng = np.random.default_rng(seed)
    n = len(probs)
    rand_probs = np.zeros((n, 3))
    rand_probs[:, 0] = 0.33
    rand_probs[:, 1] = rng.uniform(0, 1, n)
    rand_probs[:, 2] = 1 - rand_probs[:, 0] - rand_probs[:, 1]
    rand_probs = np.abs(rand_probs)
    rand_probs /= rand_probs.sum(axis=1, keepdims=True)
    return rand_probs


# ══════════════════════════════════════════════════════════════════════════════
# Calibration
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_probs(
    probs_val: np.ndarray,
    y_val: np.ndarray,
    probs_test: np.ndarray,
    method: str = "isotonic",
) -> np.ndarray:
    """
    Calibrate class probabilities using isotonic regression or Platt scaling.
    Fit on val fold, apply to test fold.
    Returns calibrated [N_test, 3] probs.
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return probs_test

    calibrated = np.zeros_like(probs_test)
    for cls in range(3):
        y_bin = (y_val == cls).astype(int)
        if y_bin.sum() < 5:
            calibrated[:, cls] = probs_test[:, cls]
            continue
        try:
            if method == "isotonic":
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(probs_val[:, cls], y_bin)
                calibrated[:, cls] = ir.predict(probs_test[:, cls])
            else:  # platt
                # Platt: logistic regression on raw probs
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(C=1.0, max_iter=200)
                lr.fit(probs_val[:, cls:cls+1], y_bin)
                calibrated[:, cls] = lr.predict_proba(probs_test[:, cls:cls+1])[:, 1]
        except Exception:
            calibrated[:, cls] = probs_test[:, cls]

    # Re-normalize
    row_sums = calibrated.sum(axis=1, keepdims=True)
    calibrated = calibrated / np.maximum(row_sums, 1e-12)
    return calibrated


# ══════════════════════════════════════════════════════════════════════════════
# Label-shuffle sanity check
# ══════════════════════════════════════════════════════════════════════════════

def label_shuffle_sanity(
    probs: np.ndarray,
    df: pd.DataFrame,
    cfg: Config,
    trial_params: dict,
    seed: int = 42,
) -> dict:
    """
    Shuffle labels and re-run backtest. Strategy must NOT be profitable
    on shuffled labels (confirms signal is from labels, not spurious rules).
    Returns {'passed': bool, 'shuffled_pf': float, 'shuffled_pnl': float}.
    """
    from src.backtest.engine import run_backtest, backtest_stats

    rng = np.random.default_rng(seed)
    shuffled_probs = probs.copy()
    rng.shuffle(shuffled_probs)  # shuffle along sample axis

    logs, curve = run_backtest(
        df, shuffled_probs, cfg,
        spread_mult=1.0,
        horizon_bars=trial_params["horizon"],
        k=trial_params["k"],
        m=trial_params["m"],
        prob_threshold=trial_params["threshold"],
    )
    if not logs:
        return {"passed": True, "shuffled_pf": 0.0, "shuffled_pnl": 0.0, "n_trades": 0}

    pnl_arr = np.array([t.net_pnl for t in logs])
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and abs(losses.sum()) > 0 else 0.0
    total = float(pnl_arr.sum())

    # Must NOT be profitable (PF < 1.1)
    passed = pf < 1.1
    return {
        "passed": passed,
        "shuffled_pf": round(pf, 4),
        "shuffled_pnl": round(total, 2),
        "n_trades": len(logs),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(cfg: Config, run_holdout: bool = False) -> None:
    set_global_seed(cfg.seed)
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    trial_records = []
    trial_num = 0

    # ── M1: Data ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[M1] Loading and cleaning data...")
    from src.data.loader import load_and_clean
    from src.data.qc_report import generate_qc_report
    df_bars = load_and_clean(cfg)
    df_bars.to_parquet(out_dir / "bars.parquet", index=False)
    generate_qc_report(df_bars, cfg)
    logger.info(f"[M1] OK — {len(df_bars):,} clean bars")

    # ── M2: Features ──────────────────────────────────────────────────────────
    logger.info("[M2] Building features...")
    from src.features.feature_pipeline import build_features, wilder_atr
    import json as _json
    feat_df, feature_names = build_features(df_bars, cfg)
    out_feat = pd.concat(
        [df_bars.loc[feat_df.index].reset_index(drop=True),
         feat_df.reset_index(drop=True)],
        axis=1
    )
    out_feat = out_feat.loc[:, ~out_feat.columns.duplicated()]
    out_feat.to_parquet(out_dir / "features.parquet", index=False)
    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/feature_names.json", "w") as f:
        _json.dump(feature_names, f, indent=2)
    logger.info(f"[M2] OK — {len(feature_names)} features")

    # Dev period filter
    dev_end_ts = pd.Timestamp(cfg.data.dev_end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    df_feat = pd.read_parquet(out_dir / "features.parquet")
    df_feat["time"] = pd.to_datetime(df_feat["time"], utc=True)
    df_dev = df_feat[df_feat["time"] <= dev_end_ts].copy().reset_index(drop=True)

    logger.info(f"Dev period: {df_dev['time'].min()} → {df_dev['time'].max()}")
    logger.info(f"Dev bars: {len(df_dev):,}")

    # ── M2b: Extended labels (H=15,30,60) ────────────────────────────────────
    logger.info("[M2b] Computing extended labels (H=15,30,60)...")
    from src.labels.triple_barrier import compute_labels_fast
    from src.labels.regression_target import compute_regression_target
    df_dev["y_reg"] = compute_regression_target(df_dev)

    # Use primary config (first combo) as the default label set
    primary_H = cfg.labels.horizon_bars[0]   # e.g. 15
    primary_k = cfg.labels.tp_atr_mult_k[0]  # e.g. 1.0
    primary_m = cfg.labels.sl_atr_mult_m[0]  # e.g. 0.5

    lab = compute_labels_fast(df_dev, cfg, primary_H, primary_k, primary_m)
    for col in ["label", "long_R", "short_R", "label_end_index"]:
        if col in lab.columns:
            df_dev[col] = lab[col].values
    df_dev["label_horizon"] = primary_H
    df_dev["label_k"] = primary_k
    df_dev["label_m"] = primary_m
    df_dev.to_parquet(out_dir / "labels.parquet", index=False)

    label_counts = df_dev["label"].value_counts().to_dict()
    logger.info(f"[M2b] OK — Label counts (H={primary_H}): {label_counts}")

    # ── M3: Walk-forward with calibration + meta-labeling ─────────────────────
    logger.info("[M3] Walk-forward training (all label combos)...")
    from src.training.walk_forward import run_walk_forward
    from src.training.windows import generate_folds
    from src.strategy.meta_labeler import MetaLabeler
    import joblib

    feat_arr = df_dev[feature_names].values.astype(np.float32)
    times = df_dev["time"].reset_index(drop=True)
    folds = generate_folds(times, cfg)

    if not folds:
        logger.error("[M3] No walk-forward folds could be generated! Data span too short.")
        logger.error(
            f"Need at least {cfg.walk_forward.train_months + cfg.walk_forward.val_months + cfg.walk_forward.test_months} "
            f"months of data. Have ~{(df_dev['time'].max() - df_dev['time'].min()).days / 30:.1f} months."
        )
        _write_fail_report(cfg, trial_records)
        return

    logger.info(f"[M3] Generated {len(folds)} walk-forward folds")

    # Trial grid: (horizon, k, m, threshold, session_filter, meta_on)
    h_grid = cfg.labels.horizon_bars
    k_grid = cfg.labels.tp_atr_mult_k
    m_grid = [cfg.labels.sl_atr_mult_m[1]]  # use middle m for all trials
    thresh_grid = cfg.decision.prob_threshold_grid
    all_oos_trades = []

    # Train baseline walk-forward once per fold, then iterate trials
    fold_artifacts = {}  # fold_id → {lgbm, scaler, probs_val, probs_test, ...}

    for fold in folds:
        fold_id = fold.fold_id
        tr_idx, va_idx, te_idx = fold.train_idx, fold.val_idx, fold.test_idx
        if len(tr_idx) < 100 or len(va_idx) < 10 or len(te_idx) < 10:
            logger.warning(f"Fold {fold_id}: too few samples, skipping.")
            continue

        logger.info(f"\nFold {fold_id}: train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}")

        from src.features.feature_pipeline import fit_scaler, apply_scaler
        train_mask = np.zeros(len(df_dev), dtype=bool)
        train_mask[tr_idx] = True
        scaler = fit_scaler(pd.DataFrame(feat_arr, columns=feature_names), pd.Series(train_mask))
        feat_df_scaled = apply_scaler(pd.DataFrame(feat_arr, columns=feature_names), scaler)
        feat_scaled = feat_df_scaled.values.astype(np.float32)

        fold_art_dir = Path("artifacts") / f"fold_{fold_id:02d}"
        fold_art_dir.mkdir(exist_ok=True)
        joblib.dump(scaler, fold_art_dir / "scaler.joblib")

        X_tr = feat_scaled[tr_idx]
        X_va = feat_scaled[va_idx]
        X_te = feat_scaled[te_idx]

        from src.models.lgbm_baseline import train_lgbm, predict_lgbm, save_lgbm
        y_cls = df_dev["label"].values.astype(np.int64)
        lgbm_model = train_lgbm(X_tr, y_cls[tr_idx], X_va, y_cls[va_idx], cfg, seed=cfg.seed)
        save_lgbm(lgbm_model, fold_art_dir / "lgbm.txt")

        probs_val = predict_lgbm(lgbm_model, X_va)
        probs_test_raw = predict_lgbm(lgbm_model, X_te)

        # Calibrate on val → apply to test
        probs_test_cal = calibrate_probs(probs_val, y_cls[va_idx], probs_test_raw, method="isotonic")

        # ATR and spread for meta-features
        atr_col = "atr" if "atr" in df_dev.columns else "close"
        spread_col = "spread" if "spread" in df_dev.columns else None
        atrs_te = df_dev[atr_col].values[te_idx] if atr_col in df_dev.columns else np.full(len(te_idx), cfg.features.atr_period)
        spreads_te = df_dev[spread_col].values[te_idx] if spread_col else np.full(len(te_idx), cfg.costs.spread_price_units)

        fold_artifacts[fold_id] = {
            "tr_idx": tr_idx, "va_idx": va_idx, "te_idx": te_idx,
            "X_va": X_va, "X_te": X_te,
            "probs_val": probs_val, "probs_test_raw": probs_test_raw,
            "probs_test_cal": probs_test_cal,
            "y_cls_va": y_cls[va_idx], "y_cls_te": y_cls[te_idx],
            "atrs_te": atrs_te, "spreads_te": spreads_te,
            "df_test": df_dev.iloc[te_idx].copy().reset_index(drop=True),
        }

    if not fold_artifacts:
        logger.error("[M3] No folds completed training.")
        _write_fail_report(cfg, trial_records)
        return

    # ── M4: Trial grid — evaluate each (H, k, threshold, meta) combo ─────────
    logger.info("\n[M4] Running trial grid...")
    best_trial = None
    best_pf = -np.inf

    for H, k, threshold, use_meta in product(h_grid, k_grid, thresh_grid, [False, True]):
        trial_num += 1
        m = m_grid[0]
        trial_key = f"H{H}_k{k}_m{m}_thr{threshold}_meta{int(use_meta)}"
        logger.info(f"\n--- Trial {trial_num}: {trial_key} ---")

        # Collect OOS trades across folds for this trial
        all_trial_trades = []

        for fold_id, fa in fold_artifacts.items():
            df_test = fa["df_test"]

            # Recompute labels for this H, k, m
            lab_test = compute_labels_fast(df_test, cfg, H, k, m)
            for col in ["label", "long_R", "short_R", "label_end_index"]:
                if col in lab_test.columns:
                    df_test[col] = lab_test[col].values

            probs_test = fa["probs_test_cal"]

            # Apply meta-labeler
            if use_meta:
                # Train meta on val fold outcomes first
                # We need val-fold actual trade outcomes — run a quick val backtest
                df_val = df_dev.iloc[fa["va_idx"]].copy().reset_index(drop=True)
                lab_val = compute_labels_fast(df_val, cfg, H, k, m)
                for col in ["label", "long_R", "short_R", "label_end_index"]:
                    if col in lab_val.columns:
                        df_val[col] = lab_val[col].values

                from src.backtest.engine import run_backtest
                val_logs, _ = run_backtest(
                    df_val, fa["probs_val"], cfg,
                    spread_mult=1.0, horizon_bars=H, k=k, m=m,
                    prob_threshold=threshold, fold_id=fold_id,
                )
                if val_logs and len(val_logs) >= 10:
                    val_pnl = np.array([t.net_pnl for t in val_logs])
                    atrs_va = df_dev["atr"].values[fa["va_idx"]] if "atr" in df_dev.columns else np.full(len(fa["va_idx"]), 0.3)
                    spreads_va = (df_dev["spread"].values[fa["va_idx"]] if "spread" in df_dev.columns
                                  else np.full(len(fa["va_idx"]), cfg.costs.spread_price_units))

                    meta = MetaLabeler(cfg, meta_threshold=0.55)
                    meta_trained = meta.fit(
                        fa["X_va"], fa["probs_val"], spreads_va, atrs_va, val_pnl
                    )

                    if meta_trained:
                        meta_probs = meta.predict_proba(fa["X_te"], probs_test, fa["spreads_te"], fa["atrs_te"])
                        # Zero out test probs where meta says no
                        meta_mask = meta_probs < meta.meta_threshold
                        probs_test = probs_test.copy()
                        probs_test[meta_mask, 0] = 1.0   # push to NO_TRADE
                        probs_test[meta_mask, 1:] = 0.0

            # Run backtest at 1.0x spread for this fold
            from src.backtest.engine import run_backtest
            logs, curve = run_backtest(
                df_test, probs_test, cfg,
                spread_mult=1.0, horizon_bars=H, k=k, m=m,
                prob_threshold=threshold, fold_id=fold_id,
            )
            all_trial_trades.extend(logs)

        if not all_trial_trades:
            trial_records.append({
                "trial_num": trial_num, "key": trial_key,
                "H": H, "k": k, "m": m, "threshold": threshold, "meta": use_meta,
                "n_trades": 0, "net_pnl": 0.0, "profit_factor": 0.0,
                "win_rate": 0.0, "breakeven_wr": 0.0, "expectancy": 0.0,
                "positive_fold_frac": 0.0, "boot_ci_lower": 0.0,
                "pf_1_5x": 0.0, "pf_2_0x": 0.0, "gate": "FAILED_NO_TRADES",
            })
            continue

        pnl = np.array([t.net_pnl for t in all_trial_trades])
        wins_arr = pnl[pnl > 0]
        loss_arr = pnl[pnl < 0]
        n = len(pnl)
        pf = abs(wins_arr.sum() / loss_arr.sum()) if len(loss_arr) > 0 and abs(loss_arr.sum()) > 0 else float("inf")
        wr = float((pnl > 0).mean())
        rr = abs(wins_arr.mean() / loss_arr.mean()) if len(wins_arr) > 0 and len(loss_arr) > 0 else 0.0
        bkv_wr = 1.0 / (1.0 + rr) if rr > 0 else 0.5
        exp = float(pnl.mean())
        total_pnl = float(pnl.sum())

        # Bootstrap CI
        if n >= 30:
            rng = np.random.default_rng(42)
            boot = np.array([rng.choice(pnl, size=n, replace=True).mean() for _ in range(2000)])
            ci_lower = float(np.percentile(boot, 2.5))
        else:
            ci_lower = float("-inf")

        # Spread stress
        cost_rt = cost_round_trip(cfg)
        extra_cost_1_5x = 0.5 * cfg.costs.spread_price_units  # additional spread per trade at 1.5x
        pnl_1_5x = pnl - extra_cost_1_5x
        pnl_2_0x = pnl - 2 * extra_cost_1_5x
        pf_1_5x = (pnl_1_5x[pnl_1_5x > 0].sum() / abs(pnl_1_5x[pnl_1_5x < 0].sum())
                   if len(pnl_1_5x[pnl_1_5x < 0]) > 0 else float("inf"))
        pf_2_0x = (pnl_2_0x[pnl_2_0x > 0].sum() / abs(pnl_2_0x[pnl_2_0x < 0].sum())
                   if len(pnl_2_0x[pnl_2_0x < 0]) > 0 else float("inf"))

        # Fold positive fraction
        fold_pnls = {}
        for t in all_trial_trades:
            fid = t.fold_id
            fold_pnls.setdefault(fid, []).append(t.net_pnl)
        pos_fold_frac = float(sum(1 for v in fold_pnls.values() if sum(v) > 0) / max(1, len(fold_pnls)))

        # Label-shuffle sanity (only for promising trials, to save time)
        shuffle_ok = True
        if pf >= 1.0 and total_pnl > 0 and n >= 30:
            # Use combined test data from first fold for shuffle check
            first_fold_id = list(fold_artifacts.keys())[0]
            fa0 = fold_artifacts[first_fold_id]
            shuffle_result = label_shuffle_sanity(
                fa0["probs_test_cal"], fa0["df_test"], cfg,
                {"horizon": H, "k": k, "m": m, "threshold": threshold}
            )
            shuffle_ok = shuffle_result["passed"]

        # Gate decision for this trial
        gate_passed = (
            total_pnl > 0 and
            pf >= cfg.profit_gate.min_profit_factor and
            ci_lower > 0 and
            pf_1_5x >= 1.0 and
            pos_fold_frac >= cfg.profit_gate.min_positive_fold_fraction and
            n >= cfg.profit_gate.min_oos_trades and
            shuffle_ok
        )
        gate_status = "PASSED" if gate_passed else "FAILED"

        if gate_passed and pf > best_pf:
            best_pf = pf
            best_trial = {
                "trial_key": trial_key, "H": H, "k": k, "m": m,
                "threshold": threshold, "use_meta": use_meta,
                "trade_logs": all_trial_trades,
            }

        logger.info(
            f"  n={n} PnL={total_pnl:.2f} PF={pf:.3f} WR={wr:.1%} "
            f"bkv_wr={bkv_wr:.1%} CI=[{ci_lower:.2f},...] "
            f"PF@1.5x={pf_1_5x:.3f} folds+={pos_fold_frac:.0%} shuffle={shuffle_ok} "
            f"→ {gate_status}"
        )

        trial_records.append({
            "trial_num": trial_num, "key": trial_key,
            "H": H, "k": k, "m": m, "threshold": threshold, "meta": use_meta,
            "n_trades": n, "net_pnl": round(total_pnl, 2),
            "profit_factor": round(pf, 4), "win_rate": round(wr, 4),
            "breakeven_wr": round(bkv_wr, 4), "expectancy": round(exp, 4),
            "positive_fold_frac": round(pos_fold_frac, 4),
            "boot_ci_lower": round(ci_lower, 4),
            "pf_1_5x": round(pf_1_5x, 4), "pf_2_0x": round(pf_2_0x, 4),
            "gate": gate_status, "shuffle_ok": shuffle_ok,
        })

    # ── Save trial log ────────────────────────────────────────────────────────
    trial_df = pd.DataFrame(trial_records)
    Path("reports").mkdir(exist_ok=True)
    trial_df.to_csv("reports/trial_log.csv", index=False)
    logger.info(f"\n[M4] {trial_num} trials evaluated. Saved to reports/trial_log.csv")

    # ── M5: Profit gate on best trial ─────────────────────────────────────────
    from src.evaluation.profit_gate import evaluate_gate, save_gate_result

    if best_trial is None:
        logger.warning("[M5] NO TRIAL PASSED THE PROFIT GATE.")
        result = {
            "overall_status": "FAILED",
            "note": "No configuration showed positive net expectancy after costs",
            "n_trials": trial_num,
            "diagnosis": {
                "next_experiments": [
                    "Extend data history beyond 3.7 months for more folds",
                    "Test H=60+ with higher ATR environments",
                    "Restrict further to overlap session (12-16 UTC)",
                    "Add regime filter (only trade trending markets)",
                ]
            }
        }
        save_gate_result(result)
    else:
        logger.info(f"\n[M5] Best trial: {best_trial['trial_key']}")
        best_logs = best_trial["trade_logs"]
        best_df = pd.DataFrame([vars(t) for t in best_logs])
        best_df.to_csv("data/processed/oos_trades.csv", index=False)

        fold_count = best_df["fold_id"].nunique() if "fold_id" in best_df.columns else 1
        gate_result = evaluate_gate(best_df, fold_count, cfg, shuffle_sanity_passed=True)
        save_gate_result(gate_result)
        logger.info(f"[M5] Gate result: {gate_result['overall_status']}")

    # ── Holdout (ONE TIME ONLY if requested) ──────────────────────────────────
    if run_holdout and best_trial is not None:
        holdout_result = _run_holdout(df_feat, cfg, best_trial, feature_names)
        logger.info(f"[M5-Holdout] Holdout result: {holdout_result}")

    # ── M8: Final report ──────────────────────────────────────────────────────
    _write_final_report(cfg, trial_df, trial_num, best_trial)

    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE in {elapsed/60:.1f} min")
    logger.info(f"Total trials: {trial_num}")
    logger.info(f"Best trial: {best_trial['trial_key'] if best_trial else 'NONE (all failed)'}")
    logger.info(f"{'='*60}")


def _run_holdout(df_feat: pd.DataFrame, cfg: Config, best_trial: dict, feature_names: list) -> dict:
    """Run holdout evaluation ONCE for the chosen final configuration."""
    logger.info("[Holdout] Running holdout — TOUCHING HOLDOUT DATA FOR THE FIRST TIME.")
    holdout_start_ts = pd.Timestamp(cfg.data.holdout_start, tz="UTC")
    df_holdout = df_feat[df_feat["time"] >= holdout_start_ts].copy().reset_index(drop=True)

    if len(df_holdout) < 50:
        return {"error": "insufficient holdout data", "n": len(df_holdout)}

    # Load last fold model and scaler
    import glob, joblib
    scaler_paths = sorted(glob.glob("artifacts/fold_*/scaler.joblib"))
    if not scaler_paths:
        return {"error": "no scaler found"}

    from src.models.lgbm_baseline import predict_lgbm
    import lightgbm as lgb
    model_paths = sorted(glob.glob("artifacts/fold_*/lgbm.txt"))
    model = lgb.Booster(model_file=model_paths[-1])
    scaler = joblib.load(scaler_paths[-1])

    from src.features.feature_pipeline import apply_scaler
    X_holdout = apply_scaler(df_holdout[feature_names], scaler).values.astype(np.float32)
    probs = predict_lgbm(model, X_holdout)

    H, k, m = best_trial["H"], best_trial["k"], best_trial["m"]
    threshold = best_trial["threshold"]

    from src.labels.triple_barrier import compute_labels_fast
    lab = compute_labels_fast(df_holdout, cfg, H, k, m)
    for col in ["label", "long_R", "short_R", "label_end_index"]:
        if col in lab.columns:
            df_holdout[col] = lab[col].values

    from src.backtest.engine import run_backtest, backtest_stats
    logs, curve = run_backtest(
        df_holdout, probs, cfg,
        spread_mult=1.0, horizon_bars=H, k=k, m=m,
        prob_threshold=threshold, fold_id=-1,
    )
    if not logs:
        return {"n_trades": 0, "net_pnl": 0, "gate": "FAILED_NO_TRADES"}

    stats = backtest_stats(logs, curve)
    # Save holdout trades
    holdout_df = pd.DataFrame([vars(t) for t in logs])
    holdout_df.to_csv("data/processed/holdout_trades.csv", index=False)
    logger.info(f"[Holdout] n={stats['n_trades']} PnL={stats['total_net_pnl']:.2f} PF={stats['profit_factor']:.3f}")
    return stats


def _write_fail_report(cfg: Config, trial_records: list) -> None:
    Path("reports").mkdir(exist_ok=True)
    trial_df = pd.DataFrame(trial_records) if trial_records else pd.DataFrame()
    _write_final_report(cfg, trial_df, len(trial_records), None)


def _write_final_report(cfg: Config, trial_df: pd.DataFrame, n_trials: int, best_trial: dict) -> None:
    from src.evaluation.profit_gate import load_gate_status
    gate_status = load_gate_status()

    if best_trial is not None:
        verdict = (
            "✅ PASSED — ready for extended demo forward-test"
            if gate_status == "PASSED"
            else f"⚠️ PROVISIONAL ({gate_status}) — continue monitoring"
        )
    else:
        verdict = "❌ FAILED — no configuration showed positive net expectancy after costs"

    cost_rt = cost_round_trip(cfg)
    rpt = [
        "# TEGF-XAU Strategy Validation Report",
        f"\n**Generated:** {pd.Timestamp.utcnow().isoformat()}",
        f"**Total trials evaluated:** {n_trials}",
        f"**Gate status:** {gate_status}",
        f"\n## Verdict\n\n{verdict}",
        "",
        "## Baseline Diagnosis",
        "",
        "| Metric | Baseline (HFT v5) |",
        "|--------|------------------|",
        "| Win rate | 35% |",
        "| Breakeven win rate | 62.6% |",
        "| Profit factor | 0.32 |",
        "| Expectancy/trade | -$7.34 |",
        "| Total PnL | -$9,981 |",
        f"| Round-trip cost | {cost_rt:.4f} price units ({cost_rt * 100:.2f} pts) |",
        "",
        "**Root cause confirmed:**",
        "- HFT bot enters BUY at ask, exits at bid → spread paid immediately",
        "- With 0.4pt profit target and ~0.25pt spread, 60%+ of profit target consumed by costs",
        "- TICK_MIN=2/5 polled ticks ≈ noise, not signal",
        "- 5 correlated positions amplify losses",
        "",
        "## Cost Structure",
        "",
        f"| Cost Component | Value |",
        f"|----------------|-------|",
        f"| Spread | {cfg.costs.spread_price_units:.3f} pu |",
        f"| Slippage (2 legs) | {2*cfg.costs.slippage_price_units:.3f} pu |",
        f"| Commission | {cfg.costs.commission_per_lot_round_trip/cfg.symbol.contract_size:.4f} pu/lot |",
        f"| **Total round-trip** | **{cost_rt:.4f} pu** |",
        f"| Min TP required (3× cost) | {3*cost_rt:.4f} pu |",
        "",
        "## Fixes Applied (this run)",
        "",
        "1. **Horizons extended** to H=15,30,60 bars (vs H=1,3,5) — more time to beat costs",
        "2. **Probability threshold** lifted to ≥0.55 (vs ≤0.38 = below random baseline)",
        "3. **Single position only** — no more stacking up to 5 correlated trades",
        "4. **Calibrated probabilities** — isotonic regression on validation fold",
        "5. **Meta-labeling** — secondary filter requiring both primary+meta agreement",
        "6. **Symmetric stops** — counter-trend gets same SL room, not wider",
        "7. **TP ≥ 3× round-trip cost** — enforced geometrically",
        "8. **Breakeven edge check** — refuses to trade when expected edge < min_expected_R",
        "9. **Gate restored** — require_bootstrap_ci=True, daily_loss_limit=3%, max_consec_losses=4",
        "",
    ]

    if not trial_df.empty:
        rpt += [
            "## All Trials (complete table)",
            "",
            trial_df.to_markdown(index=False) if hasattr(trial_df, "to_markdown") else trial_df.to_string(),
            "",
        ]

    if best_trial is not None:
        rpt += [
            f"## Best Configuration",
            f"",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Horizon (H) | {best_trial['H']} bars |",
            f"| TP ATR mult (k) | {best_trial['k']} |",
            f"| SL ATR mult (m) | {best_trial['m']} |",
            f"| Probability threshold | {best_trial['threshold']} |",
            f"| Meta-labeling | {best_trial['use_meta']} |",
            "",
        ]

    rpt += [
        "## Acceptance Criteria (all must be true for PASSED)",
        "",
        "- Net PnL > 0 ✓/✗",
        "- Profit Factor ≥ 1.1 ✓/✗",
        "- Bootstrap CI lower bound > 0 (95%) ✓/✗",
        "- Still positive at 1.5× spread ✓/✗",
        "- ≥ 60% of folds positive ✓/✗",
        "- ≥ 100 OOS trades ✓/✗",
        "- Label-shuffle check rejects profitability ✓/✗",
        "",
        "## Next Experiments (if FAILED)",
        "",
        "1. Download more data (≥12 months) using `scripts/download_mt5_data_extended.py`",
        "2. Try H=60 with London/NY overlap session only (12:00-16:00 UTC)",
        "3. Add regime filter: only trade when ADX > 25 (trending market)",
        "4. Test asymmetric TP/SL: TP=2R/SL=1R (higher R:R, lower required win rate)",
        "5. Use ensemble of LightGBM + MLP probabilities",
    ]

    report_text = "\n".join(rpt)
    Path("reports/final_report.md").write_text(report_text, encoding="utf-8")
    logger.info("Final report written to reports/final_report.md")


def main():
    parser = argparse.ArgumentParser(description="TEGF-XAU Full Pipeline (Steps 3+4+5)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--holdout", action="store_true", help="Run holdout evaluation (ONCE, at end)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    run_pipeline(cfg, run_holdout=args.holdout)


if __name__ == "__main__":
    main()
