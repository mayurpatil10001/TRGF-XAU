"""
src/training/walk_forward.py
Orchestrates the full purged walk-forward training and produces per-fold artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, Config
from src.common.logging import setup_logging, get_logger
from src.common.seed import set_global_seed
from src.training.windows import generate_folds
from src.training.trainer import train_model, predict, get_device
from src.features.feature_pipeline import fit_scaler, apply_scaler
from src.models.lgbm_baseline import train_lgbm, predict_lgbm, save_lgbm

logger = get_logger("training.walk_forward")


def _build_sequence_tensor(
    feat_arr: np.ndarray, idx: np.ndarray, lookback: int
) -> np.ndarray:
    """Build [N, L, F] sequence tensor for deep models."""
    F = feat_arr.shape[1]
    seqs = np.zeros((len(idx), lookback, F), dtype=np.float32)
    for i, bar_i in enumerate(idx):
        start = max(0, bar_i - lookback + 1)
        seg = feat_arr[start:bar_i + 1]
        seqs[i, lookback - len(seg):, :] = seg
    return seqs


def run_walk_forward(cfg: Config) -> Dict[str, Any]:
    """Run the full walk-forward pipeline. Returns concatenated OOS predictions."""
    set_global_seed(cfg.seed)
    device = get_device()

    # Load data
    feat_path = Path("data/processed/labels.parquet")
    if not feat_path.exists():
        raise FileNotFoundError(f"Run label pipeline first: {feat_path}")

    df = pd.read_parquet(feat_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Load feature names
    with open("artifacts/feature_names.json") as f:
        feature_names = json.load(f)

    # Filter to dev period only
    dev_end_ts = pd.Timestamp(cfg.data.dev_end, tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    df = df[df["time"] <= dev_end_ts].reset_index(drop=True)

    feat_arr = df[feature_names].values.astype(np.float32)
    y_cls = df["label"].values.astype(np.int64)
    y_reg = df["y_reg"].fillna(0.0).values.astype(np.float32)
    times = df["time"].reset_index(drop=True)

    folds = generate_folds(times, cfg)
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    all_test_results = []

    for fold in folds:
        logger.info(f"\n{'='*60}\nFold {fold.fold_id}\n{'='*60}")

        tr_idx, va_idx, te_idx = fold.train_idx, fold.val_idx, fold.test_idx
        if len(tr_idx) < 100 or len(va_idx) < 10 or len(te_idx) < 10:
            logger.warning(f"Fold {fold.fold_id}: too few samples, skipping.")
            continue

        # ── Fit scaler on train only ─────────────────────────────────────────
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[tr_idx] = True
        scaler = fit_scaler(pd.DataFrame(feat_arr, columns=feature_names), pd.Series(train_mask))
        feat_df_scaled = apply_scaler(pd.DataFrame(feat_arr, columns=feature_names), scaler)
        feat_scaled = feat_df_scaled.values.astype(np.float32)

        # Save scaler
        fold_art_dir = artifacts_dir / f"fold_{fold.fold_id:02d}"
        fold_art_dir.mkdir(exist_ok=True)
        joblib.dump(scaler, fold_art_dir / "scaler.joblib")

        # ── LightGBM (tabular) ───────────────────────────────────────────────
        X_tr_tab = feat_scaled[tr_idx]
        X_va_tab = feat_scaled[va_idx]
        X_te_tab = feat_scaled[te_idx]

        lgbm_model = train_lgbm(X_tr_tab, y_cls[tr_idx], X_va_tab, y_cls[va_idx], cfg, seed=cfg.seed)
        save_lgbm(lgbm_model, fold_art_dir / "lgbm.txt")
        te_probs_lgbm = predict_lgbm(lgbm_model, X_te_tab)

        # ── Deep models ──────────────────────────────────────────────────────
        X_tr_seq = _build_sequence_tensor(feat_scaled, tr_idx, cfg.features.lookback)
        X_va_seq = _build_sequence_tensor(feat_scaled, va_idx, cfg.features.lookback)
        X_te_seq = _build_sequence_tensor(feat_scaled, te_idx, cfg.features.lookback)

        F_dim = feat_scaled.shape[1]
        deep_models = _instantiate_models(cfg, F_dim, cfg.features.lookback)

        fold_preds = {
            "lgbm": te_probs_lgbm,
            "times": df["time"].iloc[te_idx].values,
            "y_cls": y_cls[te_idx],
            "y_reg": y_reg[te_idx],
            "fold_id": fold.fold_id,
        }

        for model_name, model in deep_models.items():
            logger.info(f"  Training {model_name}...")
            # MLP uses flat tabular features (no sequence), all others use sequence
            if model_name == "mlp":
                X_tr_m, X_va_m, X_te_m = X_tr_tab, X_va_tab, X_te_tab
            else:
                X_tr_m, X_va_m, X_te_m = X_tr_seq, X_va_seq, X_te_seq
            try:
                result = train_model(
                    model,
                    X_tr_m, y_cls[tr_idx], y_reg[tr_idx],
                    X_va_m, y_cls[va_idx], y_reg[va_idx],
                    cfg, lr=1e-3, save_path=fold_art_dir / f"{model_name}.pt",
                )
                probs = predict(result["model"], X_te_m, device)
                fold_preds[model_name] = probs
                logger.info(f"    {model_name}: best_val_loss={result['best_val_loss']:.4f}")
            except RuntimeError as e:
                if "memory" in str(e).lower() or "alloc" in str(e).lower():
                    logger.warning(f"  {model_name}: OOM - skipping. Error: {e}")
                    import torch
                    torch.cuda.empty_cache()
                    fold_preds[model_name] = None
                else:
                    raise

        all_test_results.append(fold_preds)

    # Save combined OOS predictions
    out = {}
    for key in ["lgbm", "mlp", "lstm", "transformer", "gated_fusion"]:
        try:
            out[key] = np.concatenate([f[key] for f in all_test_results], axis=0)
        except Exception:
            pass
    out["times"] = np.concatenate([f["times"] for f in all_test_results])
    out["y_cls"] = np.concatenate([f["y_cls"] for f in all_test_results])
    out["y_reg"] = np.concatenate([f["y_reg"] for f in all_test_results])

    np.save(artifacts_dir / "oos_predictions.npy", out)
    logger.info("Walk-forward complete. OOS predictions saved.")
    return out


def _instantiate_models(cfg: Config, F_dim: int, lookback: int) -> Dict:
    """Instantiate all deep models."""
    from src.models.mlp import MLP
    from src.models.lstm import LSTMModel
    from src.models.transformer import TransformerModel
    from src.models.gated_fusion import GatedFusion

    models = {}
    if "mlp" in cfg.model.names:
        # MLP takes flat tabular features, NOT flattened sequences
        models["mlp"] = MLP(input_dim=F_dim, dropout=0.1)
    if "lstm" in cfg.model.names:
        models["lstm"] = LSTMModel(input_dim=F_dim, dropout=0.1)
    if "transformer" in cfg.model.names:
        models["transformer"] = TransformerModel(input_dim=F_dim, dropout=0.1, max_len=lookback + 10)
    if "gated_fusion" in cfg.model.names:
        models["gated_fusion"] = GatedFusion(input_dim=F_dim, lookback=lookback, dropout=0.1)
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    run_walk_forward(cfg)


if __name__ == "__main__":
    main()
