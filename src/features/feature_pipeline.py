"""
src/features/feature_pipeline.py

Full feature engineering pipeline (Groups A–G).
All features are strictly causal (use only bars <= t).
Normalization (RobustScaler) is FIT ON TRAIN FOLD ONLY and applied here.

Usage:
  python -m src.features.feature_pipeline --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, Config
from src.common.logging import setup_logging, get_logger
from src.features.sessions import session_features
from src.features.timing import timing_proxy

logger = get_logger("features.pipeline")

EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Wilder ATR (causal)
# ─────────────────────────────────────────────────────────────────────────────

def wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Group A – Returns
# ─────────────────────────────────────────────────────────────────────────────

def group_a(df: pd.DataFrame, lookback: int, atr: pd.Series) -> pd.DataFrame:
    out = {}
    r_close = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    atr_norm = (atr / (df["close"] + EPS)).fillna(EPS)

    for lag in range(lookback):
        r = r_close.shift(lag)
        out[f"r_close_lag{lag}"] = r
        out[f"r_close_lag{lag}_atr"] = r / (atr_norm + EPS)

    # Current bar price relationships
    h, l, c, o = df["high"], df["low"], df["close"], df["open"]
    out["ln_h_c"] = np.log(h / (c + EPS))
    out["ln_l_c"] = np.log(l / (c + EPS))
    out["ln_h_l"] = np.log((h + EPS) / (l + EPS))
    out["ln_c_o"] = np.log((c + EPS) / (o + EPS))
    rng = h - l + EPS
    out["ln_h_c_atr"] = out["ln_h_c"] / (atr_norm + EPS)
    out["ln_l_c_atr"] = out["ln_l_c"] / (atr_norm + EPS)
    out["ln_h_l_atr"] = out["ln_h_l"] / (atr_norm + EPS)
    out["ln_c_o_atr"] = out["ln_c_o"] / (atr_norm + EPS)
    return pd.DataFrame(out, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Group B – Bar shape
# ─────────────────────────────────────────────────────────────────────────────

def group_b(df: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
    h, l, c, o = df["high"], df["low"], df["close"], df["open"]
    rng = h - l + EPS
    out = {
        "bar_body":       (c - o) / rng,
        "bar_upper_wick": (h - np.maximum(o, c)) / rng,
        "bar_lower_wick": (np.minimum(o, c) - l) / rng,
        "bar_close_loc":  (c - l) / rng,
        "bar_range_atr":  rng / (atr + EPS),
    }
    return pd.DataFrame(out, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Group C – Volume
# ─────────────────────────────────────────────────────────────────────────────

def group_c(df: pd.DataFrame) -> pd.DataFrame:
    vol = df["tick_volume"].copy().clip(lower=0.0)
    log_vol = np.log1p(vol)
    out: dict = {"vol_log": log_vol}

    windows = [5, 15, 60, 300]  # bars = minutes; 300 ≈ 5 h
    for w in windows:
        roll = log_vol.rolling(w, min_periods=1)
        out[f"vol_roll_mean_{w}"] = roll.mean()
        out[f"vol_roll_std_{w}"] = roll.std().fillna(0.0)
        med = roll.median()
        mad = (log_vol - med).abs().rolling(w, min_periods=1).median()
        out[f"vol_z_{w}"] = (log_vol - med) / (mad + EPS)

    # Classic z-score over 60
    mu60 = log_vol.rolling(60, min_periods=1).mean()
    sd60 = log_vol.rolling(60, min_periods=1).std().fillna(EPS)
    out["vol_z_classic_60"] = (log_vol - mu60) / (sd60 + EPS)

    # Dollar volume proxy
    out["dollar_vol"] = np.log1p(vol * df["close"])

    # Volume ratio to same-minute-of-day causal expanding mean
    min_of_day = df["time"].dt.hour * 60 + df["time"].dt.minute
    df_tmp = pd.DataFrame({"log_vol": log_vol, "mod": min_of_day})
    # causal expanding mean per minute-of-day group
    expanding_means = df_tmp.groupby("mod")["log_vol"].expanding().mean().reset_index(level=0, drop=True)
    out["vol_ratio_tod"] = log_vol / (expanding_means + EPS)

    return pd.DataFrame(out, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Group D – Volatility
# ─────────────────────────────────────────────────────────────────────────────

def group_d(df: pd.DataFrame, atr: pd.Series, vol_windows: List[int]) -> pd.DataFrame:
    r = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    h, l, c, o = df["high"], df["low"], df["close"], df["open"]
    out: dict = {}

    for w in vol_windows:
        out[f"rv_std_{w}"] = r.rolling(w, min_periods=2).std()

    # Parkinson volatility (uses H/L; causal rolling)
    hl_sq = (np.log(h / (l + EPS)) ** 2)
    for w in vol_windows:
        out[f"rv_parkinson_{w}"] = np.sqrt(
            hl_sq.rolling(w, min_periods=1).mean() / (4 * np.log(2))
        )

    out["atr_c_ratio"] = atr / (c + EPS)
    # Use first and last configured vol_windows for ratio
    w_short = vol_windows[0]   # e.g. 5
    w_long  = vol_windows[-1]  # e.g. 30 (whatever is last in config)
    out["vol_ratio_short_long"] = out[f"rv_std_{w_short}"] / (out[f"rv_std_{w_long}"] + EPS)

    # Realized-vol percentile rank over last 20*390 bars (~20 trading days)
    rv_long = out[f"rv_std_{w_long}"]
    rv_long_rank = rv_long.rolling(20 * 390, min_periods=w_long).rank(pct=True)
    out[f"rv{w_long}_pctrank_20d"] = rv_long_rank

    return pd.DataFrame(out, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Group F – Time/Session (delegates to sessions.py)
# ─────────────────────────────────────────────────────────────────────────────

def group_f(df: pd.DataFrame, atr: pd.Series, sessions: dict) -> pd.DataFrame:
    df_s = session_features(df, sessions)
    # VWAP distance (NY-open anchor, reset daily at 16:00 UTC)
    # Causal VWAP within trading day (rough approximation)
    daily_group = df["time"].dt.date
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["tick_volume"].clip(lower=1.0)
    cum_tp_vol = (typical * vol).groupby(daily_group).transform("cumsum")
    cum_vol = vol.groupby(daily_group).transform("cumsum")
    vwap = cum_tp_vol / cum_vol
    vwap_dist = (df["close"] - vwap) / (atr + EPS)

    cols = {k: df_s[k] for k in df_s.columns if k not in df.columns}
    cols["vwap_dist_atr"] = vwap_dist
    return pd.DataFrame(cols, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Group G – Trend / mean reversion
# ─────────────────────────────────────────────────────────────────────────────

def group_g(df: pd.DataFrame, atr: pd.Series, ema_periods: List[int]) -> pd.DataFrame:
    c = df["close"]
    out: dict = {}
    for p in ema_periods:
        ema = c.ewm(span=p, adjust=False).mean()
        out[f"ema{p}_dist_atr"] = (c - ema) / (atr + EPS)

    # EMA20 slope over 5 bars / ATR
    ema20 = c.ewm(span=20, adjust=False).mean()
    out["ema20_slope5_atr"] = (ema20 - ema20.shift(5)) / (atr + EPS)

    # RSI(14) scaled to [-1, 1]
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = gain / (loss + EPS)
    out["rsi14"] = (100.0 / (1 + rs) - 50) / 50  # scaled [-1, 1]

    # Distance to rolling 60-bar high/low
    h60 = df["high"].rolling(60, min_periods=1).max()
    l60 = df["low"].rolling(60, min_periods=1).min()
    out["dist_to_60h_atr"] = (h60 - c) / (atr + EPS)
    out["dist_to_60l_atr"] = (c - l60) / (atr + EPS)

    return pd.DataFrame(out, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Master build function
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the full feature matrix.
    Returns:
      feat_df: DataFrame aligned with df (same index), columns = feature names
      feature_names: ordered list of feature column names
    """
    logger.info(f"Building features for {len(df):,} bars...")

    atr = wilder_atr(df, cfg.features.atr_period)

    session_map = {k: (v[0], v[1]) for k, v in cfg.sessions.model_dump().items()}

    parts = [
        group_a(df, cfg.features.lookback, atr),
        group_b(df, atr),
        group_c(df),
        group_d(df, atr, cfg.features.vol_windows),
        timing_proxy(df)[["timing_high_first_proxy", "timing_surprise_proxy", "timing_continuous_proxy"]],
        group_f(df, atr, session_map),
        group_g(df, atr, cfg.features.ema_periods),
    ]
    feat_df = pd.concat(parts, axis=1)

    # ── Validate no lookahead: feature at t must not use any row > t ──────────
    # (the only forward references would be from shift(-n) which we never use)

    # Drop rows with all-NaN (mostly the first lookback bars)
    n_before = len(feat_df)
    feat_df = feat_df.dropna(how="all")
    logger.info(f"Dropped {n_before - len(feat_df):,} all-NaN feature rows (lookback warmup).")

    # Also drop rows where >50% of features are NaN
    thresh = int(feat_df.shape[1] * 0.5)
    feat_df = feat_df.dropna(thresh=thresh)

    feature_names = feat_df.columns.tolist()
    logger.info(f"Feature matrix: {feat_df.shape[0]:,} rows x {len(feature_names)} features")

    return feat_df, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Normalization (scaler fit on train fold only)
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaler(feat_df: pd.DataFrame, train_mask: pd.Series) -> RobustScaler:
    """Fit RobustScaler on the training rows only."""
    scaler = RobustScaler()
    scaler.fit(feat_df.loc[train_mask])
    return scaler


def apply_scaler(feat_df: pd.DataFrame, scaler: RobustScaler, clip: float = 10.0) -> pd.DataFrame:
    """Apply scaler and clip outliers."""
    arr = scaler.transform(feat_df.values)
    arr = np.clip(arr, -clip, clip)
    return pd.DataFrame(arr, index=feat_df.index, columns=feat_df.columns)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)

    bars_path = Path("data/processed/bars.parquet")
    if not bars_path.exists():
        raise FileNotFoundError(f"Run data loader first: {bars_path} not found.")

    df = pd.read_parquet(bars_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    feat_df, feature_names = build_features(df, cfg)

    # Align with bars (left join on index)
    out_df = df.loc[feat_df.index].copy()
    for col in feature_names:
        out_df[col] = feat_df[col].values

    out_path = Path("data/processed/features.parquet")
    out_df.to_parquet(out_path, index=False)
    logger.info(f"Saved features to {out_path}")

    # Save feature name list
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    feat_list_path = artifacts_dir / "feature_names.json"
    with open(feat_list_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"Saved feature names to {feat_list_path}")


if __name__ == "__main__":
    main()
