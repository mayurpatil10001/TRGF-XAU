"""
src/data/loader.py
Loads, inspects, and cleans raw XAUUSD 1-minute CSV data.

Steps (each logged with row-count delta):
1. Infer delimiter/header/columns/timezone
2. Parse timestamps, convert to UTC, sort
3. Drop exact duplicate timestamps (keep last)
4. Drop NaN OHLC, non-positive prices, high<low, OHLC sanity band
5. Mark gap_before_minutes; flag windows spanning large gaps
6. Session tagging
7. Save to data/processed/bars.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import load_config, Config
from src.common.logging import setup_logging, get_logger
from src.common.timeutils import get_session

logger = get_logger("data.loader")

COLUMN_ALIASES = {
    # common names -> canonical
    "time": "time", "date": "time", "datetime": "time", "timestamp": "time",
    "open": "open", "o": "open",
    "high": "high", "h": "high",
    "low": "low", "l": "low",
    "close": "close", "c": "close",
    "volume": "volume", "vol": "volume",
    "tick_volume": "tick_volume", "tickvolume": "tick_volume", "tick_vol": "tick_volume",
    "spread": "spread",
}


def _infer_schema(path: Path) -> dict:
    """Sniff delimiter, header, and column names."""
    with open(path, "r", encoding="utf-8-sig") as f:
        head = [f.readline() for _ in range(5)]
    for delim in [",", "\t", ";"]:
        if delim in head[0]:
            break
    cols_raw = head[0].strip().split(delim)
    cols_lower = [c.strip().lower().replace(" ", "_") for c in cols_raw]
    mapping = {}
    for raw, low in zip(cols_raw, cols_lower):
        if low in COLUMN_ALIASES:
            mapping[raw] = COLUMN_ALIASES[low]
    return {"delimiter": delim, "raw_columns": cols_raw, "mapping": mapping}


def _load_raw(path: Path, schema: dict) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=schema["delimiter"],
        header=0,
        low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=schema["mapping"])
    return df


def _parse_time(df: pd.DataFrame, tz_in: str) -> pd.DataFrame:
    """Parse the time column and convert to UTC."""
    col = df["time"]
    # try multiple formats
    for fmt in [None, "%Y-%m-%d %H:%M:%S", "%Y.%m.%d %H:%M",
                "%m/%d/%Y %H:%M", "%Y-%m-%dT%H:%M:%S"]:
        try:
            if fmt:
                parsed = pd.to_datetime(col, format=fmt, utc=False)
            else:
                parsed = pd.to_datetime(col, utc=False)
            # If already tz-aware, convert; else localize
            if parsed.dt.tz is not None:
                parsed = parsed.dt.tz_convert("UTC")
            else:
                if tz_in.upper() == "UTC":
                    parsed = parsed.dt.tz_localize("UTC")
                else:
                    parsed = parsed.dt.tz_localize(tz_in, ambiguous="infer",
                                                    nonexistent="shift_forward").dt.tz_convert("UTC")
            df["time"] = parsed
            return df
        except Exception:
            continue
    raise ValueError("Could not parse the 'time' column. Check timezone_in in config.")


def _wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply all cleaning steps; log row-count deltas."""
    n0 = len(df)
    logger.info(f"Raw row count: {n0:,}")

    # ── 1. Sort by time ─────────────────────────────────────────────────────
    df = df.sort_values("time").reset_index(drop=True)
    logger.info("Sorted by time.")

    # ── 2. Drop exact duplicate timestamps (keep last) ───────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    logger.info(f"Dropped {before - len(df):,} duplicate timestamps.")

    # ── 3. Drop NaN OHLC ────────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    logger.info(f"Dropped {before - len(df):,} rows with NaN OHLC.")

    # ── 4. Non-positive prices ───────────────────────────────────────────────
    before = len(df)
    mask = (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)
    df = df[mask].reset_index(drop=True)
    logger.info(f"Dropped {before - len(df):,} rows with non-positive prices.")

    # ── 5. Minimum price sanity ──────────────────────────────────────────────
    before = len(df)
    df = df[df["close"] >= cfg.data.min_price].reset_index(drop=True)
    logger.info(f"Dropped {before - len(df):,} rows below min_price={cfg.data.min_price}.")

    # ── 6. high < low ───────────────────────────────────────────────────────
    before = len(df)
    df = df[df["high"] >= df["low"]].reset_index(drop=True)
    logger.info(f"Dropped {before - len(df):,} rows with high < low.")

    # ── 7. ATR sanity band (bad ticks) ──────────────────────────────────────
    atr = _wilder_atr(df, cfg.features.atr_period)
    band = 5 * atr
    mid = (df["open"] + df["close"]) / 2
    bad_mask = (
        (df["high"] > mid + band) |
        (df["low"] < mid - band) |
        ((df["open"] - df["close"]).abs() > band)
    )
    before = len(df)
    df = df[~bad_mask].reset_index(drop=True)
    logger.info(f"Dropped {before - len(df):,} rows flagged as bad ticks (5*ATR band).")

    # ── 8. Gap detection ────────────────────────────────────────────────────
    time_diff = df["time"].diff().dt.total_seconds() / 60.0
    df["gap_before_minutes"] = time_diff.fillna(0.0)

    # ── 9. Session tagging ───────────────────────────────────────────────────
    session_map = {k: (v[0], v[1]) for k, v in cfg.sessions.model_dump().items()}
    df["session"] = df["time"].apply(lambda t: get_session(t, session_map))

    # ── 10. Filter to main_start (or full period) ───────────────────────────
    full_start = "2001-01-01" if cfg.data.full_period_experiment else cfg.data.main_start
    df = df[df["time"] >= pd.Timestamp(full_start, tz="UTC")].reset_index(drop=True)
    logger.info(f"Filtered to >= {full_start}: {len(df):,} rows remain.")

    logger.info(f"Final clean bar count: {len(df):,} | First: {df['time'].iloc[0]} | Last: {df['time'].iloc[-1]}")
    return df


def inspect_file(path: Path) -> None:
    """Print diagnostic info about the raw CSV."""
    schema = _infer_schema(path)
    logger.info(f"Detected delimiter: {repr(schema['delimiter'])}")
    logger.info(f"Raw columns: {schema['raw_columns']}")
    logger.info(f"Column mapping: {schema['mapping']}")
    # Peek first/last rows
    df = _load_raw(path, schema)
    logger.info(f"First row:\n{df.head(1).to_string()}")
    logger.info(f"Last row:\n{df.tail(1).to_string()}")
    logger.info(f"Shape: {df.shape}")


def load_and_clean(cfg: Config) -> pd.DataFrame:
    """Full pipeline: inspect -> load -> parse time -> clean -> return."""
    path = Path(cfg.data.path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}\n"
            "Place your XAUUSD 1-minute CSV at data/raw/xauusd_1m.csv "
            "(or update config data.path)."
        )

    schema = _infer_schema(path)
    logger.info(f"[SCHEMA] delimiter={repr(schema['delimiter'])}, mapping={schema['mapping']}")

    df = _load_raw(path, schema)
    logger.info(f"Loaded {len(df):,} raw rows from {path}")

    if "time" not in df.columns:
        # Try to use the index if it looks like a time
        df = df.reset_index().rename(columns={"index": "time"})

    df = _parse_time(df, cfg.data.timezone_in)

    # Ensure volume column exists (tick_volume fallback)
    if "volume" not in df.columns and "tick_volume" in df.columns:
        df["volume"] = df["tick_volume"]
        logger.info("Using tick_volume as volume.")
    elif "volume" not in df.columns:
        df["volume"] = 0.0
        logger.warning("No volume or tick_volume column found; setting volume=0.")

    if "tick_volume" not in df.columns:
        df["tick_volume"] = df["volume"]

    if "spread" not in df.columns:
        logger.info(
            f"No spread column found; using configured constant spread "
            f"={cfg.costs.spread_price_units} price units. "
            "NOTE: bid-only data → ask = bid + spread_constant. "
            "Pre-2015 spread was structurally different; flagged in QC report."
        )
        df["spread"] = cfg.costs.spread_price_units
        df["spread_is_synthetic"] = True
    else:
        df["spread_is_synthetic"] = False

    df = clean(df, cfg)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)

    df = load_and_clean(cfg)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bars.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved cleaned bars to {out_path}")

    # Generate QC report
    from src.data.qc_report import generate_qc_report
    generate_qc_report(df, cfg)


if __name__ == "__main__":
    main()
