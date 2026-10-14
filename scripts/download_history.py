"""
scripts/download_history.py
============================
Downloads 2+ years of XAUUSD 1M history by combining:
  1. Live Exness MT5 data  (most recent ~3 months, highest quality)
  2. Yahoo Finance GC=F    (Gold Futures, up to 60 days of 1-min, or
                            1-hour bars resampled to synthetic 1-min)
  3. Stooq XAUUSD hourly   (free, no API key, downloadable via pandas-datareader)

Strategy for 2-year dataset:
  - For 1-min resolution going back 2 years: we use Gold Futures hourly data
    from Yahoo Finance (GC=F goes back 2 years hourly) and generate synthetic
    1-min OHLCV by interpolating/resampling. This is standard practice for
    strategy development when historical tick data is unavailable.
  - The synthetic bars share the same hour-level price trajectory but with
    added uniform micro-noise to simulate intra-hour bar variability.
  - MT5 live bars are spliced on top for the most recent period.

IMPORTANT: Synthetic 1-min bars will have slightly lower OOS validity than
real tick-data bars, but are sufficient for multi-fold walk-forward training.
The pipeline's out-of-sample test period will use real MT5 bars only.

Usage:
    python scripts/download_history.py
"""
from __future__ import annotations

import os
import sys
import time
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

import MetaTrader5 as mt5

SYMBOL_CANDIDATES = ["XAUUSDm", "XAUUSD", "XAUUSD.a"]
OUT_PATH = Path("data/raw/xauusd_1m.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
UTC = timezone.utc


# ─── MT5 LIVE DATA ──────────────────────────────────────────────────────────

def connect_mt5() -> tuple[str | None, float]:
    path = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
    if Path(path).exists():
        mt5.initialize(path=path)
    else:
        mt5.initialize()
    login    = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server   = os.getenv("MT5_SERVER")
    if login and password and server:
        mt5.login(int(login), password=password, server=server)
        acc = mt5.account_info()
        if acc:
            print(f"MT5: #{acc.login}  balance={acc.balance:,.2f}")
    for sym in SYMBOL_CANDIDATES:
        if mt5.symbol_info(sym) is not None:
            mt5.symbol_select(sym, True)
            time.sleep(1.5)
            info = mt5.symbol_info(sym)
            return sym, (info.point if info else 0.001)
    return None, 0.001


def fetch_mt5_recent(symbol: str, point: float, days: int = 110) -> pd.DataFrame | None:
    """Pull the most recent 'days' worth of M1 bars from MT5."""
    now   = datetime.now(UTC).replace(second=0, microsecond=0)
    start = now - timedelta(days=days)
    r = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, now)
    if r is None or len(r) == 0:
        print(f"  MT5 fetch failed: {mt5.last_error()}")
        return None
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    # Fix spread
    if "spread" in df.columns:
        raw = df["spread"].mean()
        if raw > 1.0:
            df["spread"] = df["spread"] * point
        if not (0.01 <= df["spread"].mean() <= 5.0):
            df["spread"] = 0.25
    else:
        df["spread"] = 0.25
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0
    return df[["time","open","high","low","close","tick_volume","spread"]]


# ─── YAHOO FINANCE HOURLY DATA ────────────────────────────────────────────────

def fetch_yahoo_hourly(years_back: int = 2) -> pd.DataFrame | None:
    """Download Gold Futures (GC=F) hourly OHLCV from Yahoo Finance.
    yfinance supports up to 730 days of 1-hour data for futures.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed. Run: pip install yfinance")
        return None

    print(f"\nFetching Yahoo Finance GC=F hourly data ({years_back} years)...")
    try:
        ticker = yf.Ticker("GC=F")
        df = ticker.history(period=f"{years_back}y", interval="1h", auto_adjust=True)
        if df is None or len(df) == 0:
            # Fallback: try XAUUSD=X
            ticker = yf.Ticker("XAUUSD=X")
            df = ticker.history(period=f"{years_back}y", interval="1h", auto_adjust=True)
    except Exception as e:
        print(f"  Yahoo fetch error: {e}")
        return None

    if df is None or len(df) == 0:
        print("  Yahoo Finance returned no data")
        return None

    df = df.reset_index()
    # Normalise column names
    df.columns = [c.lower() for c in df.columns]
    time_col = "datetime" if "datetime" in df.columns else "date"
    df = df.rename(columns={time_col: "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if "volume" not in df.columns:
        df["volume"] = 0
    df = df[["time","open","high","low","close","volume"]].copy()
    df = df.dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    print(f"  Yahoo: {len(df):,} hourly bars  [{df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()}]")
    return df


def hourly_to_1min_synthetic(h1_df: pd.DataFrame, spread: float = 0.25,
                               rng_seed: int = 42) -> pd.DataFrame:
    """
    Resample 1-hour OHLCV bars into synthetic 1-minute bars.

    Method:
    - Open  of first minute  = Hour Open
    - Close of last  minute  = Hour Close
    - High/Low of any minute = interpolated, with micro-noise
    - Volume = uniform split across 60 minutes
    - Spread = constant (0.25 for XAUUSD)

    This is a standard technique for walk-forward training when tick data
    is unavailable. Label integrity is preserved because the pipeline
    recomputes labels from OHLC.
    """
    rng = np.random.default_rng(rng_seed)
    minute_rows = []

    for _, row in h1_df.iterrows():
        t0    = row["time"]
        op    = float(row["open"])
        hi    = float(row["high"])
        lo    = float(row["low"])
        cl    = float(row["close"])
        vol   = float(row.get("volume", 0))

        # Build a smooth path: open → close, clipped to [lo, hi]
        path  = np.linspace(op, cl, 60)
        # Add noise scaled to intra-bar range
        noise_scale = (hi - lo) * 0.08
        noise = rng.normal(0, noise_scale, 60)
        path  = np.clip(path + noise, lo, hi)
        # Enforce open/close exactly
        path[0]  = op
        path[-1] = cl

        vol_per_min = max(1, int(vol / 60))

        for m in range(60):
            t = t0 + timedelta(minutes=m)
            p = path[m]
            # Bar OHLCV
            p_next = path[min(m+1, 59)]
            b_open  = p
            b_close = p_next
            b_hi    = max(p, p_next) + abs(rng.normal(0, noise_scale * 0.3))
            b_lo    = min(p, p_next) - abs(rng.normal(0, noise_scale * 0.3))
            b_hi    = min(b_hi, hi)
            b_lo    = max(b_lo, lo)
            minute_rows.append({
                "time":        t,
                "open":        round(b_open,  3),
                "high":        round(b_hi,    3),
                "low":         round(b_lo,    3),
                "close":       round(b_close, 3),
                "tick_volume": vol_per_min,
                "spread":      spread,
            })

    df = pd.DataFrame(minute_rows)
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df


# ─── CONFIG UPDATER ───────────────────────────────────────────────────────────

def update_config(first_dt: datetime, last_dt: datetime):
    cfg_path = Path("config/config.yaml")
    if not cfg_path.exists():
        return
    text = cfg_path.read_text(encoding="utf-8")

    holdout_dt  = (last_dt.replace(day=1) - timedelta(days=1)).replace(day=1)
    dev_end_dt  = holdout_dt - timedelta(days=1)

    text = re.sub(r'(main_start:\s*")[^"]*(")', rf'\g<1>{first_dt.strftime("%Y-%m-%d")}\2', text)
    text = re.sub(r'(dev_end:\s*")[^"]*(")',    rf'\g<1>{dev_end_dt.strftime("%Y-%m-%d")}\2', text)
    text = re.sub(r'(holdout_start:\s*")[^"]*(")', rf'\g<1>{holdout_dt.strftime("%Y-%m-%d")}\2', text)
    text = re.sub(r'(train_months:\s*)\d+', r'\g<1>6', text)
    text = re.sub(r'(val_months:\s*)\d+',   r'\g<1>2', text)
    text = re.sub(r'(test_months:\s*)\d+',  r'\g<1>2', text)
    text = re.sub(r'(step_months:\s*)\d+',  r'\g<1>2', text)

    cfg_path.write_text(text, encoding="utf-8")
    print(f"config.yaml updated: {first_dt.date()} → dev_end={dev_end_dt.date()}, "
          f"holdout={holdout_dt.date()}, folds=6/2/2mo")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("TEGF-XAU  History Downloader")
    print("="*60)

    # --- Step 1: MT5 live bars (most recent ~110 days, highest quality) ---
    print("\n[1/3] Fetching recent bars from Exness MT5...")
    sym, point = connect_mt5()
    mt5_df = None
    mt5_start = None
    if sym:
        mt5_df = fetch_mt5_recent(sym, point, days=110)
        if mt5_df is not None:
            mt5_start = mt5_df["time"].min()
            print(f"  MT5: {len(mt5_df):,} bars  [{mt5_df['time'].min().date()} → {mt5_df['time'].max().date()}]")
        mt5.shutdown()
    else:
        mt5.shutdown()
        print("  MT5 unavailable — will use Yahoo Finance only")

    # --- Step 2: Yahoo Finance hourly gold data (up to 2 years) ---
    print("\n[2/3] Fetching Gold hourly data from Yahoo Finance...")
    yf_hourly = fetch_yahoo_hourly(years_back=2)

    if yf_hourly is None or len(yf_hourly) < 100:
        if mt5_df is not None and len(mt5_df) > 10_000:
            print("\nYahoo unavailable — using MT5-only dataset (3 months)")
            combined = mt5_df
        else:
            print("\nERROR: No data source available.")
            print("Install yfinance:  pip install yfinance")
            sys.exit(1)
    else:
        # Trim Yahoo data to before MT5 data starts (avoid overlap)
        if mt5_start is not None:
            yf_hourly = yf_hourly[yf_hourly["time"] < mt5_start].copy()

        print(f"\n[3/3] Converting {len(yf_hourly):,} hourly bars → synthetic 1-min...")
        yf_1min = hourly_to_1min_synthetic(yf_hourly, spread=0.25)
        print(f"  Generated: {len(yf_1min):,} synthetic 1-min bars")

        # Combine: synthetic history + real MT5 bars
        parts = [yf_1min]
        if mt5_df is not None:
            parts.append(mt5_df)
        combined = pd.concat(parts, ignore_index=True)
        combined = combined.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    # --- Save ---
    out = combined.copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(OUT_PATH, index=False)

    first_dt    = combined["time"].iloc[0]
    last_dt     = combined["time"].iloc[-1]
    span_months = (last_dt - first_dt).days / 30.5
    mb          = OUT_PATH.stat().st_size / 1e6
    real_bars   = len(mt5_df) if mt5_df is not None else 0
    synth_bars  = len(combined) - real_bars

    print(f"\n{'='*60}")
    print(f"SAVED: {OUT_PATH}")
    print(f"  Total rows:  {len(out):,}")
    print(f"  Real bars:   {real_bars:,} (from MT5)")
    print(f"  Synth bars:  {synth_bars:,} (from Yahoo hourly)")
    print(f"  Size:        {mb:.1f} MB")
    print(f"  Range:       {first_dt.strftime('%Y-%m-%d')} → {last_dt.strftime('%Y-%m-%d')}")
    print(f"  Span:        {span_months:.1f} months")
    print(f"{'='*60}")

    if span_months >= 12:
        update_config(first_dt, last_dt)
        n_folds = max(1, int((span_months - 12) / 2) + 1)
        print(f"\n  Expected walk-forward folds: ~{n_folds}")
        print(f"  Real MT5 bars used as holdout: last ~3 months")
        print(f"\nRe-running pipeline now...")
        os.execv(sys.executable, [sys.executable,
                  "scripts/run_full_pipeline.py", "--config", "config/config.yaml"])
    else:
        print(f"\nSpan {span_months:.1f}mo < 12mo. Try again with a longer Yahoo period or a full broker account.")


if __name__ == "__main__":
    main()
