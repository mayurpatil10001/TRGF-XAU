"""
scripts/download_mt5_data_extended.py
Download as much XAUUSD 1M history as the Exness terminal will serve.
Forces chart load by opening a chart window, then pulls data in monthly chunks.
"""
from __future__ import annotations

import os, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

import MetaTrader5 as mt5

SYMBOL_CANDIDATES = ["XAUUSDm", "XAUUSD", "XAUUSD.a"]
OUT_PATH = Path("data/raw/xauusd_1m.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_login() -> str | None:
    path = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
    mt5.initialize(path=path) if Path(path).exists() else mt5.initialize()
    mt5.login(int(os.getenv("MT5_LOGIN")), password=os.getenv("MT5_PASSWORD"), server=os.getenv("MT5_SERVER"))
    acc = mt5.account_info()
    print(f"Account: #{acc.login} DEMO balance={acc.balance:.2f}")
    for sym in SYMBOL_CANDIDATES:
        if mt5.symbol_info(sym):
            mt5.symbol_select(sym, True)
            print(f"Symbol: {sym}")
            return sym
    return None


def fetch_from(symbol: str, from_dt: datetime, to_dt: datetime) -> pd.DataFrame | None:
    """Try multiple MT5 methods to pull bars in window."""
    # Method A: copy_rates_range
    r = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, from_dt, to_dt)
    if r is not None and len(r) > 0:
        return pd.DataFrame(r)
    # Method B: copy_rates_from with large count
    bars_in_window = int((to_dt - from_dt).total_seconds() / 60) + 60
    r = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, to_dt, min(bars_in_window, 99000))
    if r is not None and len(r) > 0:
        df = pd.DataFrame(r)
        df["time_dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[df["time_dt"] >= from_dt].drop(columns="time_dt")
        if len(df) > 0:
            return df
    return None


def main():
    symbol = init_login()
    if not symbol:
        print("ERROR: symbol not resolved")
        sys.exit(1)

    # Get spread point info
    info = mt5.symbol_info(symbol)
    point = info.point if info else 0.001

    all_chunks = []

    # Try to pull in 6-month chunks going back as far as possible
    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    # Try years back, stop when we get nothing for 3 consecutive windows
    # Exness trial usually has ~2–5 years of 1M data
    consecutive_empty = 0
    chunk_months = 6

    current_end = end_dt
    attempt = 0
    while consecutive_empty < 3 and attempt < 40:
        attempt += 1
        current_start = current_end - timedelta(days=30 * chunk_months)

        df = fetch_from(symbol, current_start, current_end)

        if df is not None and len(df) > 0:
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            all_chunks.append(df)
            n = len(df)
            t0 = df["time"].iloc[0].strftime("%Y-%m-%d")
            t1 = df["time"].iloc[-1].strftime("%Y-%m-%d")
            print(f"  [{t0} -> {t1}] {n:,} bars")
            consecutive_empty = 0
        else:
            print(f"  [{current_start.date()} -> {current_end.date()}] empty (err={mt5.last_error()})")
            consecutive_empty += 1

        current_end = current_start
        time.sleep(0.1)

    mt5.shutdown()

    if not all_chunks:
        print("ERROR: No data downloaded at all.")
        sys.exit(1)

    # Merge, deduplicate, sort
    df = pd.concat(all_chunks, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    # Convert spread
    df["spread"] = df["spread"] * point
    if df["spread"].mean() > 5 or df["spread"].mean() < 0.005:
        df["spread"] = 0.25  # fallback

    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0

    out = df[["time","open","high","low","close","tick_volume","spread"]].copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(OUT_PATH, index=False)

    mb = OUT_PATH.stat().st_size / 1e6
    print(f"\nSAVED: {OUT_PATH}")
    print(f"  Rows:   {len(out):,}")
    print(f"  Size:   {mb:.1f} MB")
    print(f"  Range:  {out['time'].iloc[0]}  ->  {out['time'].iloc[-1]}")
    print(f"  Spread: {out['spread'].mean():.4f} mean")


if __name__ == "__main__":
    main()
