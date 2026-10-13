"""
Robust MT5 data downloader — finds symbol, downloads in chunks, appends to existing data.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
from dotenv import load_dotenv
load_dotenv()

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone, timedelta

def find_symbol():
    """Find the correct gold symbol name."""
    candidates = ["XAUUSDm", "XAUUSD", "XAUUSD.a", "XAUUSD.", "XAUUSDm."]
    for s in candidates:
        info = mt5.symbol_info(s)
        if info is not None:
            mt5.symbol_select(s, True)
            return s
    # Try searching
    symbols = mt5.symbols_get("*XAU*") or []
    for s in symbols:
        if "USD" in s.name:
            mt5.symbol_select(s.name, True)
            return s.name
    return None

def main():
    login = int(os.getenv("MT5_LOGIN", "0"))
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "")
    mt5_path = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")

    # Init with path, fallback without
    if not mt5.initialize(path=mt5_path):
        if not mt5.initialize():
            print(f"MT5 initialize failed: {mt5.last_error()}")
            return
    if not mt5.login(login, password=password, server=server):
        print(f"Login failed: {mt5.last_error()}")
        mt5.shutdown(); return

    symbol = find_symbol()
    if not symbol:
        print("Could not find XAU symbol!")
        mt5.shutdown(); return
    print(f"Using symbol: {symbol}")

    # Load existing data to find the last timestamp
    out_path = Path("data/raw/xauusd_1m.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_df = None
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        existing_df["time"] = pd.to_datetime(existing_df["time"])
        last_ts = existing_df["time"].max()
        print(f"Existing data: {len(existing_df):,} bars, last={last_ts}")
    else:
        last_ts = datetime(2026, 6, 1)

    # Download in chunks of 10,000 bars working backwards from now
    all_chunks = []
    CHUNK = 10000
    total = 0
    # Get up to 120k bars from pos 0 (most recent)
    for pos in range(0, 120001, CHUNK):
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, pos, CHUNK)
        if rates is None or len(rates) == 0:
            print(f"  Stopped at pos={pos}: {mt5.last_error()}")
            break
        chunk_df = pd.DataFrame(rates)
        chunk_df["time"] = pd.to_datetime(chunk_df["time"], unit="s", utc=True).dt.tz_localize(None)
        all_chunks.append(chunk_df)
        total += len(chunk_df)
        oldest = chunk_df["time"].min()
        newest = chunk_df["time"].max()
        print(f"  Chunk pos={pos}: {len(chunk_df):,} bars | {oldest} → {newest}")
        if oldest < pd.Timestamp("2026-06-01"):
            break
        time.sleep(0.1)

    if not all_chunks:
        print("No data downloaded!")
        mt5.shutdown(); return

    new_df = pd.concat(all_chunks).drop_duplicates("time").sort_values("time").reset_index(drop=True)

    # Merge with existing
    if existing_df is not None:
        combined = pd.concat([existing_df, new_df]).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    else:
        combined = new_df

    # Keep only needed columns
    keep = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    combined = combined[[c for c in keep if c in combined.columns]]
    combined.to_csv(out_path, index=False)

    print(f"\n=== DONE ===")
    print(f"Total bars: {len(combined):,}")
    print(f"Range: {combined['time'].min()} → {combined['time'].max()}")
    mt5.shutdown()

if __name__ == "__main__":
    main()
