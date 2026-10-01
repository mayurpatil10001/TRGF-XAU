"""
scripts/download_mt5_data.py  (v2 - uses copy_rates_from_pos fallback)
Download all available XAUUSD 1M bars from MT5 and save to data/raw/xauusd_1m.csv
"""
from __future__ import annotations

import os, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: pip install MetaTrader5")
    sys.exit(1)

SYMBOL_CANDIDATES = ["XAUUSDm", "XAUUSD", "XAUUSD.a", "XAUUSDpro", "XAUUSDeco"]
OUT_PATH = Path("data/raw/xauusd_1m.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def find_terminal() -> str | None:
    paths = [
        os.getenv("MT5_PATH", ""),
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
    ]
    for p in paths:
        if p and Path(p).exists():
            return p
    return None


def init_and_login() -> str | None:
    """Initialize MT5, login, resolve symbol. Returns resolved symbol or None."""
    path = find_terminal()
    ok = mt5.initialize(path=path) if (path and path.endswith(".exe")) else mt5.initialize()
    if not ok:
        print(f"Init failed: {mt5.last_error()}")
        return None

    login = int(os.getenv("MT5_LOGIN", "0"))
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "")
    if not mt5.login(login, password=password, server=server):
        print(f"Login failed: {mt5.last_error()}")
        return None

    acc = mt5.account_info()
    mode = "DEMO" if acc.trade_mode == 0 else "LIVE"
    print(f"✓ Logged in: #{acc.login} [{mode}] balance={acc.balance:.2f} {acc.currency}")

    for sym in SYMBOL_CANDIDATES:
        info = mt5.symbol_info(sym)
        if info is not None:
            mt5.symbol_select(sym, True)
            print(f"✓ Symbol: {sym}  digits={info.digits}  point={info.point}")
            return sym
    print("Could not resolve symbol")
    return None


def download_chunk_from_pos(symbol: str, pos: int, count: int) -> pd.DataFrame | None:
    """Download `count` bars starting at offset `pos` from current bar."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, pos, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def download_via_from_date(symbol: str) -> pd.DataFrame:
    """Try downloading using copy_rates_from (from specific date)."""
    CHUNK_DAYS = 90
    all_chunks = []
    end_dt = datetime.now(timezone.utc)
    start_dt = datetime(2015, 1, 1, tzinfo=timezone.utc)  # Exness usually has 10yr history

    current_end = end_dt
    print(f"Downloading via date-range chunks ({CHUNK_DAYS}-day chunks)...")

    while current_end > start_dt:
        current_start = current_end - timedelta(days=CHUNK_DAYS)
        if current_start < start_dt:
            current_start = start_dt

        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, current_start, current_end)

        if rates is None or len(rates) == 0:
            # Try copy_rates_from
            rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, current_end, 90 * 390)

        if rates is not None and len(rates) > 0:
            chunk = pd.DataFrame(rates)
            chunk["time"] = pd.to_datetime(chunk["time"], unit="s", utc=True)
            chunk = chunk[chunk["time"] >= current_start]
            all_chunks.append(chunk)
            print(f"  Got {len(chunk):,} bars  [{chunk['time'].iloc[0].date()} → {chunk['time'].iloc[-1].date()}]")
        else:
            print(f"  No data for {current_start.date()} → {current_end.date()}: {mt5.last_error()}")

        current_end = current_start
        time.sleep(0.05)

    if not all_chunks:
        return pd.DataFrame()
    df = pd.concat(all_chunks, ignore_index=True)
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df


def download_via_pos(symbol: str) -> pd.DataFrame:
    """Download using copy_rates_from_pos in large chunks."""
    CHUNK = 50_000
    all_chunks = []
    pos = 0
    print("Downloading via position-based method...")

    while True:
        df = download_chunk_from_pos(symbol, pos, CHUNK)
        if df is None or len(df) == 0:
            break
        all_chunks.append(df)
        print(f"  pos={pos:,}: got {len(df):,} bars  [{df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()}]")
        if len(df) < CHUNK:
            break
        pos += CHUNK
        time.sleep(0.05)

    if not all_chunks:
        return pd.DataFrame()
    df = pd.concat(all_chunks, ignore_index=True)
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df


def apply_spread(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Convert spread from points to price units."""
    info = mt5.symbol_info(symbol)
    if info and "spread" in df.columns:
        point = info.point
        df["spread"] = df["spread"] * point
        spread_mean = df["spread"].mean()
        print(f"  Spread: mean={spread_mean:.4f} USD  (point={point})")
        # If spread looks wrong (e.g. > 10), use fallback
        if spread_mean > 5.0 or spread_mean < 0.01:
            print(f"  ⚠ Spread looks wrong ({spread_mean:.4f}), using default 0.25")
            df["spread"] = 0.25
    else:
        df["spread"] = 0.25
    return df


def save(df: pd.DataFrame, symbol: str) -> None:
    df = apply_spread(df, symbol)
    if "tick_volume" not in df.columns:
        df["tick_volume"] = df.get("volume", pd.Series(0, index=df.index))

    out_cols = ["time", "open", "high", "low", "close", "tick_volume", "spread"]
    for c in out_cols:
        if c not in df.columns:
            df[c] = 0.0

    out = df[out_cols].copy()
    out["time"] = pd.to_datetime(out["time"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(OUT_PATH, index=False)
    mb = OUT_PATH.stat().st_size / 1e6
    print(f"\n✓ SAVED: {OUT_PATH}")
    print(f"  Rows : {len(out):,}")
    print(f"  Size : {mb:.1f} MB")
    print(f"  Range: {out['time'].iloc[0]}  →  {out['time'].iloc[-1]}")
    print(f"  Spread (mean): {out['spread'].mean():.4f}")


def main() -> None:
    symbol = init_and_login()
    if not symbol:
        sys.exit(1)

    # Method 1: position-based (most reliable)
    df = download_via_pos(symbol)

    # Method 2: date-range fallback if pos gave nothing
    if df.empty:
        print("\nPosition method returned no data; trying date-range method...")
        df = download_via_from_date(symbol)

    if df.empty:
        print("\nERROR: Could not download any bars. Check terminal settings:")
        print("  - Ensure MT5 terminal is open and charts are loaded for XAUUSDm M1")
        print("  - Tools > Options > Charts: tick 'Max bars in history'")
        mt5.shutdown()
        sys.exit(1)

    save(df, symbol)
    mt5.shutdown()
    print("\nMT5 shutdown. Ready to run pipeline.")


if __name__ == "__main__":
    main()
