"""Diagnose why no trades are being generated."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from src.common.config import load_config, reset_config
from src.strategy.decision import make_decision

reset_config()
cfg = load_config('config/config.yaml')

preds = np.load('artifacts/oos_predictions.npy', allow_pickle=True).item()
df_all = pd.read_parquet('data/processed/labels.parquet')
df_all['time'] = pd.to_datetime(df_all['time'], utc=True)
oos_times = pd.to_datetime(preds['times'], utc=True)
df_oos = df_all[df_all['time'].isin(oos_times)].reset_index(drop=True)

gf = preds.get('gated_fusion')
mlp = preds.get('mlp')
lgbm = preds.get('lgbm')
prob_arrays = [(gf, 2.0), (mlp, 1.0), (lgbm, 1.0)]
prob_arrays = [(a, w) for a, w in prob_arrays if a is not None]
total_w = sum(w for _, w in prob_arrays)
probs_fused = sum(a * w for a, w in prob_arrays) / total_w

print(f"Probs shape: {probs_fused.shape}")
print(f"Prob class distribution (mean): {probs_fused.mean(axis=0)}")
print(f"Max prob per class: {probs_fused.max(axis=0)}")
print(f"Threshold in config: {cfg.decision.prob_threshold_grid}")

# Check first 20 bars
reasons = {}
n_buy = n_sell = n_no = 0
for i in range(min(5000, len(df_oos))):
    row = df_oos.iloc[i]
    prob = probs_fused[i]
    atr = row.get('atr', row.get('atr_14', 0.5))
    if atr == 0 or pd.isna(atr): atr = 0.5
    spread = row.get('spread', cfg.costs.spread_price_units)
    if pd.isna(spread) or spread <= 0: spread = cfg.costs.spread_price_units

    d = make_decision(probs=prob, bar_time=row['time'], spread=spread, atr=atr, cfg=cfg)
    if d.action == 'BUY': n_buy += 1
    elif d.action == 'SELL': n_sell += 1
    else: reasons[d.reason] = reasons.get(d.reason, 0) + 1

print(f"\nFirst 5000 bars: BUY={n_buy} SELL={n_sell} NO_TRADE={sum(reasons.values())}")
print("NO_TRADE reasons:", dict(sorted(reasons.items(), key=lambda x: -x[1])[:10]))
print(f"\nSessions allowed: {cfg.decision.sessions_allowed}")
print(f"Blackout: {cfg.decision.blackout_utc}")

# What sessions do OOS bars fall into?
from src.common.timeutils import get_session
session_map = {k: (v[0], v[1]) for k, v in cfg.sessions.model_dump().items()}
sessions_in_oos = df_oos['time'].iloc[:1000].apply(lambda t: get_session(t, session_map))
print(f"\nSession distribution (first 1000 bars): {sessions_in_oos.value_counts().to_dict()}")
