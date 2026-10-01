import sys
sys.path.insert(0, '.')
from src.common.config import load_config, reset_config
from src.training.windows import generate_folds
import pandas as pd

reset_config()
cfg = load_config('config/config.yaml')

# Load actual data times
df = pd.read_csv('data/raw/xauusd_1m.csv', usecols=['time'])
df['time'] = pd.to_datetime(df['time'], utc=True)

dev_start = pd.Timestamp(cfg.data.main_start, tz='UTC')
dev_end   = pd.Timestamp(cfg.data.dev_end, tz='UTC')
times = df['time']
times_in_dev = times[(times >= dev_start) & (times < dev_end)]

print(f"Config dev_start: {dev_start}  dev_end: {dev_end}")
print(f"Data range:       {times.iloc[0]}  to  {times.iloc[-1]}")
print(f"Bars in dev:      {len(times_in_dev):,}")
print()

# Simulate fold boundary check
import pandas.tseries.offsets as off
train_m, val_m, test_m = cfg.walk_forward.train_months, cfg.walk_forward.val_months, cfg.walk_forward.test_months
fold_start = dev_start
for i in range(10):
    te = fold_start + pd.DateOffset(months=train_m+val_m+test_m)
    fits = te <= dev_end
    print(f"  fold {i}: start={fold_start.date()} test_end={te.date()} fits_in_dev={fits}")
    if not fits:
        break
    fold_start += pd.DateOffset(months=cfg.walk_forward.step_months)

print()
folds = generate_folds(times_in_dev.reset_index(drop=True), cfg)
print(f"Total folds: {len(folds)}")
for f in folds:
    print(f"  fold {f.fold_id}: train={len(f.train_idx)} val={len(f.val_idx)} test={len(f.test_idx)}")
