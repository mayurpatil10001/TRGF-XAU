import sys
sys.path.insert(0, '.')
from src.common.config import load_config, reset_config
from src.data.loader import load_and_clean
reset_config()
cfg = load_config('config/config.yaml')
df = load_and_clean(cfg)
print(f'Bars loaded: {len(df):,}')
print(f'Date range: {str(df["time"].iloc[0])} to {str(df["time"].iloc[-1])}')
print(f'Spread mean: {df["spread"].mean():.4f}')
print(f'Columns: {list(df.columns)}')
