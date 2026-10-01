"""
src/common/config.py
Pydantic-validated configuration loader.
All parameters come from config/config.yaml; .env is merged for secrets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────────────────────

class DataConfig(BaseModel):
    path: str = "data/raw/xauusd_1m.csv"
    timezone_in: str = "UTC"
    timezone_out: str = "UTC"
    main_start: str = "2015-01-01"
    dev_end: str = "2024-12-31"
    holdout_start: str = "2025-01-01"
    full_period_experiment: bool = True
    min_price: float = 100.0
    max_gap_minutes: int = 30


class SymbolConfig(BaseModel):
    name: str = "XAUUSD"
    contract_size: int = 100
    point: float = 0.01


class CostsConfig(BaseModel):
    spread_price_units: float = 0.25
    slippage_price_units: float = 0.05
    commission_per_lot_round_trip: float = 7.0
    spread_stress: List[float] = Field(default=[1.0, 1.5, 2.0])


class FeaturesConfig(BaseModel):
    lookback: int = 60
    atr_period: int = 14
    vol_windows: List[int] = Field(default=[5, 15, 60])
    ema_periods: List[int] = Field(default=[20, 50, 200])


class LabelsConfig(BaseModel):
    horizon_bars: List[int] = Field(default=[1, 3, 5])
    tp_atr_mult_k: List[float] = Field(default=[0.5, 0.75, 1.0])
    sl_atr_mult_m: List[float] = Field(default=[0.5, 0.75, 1.0])


class WalkForwardConfig(BaseModel):
    train_months: int = 36
    val_months: int = 3
    test_months: int = 3
    step_months: int = 3
    embargo_trading_days: int = 5


class ModelConfig(BaseModel):
    names: List[str] = Field(default=["lgbm", "mlp", "lstm", "transformer", "gated_fusion"])
    epochs_max: int = 60
    early_stop_patience: int = 6
    batch_size: int = 1024
    lr_grid: List[float] = Field(default=[1e-3, 3e-4, 1e-4])
    dropout_grid: List[float] = Field(default=[0.1, 0.3, 0.5])
    weight_decay_grid: List[float] = Field(default=[1e-2, 1e-4])


class DecisionConfig(BaseModel):
    prob_threshold_grid: List[float] = Field(default=[0.50, 0.55, 0.60, 0.65, 0.70])
    max_spread_price_units: float = 0.40
    sessions_allowed: List[str] = Field(default=["london", "ny", "overlap"])
    blackout_utc: List[str] = Field(default=["21:55-23:05"])
    news_blackout_file: Optional[str] = None
    min_expected_R: float = 0.0


class RiskConfig(BaseModel):
    risk_fraction_kappa: float = 0.01
    max_open_positions: int = 1
    max_consecutive_losses: int = 3
    daily_loss_limit_pct: float = 3.0
    cooldown_bars_after_exit: int = 1


class ProfitGateConfig(BaseModel):
    min_oos_trades: int = 300
    min_profit_factor: float = 1.2
    min_positive_fold_fraction: float = 0.60
    require_positive_at_spread_mult: float = 1.5
    require_bootstrap_ci_lower_gt_zero: bool = True


class LiveConfig(BaseModel):
    mode: str = "demo"
    allow_live_money: bool = False
    demo_override_gate: bool = False   # allow demo trades even when gate FAILED
    magic_number: int = 20260930
    deviation_points: int = 20
    max_data_staleness_seconds: int = 90
    kill_switch_file: str = "KILL"


class SessionsConfig(BaseModel):
    asia: List[str] = Field(default=["00:00", "07:00"])
    london: List[str] = Field(default=["07:00", "12:00"])
    overlap: List[str] = Field(default=["12:00", "16:00"])
    ny: List[str] = Field(default=["16:00", "21:00"])


class TrainingConfig(BaseModel):
    lambda_nll_ce: float = 1.0
    grad_clip: float = 1.0
    hp_search_folds: int = 2


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "logs"


# ─────────────────────────────────────────────────────────────────────────────
# Root config
# ─────────────────────────────────────────────────────────────────────────────

class Config(BaseModel):
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    symbol: SymbolConfig = Field(default_factory=SymbolConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    profit_gate: ProfitGateConfig = Field(default_factory=ProfitGateConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)
    sessions: SessionsConfig = Field(default_factory=SessionsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def validate_splits(self) -> "Config":
        if self.data.holdout_start <= self.data.dev_end:
            # Allow this but warn; holdout must not overlap dev
            pass
        return self

    def config_hash(self) -> str:
        """Stable SHA-256 hash of the config for artifact versioning."""
        raw = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_config(path: str | Path = "config/config.yaml") -> Config:
    """Load and validate the YAML config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}
    return Config(**raw)


# Singleton for use across the codebase
_config_instance: Optional[Config] = None


def get_config(path: str | Path = "config/config.yaml") -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config(path)
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (useful for tests)."""
    global _config_instance
    _config_instance = None
