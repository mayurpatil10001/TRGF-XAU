"""
src/live/inference.py
Live inference: load frozen artifacts, build features from rolling buffer, run model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.common.config import Config
from src.common.logging import get_logger
from src.features.feature_pipeline import build_features, apply_scaler
from src.models.lgbm_baseline import load_lgbm

logger = get_logger("live.inference")


class LiveInferenceEngine:
    """
    Loads the latest fold artifacts and performs inference on a rolling bar buffer.
    Uses the FROZEN scaler and model from the walk-forward pipeline.
    """

    def __init__(self, cfg: Config, fold_id: int = -1, model_name: str = "gated_fusion"):
        self.cfg = cfg
        self.model_name = model_name
        self._model = None
        self._scaler = None
        self._feature_names: List[str] = []
        self._model_version = "unknown"
        self._load_artifacts(fold_id)

    def _load_artifacts(self, fold_id: int) -> None:
        artifacts_dir = Path("artifacts")
        feat_names_path = artifacts_dir / "feature_names.json"
        if feat_names_path.exists():
            with open(feat_names_path) as f:
                self._feature_names = json.load(f)

        # Find latest fold
        if fold_id < 0:
            fold_dirs = sorted(artifacts_dir.glob("fold_*/"), reverse=True)
            if not fold_dirs:
                logger.warning("No fold artifacts found; model will return random probs.")
                return
            fold_dir = fold_dirs[0]
        else:
            fold_dir = artifacts_dir / f"fold_{fold_id:02d}"

        scaler_path = fold_dir / "scaler.joblib"
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")

        # Try gated_fusion first, fallback to lgbm
        model_path = fold_dir / f"{self.model_name}.pt"
        if model_path.exists():
            import torch
            from src.models.gated_fusion import GatedFusion
            F = len(self._feature_names)
            L = self.cfg.features.lookback
            model = GatedFusion(input_dim=F, lookback=L)
            model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
            model.eval()
            self._model = ("torch", model)
            self._model_version = hashlib.md5(model_path.read_bytes()).hexdigest()[:8]
            logger.info(f"Loaded {self.model_name} from {model_path}")
        else:
            lgbm_path = fold_dir / "lgbm.txt"
            if lgbm_path.exists():
                lgbm = load_lgbm(lgbm_path)
                self._model = ("lgbm", lgbm)
                self._model_version = hashlib.md5(lgbm_path.read_bytes()).hexdigest()[:8]

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, bar_buffer: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Given a rolling buffer of bars, build features and run inference.

        Returns:
            probs: [3] float (p_none, p_long, p_short)
            features_hash: str
        """
        if self._model is None:
            return np.array([0.99, 0.005, 0.005]), "no_model"

        # Build features
        feat_df, _ = build_features(bar_buffer, self.cfg)
        if len(feat_df) == 0:
            return np.array([0.99, 0.005, 0.005]), "no_features"

        # Align to known feature names
        missing = [c for c in self._feature_names if c not in feat_df.columns]
        for c in missing:
            feat_df[c] = 0.0
        feat_df = feat_df[self._feature_names]

        # Scale
        if self._scaler is not None:
            feat_arr = np.clip(self._scaler.transform(feat_df.values[-1:]), -10, 10)
        else:
            feat_arr = feat_df.values[-1:]

        feat_hash = hashlib.md5(feat_arr.tobytes()).hexdigest()[:8]

        kind, model = self._model
        if kind == "lgbm":
            probs = model.predict(feat_arr)[0]
        else:
            import torch
            L = self.cfg.features.lookback
            F = len(self._feature_names)
            # Build sequence from last L bars
            scaled_all = np.clip(self._scaler.transform(feat_df.values) if self._scaler else feat_df.values, -10, 10)
            seq = np.zeros((1, L, F), dtype=np.float32)
            seg = scaled_all[-L:]
            seq[0, L - len(seg):] = seg
            with torch.no_grad():
                out = model(torch.tensor(seq))
                if len(out) == 4:
                    logits = out[0]
                else:
                    logits = out[0]
                probs = torch.softmax(logits, dim=-1).numpy()[0]

        return probs, feat_hash
