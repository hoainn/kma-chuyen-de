"""
Ensemble scorer: alpha * VAE_score + (1-alpha) * iForest_score

Uses a p1..p99 RobustScaler per component to prevent one component from
dominating due to scale differences.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import RobustScaler


class EnsembleScorer:
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self._vae_scaler = RobustScaler(quantile_range=(1.0, 99.0))
        self._if_scaler = RobustScaler(quantile_range=(1.0, 99.0))
        self._fitted = False

    def score(self, vae_scores: np.ndarray, if_scores: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() before score()"
        v = self._vae_scaler.transform(vae_scores.reshape(-1, 1)).ravel()
        f = self._if_scaler.transform(if_scores.reshape(-1, 1)).ravel()
        return self.alpha * v + (1 - self.alpha) * f

    @classmethod
    def load(cls, out_dir: str | Path) -> "EnsembleScorer":
        import joblib
        out_dir = Path(out_dir)
        params = json.loads((out_dir / "ensemble_params.json").read_text())
        obj = cls(alpha=params["alpha"])
        obj._vae_scaler = joblib.load(out_dir / "ensemble_vae_scaler.joblib")
        obj._if_scaler = joblib.load(out_dir / "ensemble_if_scaler.joblib")
        obj._fitted = params["fitted"]
        return obj
