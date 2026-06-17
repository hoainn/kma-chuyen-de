"""
IF + VAE ensemble scorer — pure inference, no I/O.
"""

from __future__ import annotations

import numpy as np


class Scorer:
    def __init__(
        self,
        iforest,
        encoder,
        decoder,
        threshold: float,
        ensemble=None,
        legacy_norm: dict | None = None,
    ):
        self._iforest    = iforest
        self._encoder    = encoder
        self._decoder    = decoder
        self._threshold  = float(threshold)
        self._ensemble   = ensemble
        self._legacy_norm = legacy_norm

    @property
    def threshold(self) -> float:
        return self._threshold

    def set_threshold(self, t: float) -> None:
        """Mutate the active cut-off — used by the EMA loop in detect.py."""
        self._threshold = float(t)

    def score(self, feat: np.ndarray) -> dict:
        """Score a (1, input_dim) feature array. Returns dict with if_raw, vae_raw, ensemble, is_attack."""
        import tensorflow as tf

        # IF score (higher = more anomalous)
        if_raw = float(-self._iforest.decision_function(feat)[0])

        # VAE reconstruction error
        x_tf = tf.constant(feat, dtype=tf.float32)
        z_mean, z_log_var, z = self._encoder(x_tf, training=False)
        x_hat = self._decoder(z, training=False)
        vae_raw = float(np.mean((feat - x_hat.numpy()) ** 2))

        # Ensemble score
        if self._ensemble is not None:
            ens_score = float(
                self._ensemble.score(np.array([vae_raw]), np.array([if_raw]))[0]
            )
        else:
            n = self._legacy_norm or {}
            alpha   = n.get("alpha", 0.7)
            if_lo   = n.get("if_lo",  -0.3)
            if_hi   = n.get("if_hi",   0.3)
            vae_lo  = n.get("vae_lo",  0.0)
            vae_hi  = n.get("vae_hi",  1e6)
            if_norm  = float(np.clip((if_raw  - if_lo)  / (if_hi  - if_lo  + 1e-9), 0, 1))
            vae_norm = float(np.clip((vae_raw - vae_lo) / (vae_hi - vae_lo + 1e-9), 0, 1))
            ens_score = alpha * vae_norm + (1 - alpha) * if_norm

        return {
            "if_raw":    round(if_raw, 4),
            "vae_raw":   round(vae_raw, 4),
            "ensemble":  round(ens_score, 4),
            "is_attack": bool(ens_score >= self._threshold),
        }
