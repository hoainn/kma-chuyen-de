"""
Artifact loader — auto-detects Mode A (fe_report.json) or Mode B (feature_config.json).
"""

from __future__ import annotations

import json
import logging
import os

import joblib
import numpy as np

from src.vae import build_vae
from src.ensemble import EnsembleScorer


def _find(d: str, *names: str) -> str | None:
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    return None


def load(artifacts_dir: str, log: logging.Logger | None = None) -> dict:
    """Load all model artifacts from the given directory.

    Returns a dict with keys:
        top_ids, disc_ids, top_bigrams, ver_cols, input_dim,
        feature_scaler, iforest, encoder, decoder,
        ensemble, legacy_norm, threshold
    """
    if log is None:
        log = logging.getLogger(__name__)

    d = artifacts_dir

    # ── Feature vocabulary ────────────────────────────────────────────────────
    fe_report_path = _find(d, "fe_report.json")
    fc_path = _find(d, "feature_config.json")

    if fe_report_path:
        log.info("Feature mode: legacy DongTing (fe_report.json)")
        with open(fe_report_path) as f:
            fe = json.load(f)
        top_ids  = list(fe["top_ids"])
        disc_ids = list(fe["disc_ids"])
        # support both old key (top_bigrams) and new key (top_ngrams)
        raw_ng   = fe.get("top_ngrams") or fe.get("top_bigrams", [])
        top_ngrams = [tuple(g) for g in raw_ng]
        ngram_n  = int(fe.get("ngram_n", 2))
        ver_cols = fe.get("ver_cols", [])
        input_dim = len(top_ids) + len(disc_ids) + 8 + len(top_ngrams) + len(ver_cols)
    elif fc_path:
        raw_fc = json.loads(open(fc_path).read())
        if "top_syscalls" in raw_fc and raw_fc["top_syscalls"]:
            log.info("Feature mode: FeatureConfig (feature_config.json)")
            top_ids  = raw_fc["top_syscalls"]
            disc_ids = raw_fc.get("disc_syscalls", [])
            raw_ng   = raw_fc.get("top_ngrams") or raw_fc.get("top_bigrams", [])
            top_ngrams = [tuple(g) for g in raw_ng]
            ngram_n  = int(raw_fc.get("ngram_n", 2))
            ver_cols = raw_fc.get("kernel_versions", [])
            input_dim = len(top_ids) + len(disc_ids) + 8 + len(top_ngrams) + len(ver_cols)
        else:
            raise RuntimeError(
                f"feature_config.json at {fc_path} has no top_syscalls. "
                "Use an artifacts dir with fe_report.json (DongTing) or a full FeatureConfig."
            )
    else:
        raise FileNotFoundError(
            f"No feature vocabulary found in {d}. "
            "Expected fe_report.json (DongTing) or feature_config.json (ADFA-LD)."
        )

    log.info(
        f"  vocab: freq={len(top_ids)} disc={len(disc_ids)} "
        f"{ngram_n}-grams={len(top_ngrams)} ver_bins={len(ver_cols)} → input_dim={input_dim}"
    )

    # ── Feature scaler ────────────────────────────────────────────────────────
    scaler_path = _find(d, "feature_scaler.joblib", "scaler_fe.pkl")
    feature_scaler = None
    if scaler_path:
        feature_scaler = joblib.load(scaler_path)
        log.info(f"Feature scaler loaded: {os.path.basename(scaler_path)}")
    else:
        log.warning("No feature scaler found — features will not be standardized.")

    # ── Isolation Forest ──────────────────────────────────────────────────────
    if_path = _find(d, "iforest.joblib", "if_model.joblib")
    if if_path is None:
        raise FileNotFoundError(f"Isolation Forest not found in {d}")
    iforest = joblib.load(if_path)
    log.info(f"IF loaded: {os.path.basename(if_path)}")

    # ── VAE ───────────────────────────────────────────────────────────────────
    enc_path = _find(d, "vae_encoder.weights.h5")
    dec_path = _find(d, "vae_decoder.weights.h5")
    if enc_path is None or dec_path is None:
        raise FileNotFoundError(f"VAE weights not found in {d}")

    mp_path = _find(d, "model_params.json")
    latent_dim = 8    # paper default: 32→8→32 architecture
    hidden_dim = 32   # paper default
    # results.json (written by train_dongting.ipynb) takes priority
    res_path = _find(d, "results.json", "model_report_fe.json")
    if res_path:
        _res = json.loads(open(res_path).read())
        latent_dim = int(_res.get("latent_dim", latent_dim))
        hidden_dim = int(_res.get("hidden_dim", hidden_dim))
    elif mp_path:
        mp = json.loads(open(mp_path).read())
        latent_dim = mp.get("latent_dim", latent_dim)
        hidden_dim = mp.get("hidden_dim", hidden_dim)

    import tensorflow as tf

    encoder, decoder, _ = build_vae(
        input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=0.2
    )
    _dummy = tf.zeros((1, input_dim))
    encoder(_dummy, training=False)
    decoder(tf.zeros((1, latent_dim)), training=False)
    encoder.load_weights(enc_path)
    decoder.load_weights(dec_path)
    log.info(f"VAE loaded: latent_dim={latent_dim}")

    # ── Ensemble scorer ───────────────────────────────────────────────────────
    ens_params_path = _find(d, "ensemble_params.json")
    ensemble = None
    if ens_params_path and _find(d, "ensemble_if_scaler.joblib"):
        ensemble = EnsembleScorer.load(d)
        log.info(f"EnsembleScorer loaded (alpha={ensemble.alpha})")
    else:
        log.warning("EnsembleScorer not found — falling back to raw min-max normalization.")

    # ── Decision threshold ────────────────────────────────────────────────────
    results_path = _find(d, "results.json", "model_report_fe.json")
    threshold: float
    if results_path:
        res = json.loads(open(results_path).read())
        if "models" in res and "ensemble" in res["models"]:
            threshold = float(res["models"]["ensemble"]["threshold"])
        else:
            threshold = float(res.get("threshold", 0.5))
        log.info(f"Threshold loaded from results.json: {threshold:.4f}")
    elif mp_path:
        mp_data = json.loads(open(mp_path).read())
        threshold = float(mp_data.get("ens_threshold", 0.5))
        log.info(f"Threshold loaded from model_params.json: {threshold:.4f}")
    else:
        threshold = 0.5
        log.warning(f"No threshold file found — using default {threshold}")

    # ── Legacy min-max normalizer (fallback when no EnsembleScorer) ───────────
    legacy_norm = None
    if ensemble is None and mp_path:
        mp_data = json.loads(open(mp_path).read())
        legacy_norm = {
            "if_lo":  mp_data.get("if_lo",  -0.3),
            "if_hi":  mp_data.get("if_hi",   0.3),
            "vae_lo": mp_data.get("vae_lo",  0.0),
            "vae_hi": mp_data.get("vae_hi",  1e6),
            "alpha":  mp_data.get("alpha",   0.7),
        }
        log.info(f"Legacy normalizer: alpha={legacy_norm['alpha']}")

    return {
        "top_ids":       top_ids,
        "disc_ids":      disc_ids,
        "top_ngrams":    top_ngrams,
        "ngram_n":       ngram_n,
        "ver_cols":      ver_cols,
        "input_dim":     input_dim,
        "feature_scaler": feature_scaler,
        "iforest":       iforest,
        "encoder":       encoder,
        "decoder":       decoder,
        "ensemble":      ensemble,
        "legacy_norm":   legacy_norm,
        "threshold":     threshold,
    }
