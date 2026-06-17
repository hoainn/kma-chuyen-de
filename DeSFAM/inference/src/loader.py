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


def _build_patterns_by_first(patterns: list) -> dict[int, list]:
    """Index benign patterns by first syscall ID — mirrors
    train.py::build_patterns_by_first (same sort) for byte-identical matching."""
    from collections import defaultdict
    by_first: dict[int, list] = defaultdict(list)
    for p in patterns:
        if p:
            by_first[int(p[0])].append(tuple(int(x) for x in p))
    return {k: sorted(v) for k, v in by_first.items()}


def load(artifacts_dir: str, log: logging.Logger | None = None) -> dict:
    """Load all model artifacts from the given directory.

    Returns a dict with keys:
        cat_cols, id_to_cat, patterns_by_first, ps_dims,
        has_temporal, temporal_dims, input_dim,
        feature_scaler, iforest, encoder, decoder,
        ensemble, legacy_norm, threshold
    """
    if log is None:
        log = logging.getLogger(__name__)

    d = artifacts_dir

    # ── Paper-only feature config (fe_report.json) ─────────────────────────────
    fe_report_path = _find(d, "fe_report.json")
    if not fe_report_path:
        raise FileNotFoundError(f"No fe_report.json found in {d}.")
    with open(fe_report_path) as f:
        fe = json.load(f)
    # `feature_mode` is no longer required (paper-plus is the only supported
    # configuration). Older artefacts still carry the field — accept either
    # absent or any string value here for backward compatibility.
    mode = fe.get("feature_mode", "paper_plus")

    cat_cols = fe.get("cat_cols", [])
    raw_id2cat = fe.get("syscall_id_to_cat", {}) or {}
    id_to_cat: dict[int, str] = {int(k): v for k, v in raw_id2cat.items()}
    has_temporal  = bool(fe.get("has_temporal", False))
    temporal_dims = int(fe.get("temporal_dims", 0))
    ps_dims       = int(fe.get("prefixspan_dims", 0))
    # Sensor alphabet — the syscall IDs training was restricted to. We re-filter
    # live windows to it so a TracingPolicy that drifts wider than training can't
    # skew cat_K (which maps ANY id via id_to_cat). Empty/absent → no filter.
    sensor_alphabet = {int(x) for x in fe.get("sensor_alphabet", [])} or None
    # paper_plus engineered vocab (empty under paper_only)
    top_ids      = [int(x) for x in fe.get("top_ids", [])]
    disc_ids     = [int(x) for x in fe.get("disc_ids", [])]
    top_bigrams  = [tuple(g) for g in fe.get("top_bigrams", [])]
    enable_stats = bool(fe.get("enable_stats", False))

    # Benign pattern DB (PrefixSpan)
    patterns_by_first: dict[int, list] = {}
    bp_name = fe.get("benign_patterns_file", "benign_patterns.json")
    bp_path = _find(d, bp_name, "benign_patterns.json")
    if ps_dims and bp_path:
        with open(bp_path) as f:
            patterns_by_first = _build_patterns_by_first(json.load(f).get("patterns", []))

    n_stats = 8 if enable_stats else 0
    input_dim = int(fe.get("total_dims",
                           len(top_ids) + len(disc_ids) + n_stats + len(top_bigrams)
                           + len(cat_cols) + temporal_dims + ps_dims))

    log.info(
        f"  {mode}: freq={len(top_ids)} disc={len(disc_ids)} stats={n_stats} "
        f"bigrams={len(top_bigrams)} cat={len(cat_cols)} "
        f"temporal={temporal_dims} prefixspan={ps_dims}  → input_dim={input_dim}"
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
    # Silence joblib.Parallel logging: models trained with verbose=1 re-emit a
    # progress block on every decision_function call, flooding the detector log.
    try:
        iforest.verbose = 0
    except Exception:
        pass
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
    # Per-component T_0 (paper §IV.B.3 initial threshold) — for the Grafana
    # "Component Thresholds" panel and the eventual EMA loop in detect.py.
    component_thresholds: dict[str, float] = {}
    experiment: str = ""
    if results_path:
        res = json.loads(open(results_path).read())
        experiment = str(res.get("experiment", ""))
        if "models" in res and "ensemble" in res["models"]:
            threshold = float(res["models"]["ensemble"]["threshold"])
            # results.json keys are 'isolation_forest' / 'vae' / 'ensemble'; we
            # normalise the IF label to 'iforest' for the dashboard.
            _label_map = {"vae": "vae", "isolation_forest": "iforest", "ensemble": "ensemble"}
            for k, dst in _label_map.items():
                if k in res["models"]:
                    component_thresholds[dst] = float(res["models"][k]["threshold"])
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
        "feature_mode":      mode,
        "sensor_alphabet":   sensor_alphabet,
        "cat_cols":          cat_cols,
        "id_to_cat":         id_to_cat,
        "patterns_by_first": patterns_by_first,
        "ps_dims":           ps_dims,
        "has_temporal":      has_temporal,
        "temporal_dims":     temporal_dims,
        "top_ids":           top_ids,
        "disc_ids":          disc_ids,
        "top_bigrams":       top_bigrams,
        "enable_stats":      enable_stats,
        "input_dim":         input_dim,
        "feature_scaler":    feature_scaler,
        "iforest":           iforest,
        "encoder":           encoder,
        "decoder":           decoder,
        "ensemble":          ensemble,
        "legacy_norm":       legacy_norm,
        "threshold":         threshold,
        "component_thresholds": component_thresholds,
        "experiment":        experiment,
    }
