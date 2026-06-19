#!/usr/bin/env python3
"""
vae_keras.py — REAL VAE arm using a deep-learning framework (TensorFlow/Keras).

Drop-in replacement for the AE component, identical pipeline & CAL/TEST windows
as build_detector.py (same split seeds) so the comparison is apples-to-apples.
Runs in the project's `training-train` Docker image (TF + scikit-learn).

    A(W) = 0.7 * z(VAE reconstruction error) + 0.3 * z(IsolationForest)

Output: output/scored_windows_vae.npz (same schema) + output/detector_report_vae.json
"""
from __future__ import annotations
import json, os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import build_detector as bd

OUT = bd.OUT
ALPHA = bd.ALPHA          # 0.7
T0_PCTILE = bd.T0_PCTILE  # 99.5
H_DIM, Z_DIM = 24, 6
EPOCHS, BATCH, LR, BETA_KL = 40, 256, 1e-3, 0.5
SEED = 0
FILTER = os.environ.get("FILTER", "1") == "1"   # FILTER=0 → no-filter ablation (top-20 freq)
TOP_K = 20


def _windows_full(seqs):
    out = []
    for s in seqs:
        if len(s) < bd.MIN_ALPHA:
            continue
        if len(s) < bd.WIN_LEN:
            out.append(s)
        else:
            for i in range(0, len(s) - bd.WIN_LEN + 1, bd.WIN_STRIDE):
                out.append(s[i:i + bd.WIN_LEN])
    return out


def _featurizer_top20(Wtr_full):
    from collections import Counter
    c = Counter()
    for w in Wtr_full:
        c.update(w.tolist())
    top = [sid for sid, _ in c.most_common(TOP_K)]; tidx = {s: i for i, s in enumerate(top)}
    def feat(wins):
        X = np.zeros((len(wins), len(top)), dtype=np.float64)
        for r, w in enumerate(wins):
            L = max(1, len(w))
            for sid, n in Counter(w.tolist()).items():
                if sid in tidx: X[r, tidx[sid]] = n / L
        return X
    return feat


def build_phase0():
    win = bd.windows if FILTER else _windows_full
    Wtr  = win(bd.load_part("DT-Normal", 0))
    Wval = win(bd.load_part("DT-Normal", 1))
    Watk = win(bd.load_part("DT-Abnormal", 0))
    if len(Wtr) > bd.MAX_TRAIN_WINDOWS:
        sel = np.random.default_rng(0).choice(len(Wtr), bd.MAX_TRAIN_WINDOWS, replace=False)
        Wtr = [Wtr[i] for i in sel]
    Wval_cal, Wval_test = bd.split_half(Wval, 1)
    Watk_cal, Watk_test = bd.split_half(Watk, 2)
    featurize = bd.featurize if FILTER else _featurizer_top20(Wtr)
    return Wtr, Wval_cal, Wval_test, Watk_cal, Watk_test, featurize


class VAE(keras.Model):
    """Gaussian VAE; call() returns deterministic reconstruction (z = mean) for scoring."""
    def __init__(self, d):
        super().__init__()
        self.enc_h   = layers.Dense(H_DIM, activation="tanh")
        self.z_mean  = layers.Dense(Z_DIM)
        self.z_lv    = layers.Dense(Z_DIM)
        self.dec_h   = layers.Dense(H_DIM, activation="tanh")
        self.dec_out = layers.Dense(d)

    def encode(self, x):
        h = self.enc_h(x)
        return self.z_mean(h), self.z_lv(h)

    def decode(self, z):
        return self.dec_out(self.dec_h(z))

    def call(self, x):
        zm, _ = self.encode(x)
        return self.decode(zm)

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            zm, zlv = self.encode(x)
            eps = tf.random.normal(tf.shape(zm))
            z = zm + tf.exp(0.5 * zlv) * eps
            out = self.decode(z)
            recon = tf.reduce_sum(tf.square(x - out), axis=-1)
            kl = -0.5 * tf.reduce_sum(1 + zlv - tf.square(zm) - tf.exp(zlv), axis=-1)
            loss = tf.reduce_mean(recon + BETA_KL * kl)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}


def main():
    tf.random.set_seed(SEED); np.random.seed(SEED)
    mode = "sensor-alphabet filter (42-dim)" if FILTER else f"NO filter (ablation, top-{TOP_K} freq)"
    print(f"[PHASE 0] data — mode = {mode}")
    Wtr, Wval_cal, Wval_test, Watk_cal, Watk_test, featurize = build_phase0()
    print(f"           train={len(Wtr)} cal(b={len(Wval_cal)},a={len(Watk_cal)}) "
          f"test(b={len(Wval_test)},a={len(Watk_test)})")

    print("[PHASE 1] fit RobustScaler + Keras VAE + IsolationForest (normal-only)")
    Xtr = featurize(Wtr)
    scaler = RobustScaler(quantile_range=(1.0, 99.0)).fit(Xtr)
    Str = scaler.transform(Xtr).astype(np.float32)

    vae = VAE(Str.shape[1])
    vae.compile(optimizer=keras.optimizers.Adam(LR))
    vae.fit(Str, epochs=EPOCHS, batch_size=BATCH, verbose=0)

    def recon_err(S):
        return ((vae.predict(S, verbose=0) - S) ** 2).mean(1)

    iforest = IsolationForest(n_estimators=200, contamination=0.02, random_state=0).fit(Str)
    aesc = RobustScaler().fit(recon_err(Str).reshape(-1, 1))
    ifsc = RobustScaler().fit((-iforest.score_samples(Str)).reshape(-1, 1))

    def score(wins):
        S = scaler.transform(featurize(wins)).astype(np.float32)
        za = aesc.transform(recon_err(S).reshape(-1, 1)).ravel()
        zi = ifsc.transform((-iforest.score_samples(S)).reshape(-1, 1)).ravel()
        return ALPHA * za + (1 - ALPHA) * zi

    print("[PHASE 2] calibrate thresholds + band edges on CAL")
    A_val_cal = score(Wval_cal)
    T0   = float(np.percentile(A_val_cal, T0_PCTILE))
    T_op = float(np.percentile(A_val_cal, 95))
    T_min = float(np.percentile(A_val_cal, 50))
    bp50  = float(np.percentile(A_val_cal, 50))
    bp90  = float(np.percentile(A_val_cal, 90))
    bp999 = float(np.percentile(A_val_cal, 99.9))

    print("[PHASE 3] score disjoint TEST; AUC/AP")
    A_val_test, A_atk_test = score(Wval_test), score(Watk_test)
    y = np.r_[np.zeros(len(A_val_test)), np.ones(len(A_atk_test))]
    s = np.r_[A_val_test, A_atk_test]
    auc, ap = float(roc_auc_score(y, s)), float(average_precision_score(y, s))

    tag = "_vae" if FILTER else "_vae_nofilter"
    np.savez(os.path.join(OUT, f"scored_windows{tag}.npz"),
             benign_scores=A_val_test.astype(np.float32),
             attack_scores=A_atk_test.astype(np.float32),
             T0=np.float32(T0), T_op=np.float32(T_op), T_min=np.float32(T_min),
             bp50=np.float32(bp50), bp90=np.float32(bp90), bp999=np.float32(bp999))
    report = {
        "scorer": "REAL VAE (TensorFlow/Keras: Gaussian encoder + reparameterization + KL) + IsolationForest",
        "feature_mode": mode,
        "vae_arch": {"input": int(Str.shape[1]), "hidden": H_DIM, "latent": Z_DIM,
                     "epochs": EPOCHS, "batch": BATCH, "lr": LR, "beta_kl": BETA_KL,
                     "framework": "tensorflow " + tf.__version__},
        "feature_dim": Xtr.shape[1], "window_len": bd.WIN_LEN, "alpha": ALPHA,
        "calibration": {"T0_p99.5": round(T0,3), "T_op_p95": round(T_op,3), "T_min_p50": round(T_min,3),
                        "benign_p90": round(bp90,3), "benign_p99.9": round(bp999,3)},
        "test_quality_benign_vs_attack": {"AUC_ROC": round(auc,4), "AP": round(ap,4),
            "recall@T0": round(float((A_atk_test>T0).mean()),4),
            "FPR@T0": round(float((A_val_test>T0).mean()),4)},
    }
    with open(os.path.join(OUT, f"detector_report{tag}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["test_quality_benign_vs_attack"], indent=2))
    print(f"OK -> output/scored_windows{tag}.npz (+ detector_report{tag}.json)")


if __name__ == "__main__":
    main()
