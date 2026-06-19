#!/usr/bin/env python3
"""
vae_scorer.py — REAL VAE arm for the EMA-evasion study (reviewer item: scorer fidelity).

DeSFAM's published SyscallAD uses a *Variational* Autoencoder. The main build
(build_detector.py) substitutes a dense autoencoder so the repro stays
scikit-learn-only / CPU / <15 min. This script provides the missing arm: a
genuine VAE (Gaussian encoder + reparameterization trick + KL-regularized
latent + Adam), implemented in NumPy so no deep-learning framework is needed.

It is a drop-in replacement for the AE *component only*:
    A(W) = 0.7 · z(VAE reconstruction error) + 0.3 · z(IsolationForest)
Everything else is identical to build_detector.py and the SAME CAL/TEST windows
are used (same split seeds), so the dense-AE vs VAE comparison is apples-to-apples.

Output: output/scored_windows_vae.npz (same schema as scored_windows.npz)
        output/detector_report_vae.json
Then:   SCORED_NPZ=output/scored_windows_vae.npz OUT_SUFFIX=_vae python3 ema_evasion.py
"""
from __future__ import annotations
import json, os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# reuse the EXACT data path / filter / windowing / featurization of the main build
import build_detector as bd

OUT = bd.OUT
ALPHA = bd.ALPHA            # 0.7 ensemble weight (VAE component)
T0_PCTILE = bd.T0_PCTILE    # 99.5

# ---- VAE hyper-parameters (small net; bottleneck mirrors the dense-AE) ----
H_DIM, Z_DIM = 24, 6
EPOCHS, BATCH, LR = 40, 256, 1e-3
BETA_KL = 0.5               # KL weight (modest → avoids posterior collapse on Z=6)
SEED = 0

# FILTER=1 (default): sensor-alphabet filter + 42-dim features (the detector).
# FILTER=0: ablation — no filter, top-20 most-frequent syscalls' frequency (20-dim).
FILTER = os.environ.get("FILTER", "1") == "1"
TOP_K = 20


def _xavier(rng, nin, nout):
    return rng.normal(0, np.sqrt(2.0 / (nin + nout)), size=(nin, nout))


class VAE:
    """Gaussian VAE: enc 42->H->(mu,logvar in R^Z); dec Z->H->42 (linear out)."""
    def __init__(self, d, rng):
        self.W1 = _xavier(rng, d, H_DIM);     self.b1 = np.zeros(H_DIM)
        self.Wmu = _xavier(rng, H_DIM, Z_DIM); self.bmu = np.zeros(Z_DIM)
        self.Wlv = _xavier(rng, H_DIM, Z_DIM); self.blv = np.zeros(Z_DIM)
        self.W2 = _xavier(rng, Z_DIM, H_DIM); self.b2 = np.zeros(H_DIM)
        self.W3 = _xavier(rng, H_DIM, d);     self.b3 = np.zeros(d)
        self._keys = ["W1","b1","Wmu","bmu","Wlv","blv","W2","b2","W3","b3"]
        # Adam state
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self._keys}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self._keys}
        self.t = 0

    def encode(self, X):
        A1 = X @ self.W1 + self.b1; H1 = np.tanh(A1)
        Mu = H1 @ self.Wmu + self.bmu
        LV = H1 @ self.Wlv + self.blv
        return A1, H1, Mu, LV

    def decode(self, Z):
        A2 = Z @ self.W2 + self.b2; H2 = np.tanh(A2)
        Xhat = H2 @ self.W3 + self.b3
        return A2, H2, Xhat

    def recon_error(self, X):
        """Deterministic anomaly score: reconstruction MSE using z = mu."""
        _, _, Mu, _ = self.encode(X)
        _, _, Xhat = self.decode(Mu)
        return ((Xhat - X) ** 2).mean(1)

    def step(self, X, eps, lr):
        B = X.shape[0]
        A1, H1, Mu, LV = self.encode(X)
        Std = np.exp(0.5 * LV)
        Z = Mu + Std * eps
        A2, H2, Xhat = self.decode(Z)

        # --- backward (loss = mean-batch [0.5*sum_d recon^2] + BETA*KL) ---
        dXhat = (Xhat - X) / B
        dW3 = H2.T @ dXhat;            db3 = dXhat.sum(0)
        dH2 = dXhat @ self.W3.T
        dA2 = dH2 * (1 - H2 ** 2)
        dW2 = Z.T @ dA2;               db2 = dA2.sum(0)
        dZ = dA2 @ self.W2.T

        dMu = dZ.copy()
        dLV = dZ * eps * Std * 0.5
        # KL grads (mean over batch), weighted
        dMu += BETA_KL * (Mu / B)
        dLV += BETA_KL * (0.5 * (np.exp(LV) - 1.0) / B)

        dWmu = H1.T @ dMu; dbmu = dMu.sum(0)
        dWlv = H1.T @ dLV; dblv = dLV.sum(0)
        dH1 = dMu @ self.Wmu.T + dLV @ self.Wlv.T
        dA1 = dH1 * (1 - H1 ** 2)
        dW1 = X.T @ dA1; db1 = dA1.sum(0)

        grads = {"W1":dW1,"b1":db1,"Wmu":dWmu,"bmu":dbmu,"Wlv":dWlv,"blv":dblv,
                 "W2":dW2,"b2":db2,"W3":dW3,"b3":db3}
        # Adam
        self.t += 1; b1a, b2a, eps_a = 0.9, 0.999, 1e-8
        for k in self._keys:
            g = grads[k]
            self.m[k] = b1a * self.m[k] + (1 - b1a) * g
            self.v[k] = b2a * self.v[k] + (1 - b2a) * (g * g)
            mhat = self.m[k] / (1 - b1a ** self.t)
            vhat = self.v[k] / (1 - b2a ** self.t)
            setattr(self, k, getattr(self, k) - lr * mhat / (np.sqrt(vhat) + eps_a))

        rec = 0.5 * ((Xhat - X) ** 2).sum(1).mean()
        kl = (-0.5 * (1 + LV - Mu ** 2 - np.exp(LV)).sum(1)).mean()
        return rec + BETA_KL * kl


def windows_full(seqs):
    """No-filter windowing: window the FULL syscall sequence (len 10, stride 2)."""
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


def make_featurizer_top20(Wtr_full):
    """Top-20 most-frequent syscall IDs on normal-train → 20-dim frequency featurizer."""
    from collections import Counter
    c = Counter()
    for w in Wtr_full:
        c.update(w.tolist())
    top_ids = [sid for sid, _ in c.most_common(TOP_K)]
    tidx = {sid: i for i, sid in enumerate(top_ids)}

    def feat(win_list):
        X = np.zeros((len(win_list), len(top_ids)), dtype=np.float64)
        for r, w in enumerate(win_list):
            L = max(1, len(w))
            cc = Counter(w.tolist())
            for sid, n in cc.items():
                if sid in tidx:
                    X[r, tidx[sid]] = n / L
        return X
    return feat, top_ids


def build_phase0():
    """Replicate build_detector PHASE 0 deterministically → identical windows.
    Returns windows + the featurizer to use (filtered 42-dim, or top-20 ablation)."""
    win = bd.windows if FILTER else windows_full
    Wtr  = win(bd.load_part("DT-Normal", 0))
    Wval = win(bd.load_part("DT-Normal", 1))
    Watk = win(bd.load_part("DT-Abnormal", 0))
    if len(Wtr) > bd.MAX_TRAIN_WINDOWS:
        sel = np.random.default_rng(0).choice(len(Wtr), bd.MAX_TRAIN_WINDOWS, replace=False)
        Wtr = [Wtr[i] for i in sel]
    Wval_cal, Wval_test = bd.split_half(Wval, 1)
    Watk_cal, Watk_test = bd.split_half(Watk, 2)
    if FILTER:
        featurize = bd.featurize
    else:
        featurize, _ = make_featurizer_top20(Wtr)
    return Wtr, Wval_cal, Wval_test, Watk_cal, Watk_test, featurize


def main():
    rng = np.random.default_rng(SEED)
    mode = "sensor-alphabet filter (42-dim)" if FILTER else f"NO filter (ablation, top-{TOP_K} freq)"
    print(f"[PHASE 0 · DATA] mode = {mode}")
    Wtr, Wval_cal, Wval_test, Watk_cal, Watk_test, featurize = build_phase0()
    print(f"           train={len(Wtr)} | cal(b={len(Wval_cal)},a={len(Watk_cal)})"
          f" | test(b={len(Wval_test)},a={len(Watk_test)})  [cal ∩ test = ∅]")

    print("[PHASE 1 · TRAIN] fit RobustScaler + REAL VAE + IsolationForest (normal-only)")
    Xtr = featurize(Wtr)
    scaler = RobustScaler(quantile_range=(1.0, 99.0)).fit(Xtr)
    Str = scaler.transform(Xtr).astype(np.float64)

    vae = VAE(Str.shape[1], rng)
    n = Str.shape[0]
    for ep in range(EPOCHS):
        order = rng.permutation(n)
        last = 0.0
        for i in range(0, n, BATCH):
            xb = Str[order[i:i + BATCH]]
            eps = rng.standard_normal((xb.shape[0], Z_DIM))
            last = vae.step(xb, eps, LR)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"           epoch {ep+1:2d}/{EPOCHS}  ELBO-loss={last:.4f}")

    iforest = IsolationForest(n_estimators=200, contamination=0.02, random_state=0).fit(Str)
    aesc = RobustScaler().fit(vae.recon_error(Str).reshape(-1, 1))
    ifsc = RobustScaler().fit((-iforest.score_samples(Str)).reshape(-1, 1))

    def score(wins):
        S = scaler.transform(featurize(wins)).astype(np.float64)
        za = aesc.transform(vae.recon_error(S).reshape(-1, 1)).ravel()
        zi = ifsc.transform((-iforest.score_samples(S)).reshape(-1, 1)).ravel()
        return ALPHA * za + (1 - ALPHA) * zi

    print("[PHASE 2 · CALIBRATE] thresholds + band edges from CAL split")
    A_val_cal = score(Wval_cal)
    T0   = float(np.percentile(A_val_cal, T0_PCTILE))
    T_op = float(np.percentile(A_val_cal, 95))
    T_min = float(np.percentile(A_val_cal, 50))
    bp50  = float(np.percentile(A_val_cal, 50))
    bp90  = float(np.percentile(A_val_cal, 90))
    bp999 = float(np.percentile(A_val_cal, 99.9))

    print("[PHASE 3 · TEST] score DISJOINT held-out windows; report AUC/AP")
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
        "scorer": "REAL VAE (NumPy: Gaussian encoder + reparameterization + KL) + IsolationForest",
        "feature_mode": mode,
        "vae_arch": {"input": int(Str.shape[1]), "hidden": H_DIM, "latent": Z_DIM,
                     "epochs": EPOCHS, "batch": BATCH, "lr": LR, "beta_kl": BETA_KL},
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
