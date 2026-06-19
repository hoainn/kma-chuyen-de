#!/usr/bin/env python3
"""
build_detector.py — DeSFAM-style syscall anomaly detector on REAL DongTing,
with EXPLICIT, DISJOINT pipeline phases for reliability:

  PHASE 0  DATA      : load DongTing; sensor-alphabet filter (24 syscalls) + windowing (10/2)
  PHASE 1  TRAIN     : fit Scaler + dense-AE + IsolationForest + component-scalers
                       on NORMAL-TRAIN windows ONLY (normal-only, no attack contamination)
  PHASE 2  CALIBRATE : on a CALIBRATION split (½ normal-val + ½ attack), set the
                       threshold T0/T_op/T_min and the score-band edges (noise & attack)
  PHASE 3  TEST      : on a DISJOINT TEST split (other ½ + other ½), report AUC/AP and
                       export TEST window scores for the experiment

Calibration and test windows are DISJOINT  ⇒  the operating point / score bands used by
ema_evasion.py are never fitted on the same windows that form the evaluation streams.

Output: output/scored_windows.npz (TEST scores + calibration params) + output/detector_report.json
"""
from __future__ import annotations
import json, os
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ  = os.path.join(HERE, "data", "npz")
OUT  = os.path.join(HERE, "output"); os.makedirs(OUT, exist_ok=True)
TBL  = os.path.join(HERE, "..", "DongTing_Official", "syscall_64.tbl")

ALPHABET = ["execve","clone","setuid","setgid","capset","openat","unlinkat","renameat2",
            "mount","unshare","setns","pivot_root","socket","connect","bind","mprotect",
            "mmap","prctl","memfd_create","splice","ptrace","init_module","finit_module","bpf"]
DISC = ["setuid","setgid","capset","unshare","setns","pivot_root","mount","ptrace",
        "init_module","finit_module","bpf","memfd_create","splice","renameat2","unlinkat"]

WIN_LEN, WIN_STRIDE = 10, 2
MIN_ALPHA    = 5
ALPHA        = 0.7
T0_PCTILE    = 99.5
RNG          = np.random.default_rng(0)
MAX_TRAIN_WINDOWS = 60000


def _name_to_id():
    m = {}
    with open(TBL) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 3 and p[0].isdigit() and p[1] != "x32":
                m[p[2]] = int(p[0])
    return m

_NAME2ID  = _name_to_id()
ALPHA_IDS = [_NAME2ID[n] for n in ALPHABET]
DISC_IDS  = [_NAME2ID[n] for n in DISC]


def load_part(name, idx):
    arr = np.load(os.path.join(NPZ, f"{name}.npz"), allow_pickle=True)["arr_0"]
    return [np.asarray(s, dtype=np.int64) for s in arr[idx]]


def windows(seqs):
    out = []
    for s in seqs:
        f = s[np.isin(s, ALPHA_IDS)]
        if len(f) < MIN_ALPHA:
            continue
        if len(f) < WIN_LEN:
            out.append(f)
        else:
            for i in range(0, len(f) - WIN_LEN + 1, WIN_STRIDE):
                out.append(f[i:i + WIN_LEN])
    return out


def featurize(win_list):
    from collections import Counter
    aidx = {sid: i for i, sid in enumerate(ALPHA_IDS)}
    didx = [ALPHA_IDS.index(d) for d in DISC_IDS]
    K = len(ALPHA_IDS)
    X = np.zeros((len(win_list), K + len(DISC_IDS) + 3), dtype=np.float64)
    for r, w in enumerate(win_list):
        cc = Counter(w.tolist()); counts = np.zeros(K)
        for sid, n in cc.items():
            if sid in aidx: counts[aidx[sid]] = n
        L = max(1, len(w)); freq = counts / L
        disc = (counts[didx] > 0).astype(float)
        p = freq[freq > 0]
        entropy   = float(-(p * np.log2(p)).sum()) if p.size else 0.0
        X[r, :K] = freq
        X[r, K:K+len(DISC_IDS)] = disc
        X[r, K+len(DISC_IDS):]  = [entropy, float((counts > 0).sum()/K), float(counts.max()/L)]
    return X


def split_half(wins, seed):
    idx = np.arange(len(wins)); np.random.default_rng(seed).shuffle(idx)
    h = len(idx) // 2
    return [wins[i] for i in idx[:h]], [wins[i] for i in idx[h:]]


def main():
    print("[PHASE 0 · DATA] load DongTing + sensor-alphabet filter + windowing (10/2)")
    Wtr  = windows(load_part("DT-Normal", 0))
    Wval = windows(load_part("DT-Normal", 1))
    Watk = windows(load_part("DT-Abnormal", 0))
    if len(Wtr) > MAX_TRAIN_WINDOWS:
        sel = RNG.choice(len(Wtr), MAX_TRAIN_WINDOWS, replace=False)
        Wtr = [Wtr[i] for i in sel]
    # DISJOINT calibration / test splits (benign-val and attack each split 50/50)
    Wval_cal, Wval_test = split_half(Wval, 1)
    Watk_cal, Watk_test = split_half(Watk, 2)
    print(f"           windows: train={len(Wtr)} | cal(benign={len(Wval_cal)},attack={len(Watk_cal)})"
          f" | test(benign={len(Wval_test)},attack={len(Watk_test)})  [cal ∩ test = ∅]")

    print("[PHASE 1 · TRAIN] fit Scaler + dense-AE + IsolationForest on NORMAL-TRAIN only")
    Xtr = featurize(Wtr)
    scaler = RobustScaler(quantile_range=(1.0, 99.0)).fit(Xtr)
    Str = scaler.transform(Xtr)
    ae = MLPRegressor(hidden_layer_sizes=(12, 6, 12), activation="tanh",
                      solver="adam", max_iter=80, random_state=0).fit(Str, Str)
    iforest = IsolationForest(n_estimators=200, contamination=0.02, random_state=0).fit(Str)
    aesc = RobustScaler().fit(((ae.predict(Str) - Str) ** 2).mean(1).reshape(-1, 1))
    ifsc = RobustScaler().fit((-iforest.score_samples(Str)).reshape(-1, 1))

    def score(wins):
        S = scaler.transform(featurize(wins))
        za = aesc.transform(((ae.predict(S) - S) ** 2).mean(1).reshape(-1, 1)).ravel()
        zi = ifsc.transform((-iforest.score_samples(S)).reshape(-1, 1)).ravel()
        return ALPHA * za + (1 - ALPHA) * zi

    print("[PHASE 2 · CALIBRATE] thresholds + score-band edges from CALIBRATION split")
    A_val_cal, A_atk_cal = score(Wval_cal), score(Watk_cal)
    T0   = float(np.percentile(A_val_cal, T0_PCTILE))
    T_op = float(np.percentile(A_val_cal, 95))      # operating point (FPR≈5%)
    T_min = float(np.percentile(A_val_cal, 50))     # EMA floor
    bp50  = float(np.percentile(A_val_cal, 50))     # noise band edges (benign)
    bp90  = float(np.percentile(A_val_cal, 90))
    bp999 = float(np.percentile(A_val_cal, 99.9))   # stealthy/loud attack boundary

    print("[PHASE 3 · TEST] score DISJOINT held-out windows; report AUC/AP")
    A_val_test, A_atk_test = score(Wval_test), score(Watk_test)
    y = np.r_[np.zeros(len(A_val_test)), np.ones(len(A_atk_test))]
    s = np.r_[A_val_test, A_atk_test]
    auc, ap = float(roc_auc_score(y, s)), float(average_precision_score(y, s))

    np.savez(os.path.join(OUT, "scored_windows.npz"),
             benign_scores=A_val_test.astype(np.float32),     # TEST only
             attack_scores=A_atk_test.astype(np.float32),
             T0=np.float32(T0), T_op=np.float32(T_op), T_min=np.float32(T_min),
             bp50=np.float32(bp50), bp90=np.float32(bp90), bp999=np.float32(bp999))
    report = {
        "dataset": "DongTing (npz, real syscall sequences)",
        "feature_mode": "sensor-alphabet (24 syscalls; freq24 + disc15 + stats3)",
        "phases": {
            "train_normal_windows": len(Wtr),
            "calibrate_windows": {"benign": len(Wval_cal), "attack": len(Watk_cal)},
            "test_windows": {"benign": len(Wval_test), "attack": len(Watk_test)},
            "disjoint_cal_test": True},
        "feature_dim": Xtr.shape[1], "window_len": WIN_LEN, "window_stride": WIN_STRIDE, "alpha": ALPHA,
        "calibration": {"T0_p99.5": round(T0,3), "T_op_p95": round(T_op,3),
                        "T_min_p50": round(T_min,3), "benign_p90": round(bp90,3),
                        "benign_p99.9": round(bp999,3), "source": "CALIBRATION split only"},
        "test_quality_benign_vs_attack": {"AUC_ROC": round(auc,4), "AP": round(ap,4),
            "recall@T0": round(float((A_atk_test>T0).mean()),4),
            "FPR@T0": round(float((A_val_test>T0).mean()),4)},
        "note": "cal∩test=∅: thresholds/bands fitted on CAL, streams built from TEST → no leakage.",
    }
    with open(os.path.join(OUT, "detector_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["test_quality_benign_vs_attack"], indent=2))
    print("OK -> output/scored_windows.npz (+ calibration params), detector_report.json")


if __name__ == "__main__":
    main()
