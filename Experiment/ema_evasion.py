#!/usr/bin/env python3
"""
ema_evasion.py — RQ2 adversarial experiment on REAL DongTing anomaly scores.

Hypothesis (H1_2b): a noisy-neighbor co-tenant emitting benign-but-anomalous
syscall windows can inflate DeSFAM's online EMA anomaly threshold (paper §IV.B.3,
Eq. 3), letting a *stealthy* attack slip under the raised bar — while a fixed
threshold still catches it.

We test THREE threshold-update policies on the SAME real score streams:
  • fixed      : T = T_op constant (no adaptation)            [control]
  • ema_uncond : T_{t+1}=max(T_min, βT+(1-β)A)  every window  [paper Eq.3 as written]
  • ema_cond   : same, but UPDATE ONLY on non-flagged (A≤T) windows
                 → T is monotonically non-increasing ⇒ cannot be inflated
                 (a hardening of Eq.3; "attack/FP scores cannot drag T up")

Scores come from build_detector.py (real DongTing windows, dense-AE+iForest
ensemble, α=0.7). β=0.9 (DeSFAM). Operating point T_op = benign p95 (FPR≈5%),
a realistic operational threshold; EMA floor T_min = benign p50.

Factors: noise ∈ {none,moderate,high} × attack_class ∈ {stealthy,loud}.
Output: output/trials.csv, summary.csv, stats.json, *.png
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "output")
RNG  = np.random.default_rng(42)

# Scorer arm: dense-AE (default) or VAE via env vars (apples-to-apples re-run).
SCORED_NPZ = os.environ.get("SCORED_NPZ", os.path.join(OUT, "scored_windows.npz"))
SUF        = os.environ.get("OUT_SUFFIX", "")        # e.g. "_vae"

BETA          = 0.9
N_WARM        = 50
N_NOISE       = 150
N_ATTACK      = 30
N_TRIALS      = 40
POLICIES      = ["fixed", "ema_uncond", "ema_cond"]
NOISE_LEVELS  = ["none", "moderate", "high"]
ATTACK_CLASSES = ["stealthy", "loud"]


def run_stream(scores_warm, scores_noise, scores_attack, T_op, T_min):
    """Run all three policies over one ordered stream; return per-policy
    (attack_flags array, T_at_first_attack, full T-trace for fixed phase)."""
    stream = list(scores_warm) + list(scores_noise) + list(scores_attack)
    n_attack = len(scores_attack)
    atk_start = len(scores_warm) + len(scores_noise)
    res = {}
    for pol in POLICIES:
        T = T_op
        flags, T_at_attack, trace = [], None, []
        for i, a in enumerate(stream):
            flagged = a > T
            if i >= atk_start:
                if T_at_attack is None:
                    T_at_attack = T
                flags.append(bool(flagged))
            # threshold update
            if pol == "ema_uncond":
                T = max(T_min, BETA * T + (1 - BETA) * a)
            elif pol == "ema_cond":
                if not flagged:                      # non-flagged only
                    T = max(T_min, BETA * T + (1 - BETA) * a)
            trace.append(T)
        res[pol] = (np.array(flags), float(T_at_attack), trace)
    return res, atk_start


def main():
    # TEST window scores (PHASE 3) + thresholds/band-edges calibrated on the
    # DISJOINT CALIBRATION split (PHASE 2). No threshold/band is fitted on the
    # TEST windows that form the evaluation streams below.
    d = np.load(SCORED_NPZ)
    benign, attack = d["benign_scores"].astype(float), d["attack_scores"].astype(float)
    T_op, T_min = float(d["T_op"]), float(d["T_min"])      # from CAL
    bp50, bp90, b99_9 = float(d["bp50"]), float(d["bp90"]), float(d["bp999"])  # band edges from CAL

    # real benign pools (TEST) by intensity, split using CAL-derived edges
    pools_noise = {
        "none":     benign[benign <= bp50],
        "moderate": benign[(benign > bp50) & (benign <= bp90)],
        "high":     benign[benign > bp90],
    }
    warm_pool = benign[benign <= bp50]
    # real attack pools (TEST) by class (detectable-by-fixed: score > T_op)
    det = attack[attack > T_op]
    pools_attack = {
        "stealthy": det[det <= b99_9],     # just above operating point — maskable
        "loud":     det[det >  b99_9],     # strongly anomalous
    }
    meta = {
        "split": "thresholds/bands from CAL; streams from disjoint TEST",
        "T_op_cal_p95": round(T_op, 4), "T_min_cal_p50": round(T_min, 4),
        "beta": BETA, "benign_p99.9_cal": round(b99_9, 4),
        "pool_sizes": {**{f"noise_{k}": int(v.size) for k, v in pools_noise.items()},
                       **{f"attack_{k}": int(v.size) for k, v in pools_attack.items()}},
        "n_trials": N_TRIALS, "stream": {"warm": N_WARM, "noise": N_NOISE, "attack": N_ATTACK},
    }
    print("Config:", json.dumps(meta, indent=2))

    rows = []
    example_traces = {}     # (noise) -> traces for the plot, attack_class=stealthy
    for ac in ATTACK_CLASSES:
        apool = pools_attack[ac]
        for nz in NOISE_LEVELS:
            npool = pools_noise[nz]
            for t in range(N_TRIALS):
                sw = RNG.choice(warm_pool, N_WARM)
                sn = RNG.choice(npool,     N_NOISE)
                sa = RNG.choice(apool,     N_ATTACK)
                res, _ = run_stream(sw, sn, sa, T_op, T_min)
                rec = {p: res[p][0].mean() for p in POLICIES}
                row = {"attack_class": ac, "noise": nz, "trial": t,
                       "recall_fixed": rec["fixed"],
                       "recall_ema_uncond": rec["ema_uncond"],
                       "recall_ema_cond": rec["ema_cond"],
                       # evasion = caught by fixed but missed by the EMA policy
                       "evasion_uncond": float(((res["fixed"][0]) & (~res["ema_uncond"][0])).mean()),
                       "evasion_cond":   float(((res["fixed"][0]) & (~res["ema_cond"][0])).mean()),
                       "T_attack_uncond": res["ema_uncond"][1],
                       "T_attack_cond":   res["ema_cond"][1]}
                rows.append(row)
                if ac == "stealthy" and t == 0:
                    example_traces[nz] = {p: res[p][2] for p in POLICIES}
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"trials{SUF}.csv"), index=False)

    # ---- aggregate (mean + bootstrap 95% CI at trial level) ----
    def ci(x):
        x = np.asarray(x); bs = [RNG.choice(x, x.size).mean() for _ in range(2000)]
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    summ = []
    for ac in ATTACK_CLASSES:
        for nz in NOISE_LEVELS:
            g = df[(df.attack_class == ac) & (df.noise == nz)]
            for pol in POLICIES:
                col = f"recall_{pol}" if pol != "fixed" else "recall_fixed"
                lo, hi = ci(g[col]); summ.append({
                    "attack_class": ac, "noise": nz, "policy": pol,
                    "recall_mean": round(g[col].mean(), 4),
                    "recall_ci": f"[{lo:.3f},{hi:.3f}]",
                    "T_at_attack_mean": round(g[f"T_attack_{pol.replace('ema_','')}"].mean(), 3)
                                        if pol != "fixed" else round(T_op, 3)})
    summ_df = pd.DataFrame(summ); summ_df.to_csv(os.path.join(OUT, f"summary{SUF}.csv"), index=False)

    # ---- statistics ----
    st = {}
    s_hi = df[(df.attack_class == "stealthy") & (df.noise == "high")]
    s_no = df[(df.attack_class == "stealthy") & (df.noise == "none")]
    # H1_2a trend: stealthy recall_uncond decreasing with noise (Kruskal + Spearman)
    groups = [df[(df.attack_class=="stealthy")&(df.noise==nz)]["recall_ema_uncond"] for nz in NOISE_LEVELS]
    H, pK = stats.kruskal(*groups)
    nz_rank = df[df.attack_class=="stealthy"]["noise"].map({"none":0,"moderate":1,"high":2})
    rho, pS = stats.spearmanr(nz_rank, df[df.attack_class=="stealthy"]["recall_ema_uncond"])
    st["H1_2a_trend_stealthy_uncond"] = {"kruskal_H": round(float(H),3), "p": float(pK),
        "spearman_rho_noise_vs_recall": round(float(rho),3), "p_spearman": float(pS)}
    # H1_2b paired: under high noise stealthy, fixed catches more than ema_uncond (evasion via inflation)
    w, pw = stats.wilcoxon(s_hi["recall_fixed"], s_hi["recall_ema_uncond"], alternative="greater")
    st["H1_2b_uncond_evasion_highnoise_stealthy"] = {
        "recall_fixed_mean": round(s_hi["recall_fixed"].mean(),4),
        "recall_ema_uncond_mean": round(s_hi["recall_ema_uncond"].mean(),4),
        "evasion_uncond_mean": round(s_hi["evasion_uncond"].mean(),4),
        "wilcoxon_p_fixed_gt_uncond": float(pw),
        "T_attack_uncond_mean": round(s_hi["T_attack_uncond"].mean(),3), "T_op": round(T_op,3)}
    # mitigation: ema_cond is NOT inflated → recall_cond ≥ recall_fixed, evasion≈0
    st["mitigation_cond_highnoise_stealthy"] = {
        "recall_ema_cond_mean": round(s_hi["recall_ema_cond"].mean(),4),
        "evasion_cond_mean": round(s_hi["evasion_cond"].mean(),4),
        "T_attack_cond_mean": round(s_hi["T_attack_cond"].mean(),3)}
    # loud control under high noise
    l_hi = df[(df.attack_class=="loud") & (df.noise=="high")]
    st["loud_control_highnoise"] = {
        "recall_fixed_mean": round(l_hi["recall_fixed"].mean(),4),
        "recall_ema_uncond_mean": round(l_hi["recall_ema_uncond"].mean(),4),
        "evasion_uncond_mean": round(l_hi["evasion_uncond"].mean(),4)}
    st["_meta"] = meta
    with open(os.path.join(OUT, f"stats{SUF}.json"), "w") as f:
        json.dump(st, f, indent=2)

    # ---- plots (NO pie; per Appendix C) ----
    # Fig 1: threshold trajectory under high noise (the key mechanism figure)
    plt.figure(figsize=(8,4))
    tr = example_traces["high"]
    for pol, c in zip(POLICIES, ["#888","#d62728","#2ca02c"]):
        plt.plot(tr[pol], label=pol, color=c, lw=1.6)
    atk0 = N_WARM + N_NOISE
    plt.axvline(N_WARM, ls=":", c="gray"); plt.axvline(atk0, ls="--", c="k")
    plt.text(N_WARM+2, plt.ylim()[1]*0.95, "noise starts", fontsize=8)
    plt.text(atk0+2, plt.ylim()[1]*0.95, "attack starts", fontsize=8)
    plt.axhline(T_op, ls="-.", c="blue", lw=0.8, label=f"T_op={T_op:.2f}")
    plt.xlabel("window index"); plt.ylabel("anomaly threshold T")
    plt.title("EMA threshold trajectory under HIGH noisy-neighbor load (stealthy trial)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"fig1_threshold_trajectory{SUF}.png"), dpi=130); plt.close()

    # Fig 2: recall vs noise (stealthy) by policy — box plots
    fig, ax = plt.subplots(figsize=(8,4))
    width=0.25
    for j,pol in enumerate(POLICIES):
        col = f"recall_{pol}" if pol!="fixed" else "recall_fixed"
        data=[df[(df.attack_class=="stealthy")&(df.noise==nz)][col].values for nz in NOISE_LEVELS]
        pos=[k+ (j-1)*width for k in range(len(NOISE_LEVELS))]
        bp=ax.boxplot(data, positions=pos, widths=width*0.9, patch_artist=True,
                      medianprops=dict(color="black"))
        for b in bp["boxes"]: b.set_facecolor(["#cccccc","#f4a3a3","#a3d6a3"][j])
        ax.plot([],[],color=["#888","#d62728","#2ca02c"][j],label=pol)
    ax.set_xticks(range(len(NOISE_LEVELS))); ax.set_xticklabels(NOISE_LEVELS)
    ax.set_xlabel("noisy-neighbor intensity"); ax.set_ylabel("recall (stealthy attack windows)")
    ax.set_title("Stealthy-attack recall vs noise intensity, by threshold policy")
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"fig2_recall_vs_noise{SUF}.png"), dpi=130); plt.close()

    print("\n=== SUMMARY (recall by policy × noise × attack_class) ===")
    print(summ_df.to_string(index=False))
    print("\n=== KEY STATS ===")
    print(json.dumps({k:v for k,v in st.items() if k!="_meta"}, indent=2))
    print("\nOK -> output/{trials.csv,summary.csv,stats.json,fig1_*.png,fig2_*.png}")


if __name__ == "__main__":
    main()
