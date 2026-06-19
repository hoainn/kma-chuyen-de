# Experiment — RQ2: Noisy-neighbor EMA-threshold evasion of DeSFAM (on DongTing)

Self-contained experiment for the adversarial RQ2 (`../Docs/01_hypothesis.md`):
*can a noisy-neighbor co-tenant inflate DeSFAM's online EMA anomaly threshold so a
**stealthy** attack evades, while a fixed threshold still catches it?* — grounded in
the **real DongTing dataset** (`../DongTing_Official/npz`, pre-encoded syscall-ID sequences).

> Scope: this is a **mechanism experiment** on real DongTing-derived anomaly scores.
> The live Kubernetes + Tetragon validation is W3–W4 (needs a cluster; not run here).

## Pipeline (reproduce in ≤ ~2 min on CPU)
```bash
cd Experiment
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
unzip -o ../DongTing_Official/npz.zip -d data/         # → data/npz/DT-*.npz
.venv/bin/python build_detector.py    # train DeSFAM-style detector on real DongTing → output/scored_windows.npz
.venv/bin/python ema_evasion.py       # RQ2 experiment → output/{trials,summary}.csv, stats.json, fig1,fig2
```

## What each stage does
- **`build_detector.py`** — DeSFAM-style SyscallAD on real DongTing. **Sensor-alphabet filtering**
  (24 security syscalls) BEFORE windowing (len 10 / stride 2); features = freq(24) + **disc(15 binary
  presence)** + 3 stats (42-dim); **normal-only** training, ensemble `A = 0.7·z(AE)+0.3·z(iForest)`
  (DeSFAM α=0.7), `T0 = p99.5` benign. *VAE approximated by a CPU dense auto-encoder — no TF/torch.*
  Detector quality (benign-val vs attack): **AUC 0.848, AP 0.977, recall@T0 0.078, FPR 0.32%**
  — matches DeSFAM's DongTing-test ensemble (~0.835); sensor-alphabet filtering adds **+0.20 AUC** over
  the unfiltered top-20-freq variant (0.647).
- **`ema_evasion.py`** — runs 3 threshold-update policies on the SAME real score streams:
  `fixed` (T=T_op const), `ema_uncond` (paper Eq.3, every window), `ema_cond`
  (Eq.3 but **update only on non-flagged A≤T windows** → T monotonically non-increasing).
  Factors: noise ∈ {none,moderate,high} (real benign-drift windows by score quantile) ×
  attack_class ∈ {stealthy,loud} (real attack windows by score band). 40 trials/cell,
  bootstrap CIs + Wilcoxon/Kruskal/Spearman at trial level.

## Headline result (real DongTing scores; detector AUC 0.848)
| condition (high noise, stealthy) | fixed | paper-EMA (uncond) | hardened-EMA (cond) |
|---|---|---|---|
| recall | **1.00** | **0.715** | 1.00 |
| evasion (caught-by-fixed, missed) | — | **28.5%** | 0.0% |
| threshold at attack | 32.66 | 42→~52 (inflates) | ~0.03 (floored) |

- **Paper EMA is vulnerable:** noisy-neighbor benign drift inflates the global threshold → **28.5% of
  stealthy attack windows evade** (Wilcoxon p≈1.7e-8); evasion grows monotonically with noise
  (Kruskal p≈1.9e-12, Spearman ρ=−0.59).
- **Hardening defeats it:** conditional "update-only-on-non-flagged-windows" keeps T non-increasing →
  **0% evasion** (recall 1.0) for both attack classes. This is the defensive contribution.
- **Eq.3 is broadly unstable:** even loud attacks self-inflate the threshold under uncond-EMA
  (~54% evasion at high noise, ~42% with no neighbor) — not a clean "resist" control, but extra
  evidence that the published update rule is fragile; the conditional variant fixes this too.

## Pipeline phases (build_detector.py prints them) + feature engineering
**P0 Data** (filter+window) → **P1 Train** (fit scaler/AE/iForest on normal-train only) →
**P2 Calibrate** (`T0/T_op/T_min` + band edges on CAL split) → **P3 Test** (score + AUC on disjoint TEST).
`CAL ∩ TEST = ∅` — no threshold/band is fit on the evaluation windows. Per-window features (42-dim):
`freq(24)` + `disc(15 binary presence, dilution-robust)` + `stats(3: entropy, unique, max-freq)` →
RobustScaler(p1,p99) → ensemble `0.7·z(AE)+0.3·z(iForest)`. Diagrams `output/fig0_pipeline.png` &
`output/fig_features.png` (HTML in `pipeline.html`/`featureeng.html`, rendered via headless Chromium).

## Files
`build_detector.py` · `ema_evasion.py` · `make_pipeline.py` · `pipeline.html` · `featureeng.html` ·
`requirements.txt` · `data/npz/` (extracted) · `output/` (scored_windows.npz, detector_report.json,
trials.csv, summary.csv, stats.json, fig0_pipeline.png, fig_features.png, fig1/fig2.png).

## Honest caveats (see `../Docs/04_metrics_and_validity.md`, `../Docs/10_*`)
- Dense-AE ≠ DeSFAM's VAE; absolute scores differ, but the EMA-inflation dynamic is **scorer-agnostic**.
- Detector AUC 0.848 (sensor-alphabet filtered, ≈ DeSFAM-test); the fixed-vs-EMA evasion gap holds at
  both the filtered (0.848) and unfiltered (0.647) detector — i.e. it is not an artifact of detector strength.
- Stream order (warm→noise→attack) abstracts real co-tenant scheduling; single **global** threshold
  assumed (paper Eq.3) — per-container thresholds would localize the effect (Q5/live verification).
- DongTing = kernel-fuzzing syscalls, not live K8s privesc TTPs (external validity).
