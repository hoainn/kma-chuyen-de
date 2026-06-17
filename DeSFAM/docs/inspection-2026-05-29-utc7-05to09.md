# DeSFAM end-to-end inspection — 2026-05-29 05:00–09:00 UTC+7

> Inspection window in UTC: **2026-05-28 22:00 → 2026-05-29 02:00**
> Container clocks log in UTC; the detector's restart loop occurred at **03:22–03:24 UTC** (10:22–10:24 UTC+7), so the canonical "live" window actually spans ~10:22–10:24 UTC+7 — but artifact mutations (notebook reruns) extend through 10:51 UTC+7.

---

## Executive summary

Three live-impact issues dominate, in descending order:

1. **CRITICAL — detector cannot reach Tetragon.** The dev cluster's gRPC endpoint at `172.16.95.190:54321` refused every connection attempt across 8 boot iterations (restarts=7) in 2 minutes, after which `inference-detector-1` was killed (exit 137). No syscall events were ever ingested; no scoring, no EMA updates, no per-pod metric series exist for this window. The Grafana dashboard's live tiles for this period are empty by construction.
2. **MAJOR — paper-AUC gap is real.** `results.json` (written by the notebook at 10:51 UTC+7) records IF=0.761, VAE=0.824, Ensemble=0.754 test AUC vs the paper's Table-3 0.882 / 0.960 / 0.921 — a ~0.13 gap on every component. The VAE loss saturates at val_loss=0.1280 from epoch 5 onward (all 3 seeds), which is *posterior collapse* on the 140-dim vector. Threshold strategy `p995` then renders the VAE and Ensemble useless on test (F1=0, recall=0).
3. **MAJOR — documentation/code disagreement on the paper numbers.** The user-said-corrected paper numbers (0.882 / 0.960 / 0.921) are now in `train.py`, in the notebook source, and in `notebook outputs`. But `/Users/hoainn/Documents/Project/CaoHoc/CLAUDE.md` (the project-root CLAUDE.md, last edited May 28 01:11 UTC+7) still asserts `Paper Table 1 target values: IF AUC=0.646, VAE AUC=0.747, Ensemble AUC=0.656`, and `/tmp/desfam_retrain.log` (the docker-compose retrain at 23:29 UTC+7 May 28) also prints those old numbers. The retrain log's training output was overwritten by the later notebook execution at 10:51 UTC+7, so the live artifacts are consistent with the notebook — only the documentation and old retrain log carry the stale numbers.

Plus a cluster of dashboard- and config-drift papercuts (see §4–5).

---

## Timeline of events in the window

All timestamps converted to UTC+7 (the user's wall clock). Sources are noted in the rightmost column.

| UTC+7 time | Event | Source |
|---|---|---|
| 05:51:21 | Prometheus TSDB head GC + write block (no scrape activity ⇒ idle) | `inference-prometheus-1` log |
| 07:02:33 | Prometheus TSDB compaction (still idle, no detector data) | `inference-prometheus-1` log |
| 10:22:11 | Prometheus + Grafana started (cold boot, no foreknowledge of detector) | container `StartedAt` |
| 10:22:14 | First detector boot — loads artifacts, threshold=0.6787, EMA on (β=0.9, T_min=0.5) | `inference-detector-1` log |
| 10:22:20 | First "Streaming (ns_filter=dev) …" → 13 s later: `gRPC UNAVAILABLE` connection refused at `172.16.95.190:54321` | detector log |
| 10:22:35 | Boot #2 (restart=1) — identical pattern | detector log |
| 10:22:51 | Boot #3 | detector log |
| 10:23:08 | Boot #4 | detector log |
| 10:23:24 | Boot #5 | detector log |
| 10:23:40 | Boot #6 | detector log |
| 10:23:56 | Boot #7 | detector log |
| 10:24:12 | Boot #8 — last attempt | detector log |
| 10:24:20 | Detector exit 137 (SIGKILL — Docker / OOM-killer / compose-down) | container `FinishedAt`, ExitCode |
| 10:33:44 | `training/train.py` edited (paper-number fix applied) | filesystem `mtime` |
| 10:43:24 | First nbconvert rerun → **NameError: `ens_val_auc` is not defined** | `/tmp/desfam_nbrerun.log` |
| 10:51:32 | Final nbconvert rerun → all model artifacts rewritten (217 KB notebook, 220 KB → 220 KB ipynb) | model file mtimes, `/tmp/desfam_nbrerun2.log` |
| 10:51:34 | Final notebook .ipynb written with rendered outputs | `train_dongting.ipynb` mtime |
| 11:04:57 | Grafana SIGTERM (graceful shutdown), Prometheus SIGTERM | both container logs |

Outside the window but referenced:
- 06:29 UTC+7 (May 28): `desfam_retrain.log` records the docker-compose retrain that prints "(paper: 0.646/0.747/0.656)" — this is from the **pre-fix** version of train.py.

---

## Findings by layer

### 1. Training pipeline

**1.1 Posterior collapse on the VAE (MAJOR).**
- **Where:** `/tmp/desfam_retrain.log:705–771`; `model/results.json` `vae_seed_reports`
- **What:** All 3 seeds (0, 1, 2) converge to **identical** val_loss=0.1280 within 5 epochs and never improve. Epochs 5–12 are flat to 4 decimal places; early-stopping kicks in at 11–12 epochs each. Seed-to-seed val AUC variance is < 0.002 (0.7549, 0.7555, 0.7540). This is the textbook signature of posterior collapse: the encoder learns to ignore most of the 140-dim input and the decoder produces a constant reconstruction near the mean of the benign distribution.
- **Why it matters:** The VAE is the dominant component in the α=0.7 ensemble. If it has collapsed, the 0.13 paper gap is structural, not a tuning artefact. Adding more training data, raising the latent dim from 8, or KL annealing would help; the current architecture is too narrow for the 140-dim feature vector.
- **Recommendation:** Try KL-annealing (warm-up KL coefficient over the first 5 epochs), bump `hidden_dim` from 32 to 64 and `latent_dim` from 8 to 16, or move to a denoising autoencoder (the paper says VAE but does not constrain ELBO weighting).

**1.2 Threshold strategy p995 collapses VAE & Ensemble metrics (MAJOR).**
- **Where:** `model/results.json` `models.vae` & `models.ensemble`
- **What:** `f1=0, precision=0, recall=0, tp=0, fp=660` for VAE; same for Ensemble. The p995 threshold is set at the 99.5-th percentile of *benign* validation reconstruction error, then evaluated on the test set whose attack class has a *lower* recon-error distribution than the benign 99.5-th percentile. So no test attack ever crosses the threshold, and the only positives are 660 benign false-positives (≈ FPR 0.27 %, as designed).
- **Why it matters:** The current model is *useless as a detector at this threshold*. The IF model still gets recall=3.9 % (67 999 TP out of 1 758 000 attacks), the others get 0. Operators looking at the Grafana panels will see exactly zero confirmed attacks even on a real attack pod.
- **Recommendation:** Switch `THRESHOLD_STRATEGY` to `f1` (already supported in `train.py` line 1277), or compute a per-component threshold at the F1-optimal point on the val set and store both in `results.json`. The current value `0.6787` is far above the typical test attack score.

**1.3 IF val AUC actually beats the "old paper number" (informational).**
- **Where:** `/tmp/desfam_retrain.log:695`
- **What:** Pre-fix log says `IF val AUC: 0.6939  (paper: 0.646)` — i.e. our model beats *that* claim by 0.05. The post-fix train.py now says `(paper: 0.882)` so our 0.6939 is now 0.19 *below*. This is the source of the 0.646/0.747/0.656 references the user mentioned correcting.
- **Why it matters:** Confirms the user's diagnosis was correct: the previous paper numbers were wrong; the real paper Table 3 is 0.882/0.960/0.921, and the implementation legitimately falls short by ~0.13–0.17 on each component.

**1.4 Subsampling cap drops 32.9 M → 2.0 M training windows (informational).**
- **Where:** `desfam_retrain.log:585` — `Subsampling windows: 32,896,435 → 1,999,392 (cap=2,000,000, ratio=0.061)`
- **What:** Only 6.1 % of train-split windows are kept due to `MAX_WINDOWS_PER_SPLIT=2_000_000`.
- **Why it matters:** Random subsampling is well-justified (memory), but 32.9 M is a 16× headroom over the cap. If GPU/CPU permits, the cap could be raised to recover more training signal; or stratified subsampling by trace should be verified.

**1.5 Training data pipeline is healthy (no NaN / inf / abnormal drops).** `train=14,772 val=1,804 test=2,305`, drops `74 / 2 / 12` short traces — consistent percentages.

---

### 2. Notebook

**2.1 First rerun NameError (RESOLVED).**
- **Where:** `/tmp/desfam_nbrerun.log:80–92` — Cell 14 raised `NameError: name 'ens_val_auc' is not defined`
- **What:** The scorecard cell referenced `ens_val_auc`, but the model-building cell defines the variable as `ens_auc_val`. Now fixed: `train_dongting.ipynb:1178,1180,1308` all use `ens_auc_val`, and the second rerun (`/tmp/desfam_nbrerun2.log`) executed to completion (217788 → 220116 bytes).
- **Status:** **Fixed.** All 17 code cells executed, `execution_count` is contiguous 1..17, no `null` execution counts, zero empty `outputs` arrays.

**2.2 Notebook outputs are consistent with current model artifacts.** Output text matches `results.json`: IF val=0.6939, VAE val=0.7555, Ensemble val=0.7291; test 0.7610 / 0.8235 / 0.7541. PAPER dict uses `{IF: 0.882, VAE: 0.960, Ensemble: 0.921}` — correct.

**2.3 No stale 0.646 / 0.747 / 0.656 references in the .ipynb file.** Confirmed via `grep` — only `0.882 / 0.960 / 0.921` appear in the notebook source and outputs.

---

### 3. Inference / live detector

**3.1 CRITICAL — tight restart loop, never ingested an event.**
- **Where:** `docker logs inference-detector-1 --since 2026-05-28T22:00:00`; `docker inspect inference-detector-1` → `ExitCode=137 RestartCount=7`
- **What:** 8 successful boots (each completing artifact-load + gRPC `Streaming` start) in 2 minutes. Every one fails after ~13 s with `StatusCode.UNAVAILABLE — Failed to connect to remote host: Connection refused` at `172.16.95.190:54321`. Container exited with code 137 at 10:24:20 UTC+7. Zero `events=` log lines, zero ATTACK/normal score lines.
- **Why it matters:** From the operator's perspective, the dashboard was completely dark during the audit window. Anything labelled with a `pod` Prometheus label will be empty; `desfam_events_total = 0`; `desfam_ema_updates_total = 0`. The "EMA calculator" Grafana panel cannot have any new step-by-step values for this window.
- **Recommendation:** (a) Add a connect-time health probe to `extractor.py` so the detector exits with a clear "Tetragon endpoint unreachable" message instead of looping on `restart: unless-stopped`. (b) Add `--ema-quiet-burnin <N seconds>` so the EMA does not log "T_0 below typical benign" until at least `N` seconds of live data have arrived. (c) The IP looks like a private VPC address — confirm the host has VPN connectivity / `nc -z 172.16.95.190 54321` before bringing the container up.

**3.2 Endpoint port mismatch (MAJOR).**
- **Where:** `inference/.env.uat:3` (`TETRAGON_ADDR=172.16.95.190:54321`) vs `Kubernetes/tetragon/nodeport.yaml:13–15` (NodePort: `port=54321 targetPort=54321 nodePort=30321`)
- **What:** From inside the cluster (Tetragon pod port), gRPC is on `54321`. From OUTSIDE the cluster (a host like the laptop running docker-compose), the NodePort exposes it on `30321`. The detector is configured for `54321`, which is only reachable inside the cluster. CLAUDE.md (kma-chuyen-de) line 84 documents `TETRAGON_ADDR=10.200.118.136:30321` as the default; `detect.py:47` also defaults to `30321`. So the `.env.uat` override is the source of the unreachable endpoint.
- **Why it matters:** Production rollout will not work until the addressing is correct. If `172.16.95.190` is a node IP, the port must be `30321`. If it is the Tetragon pod IP, the detector must run as a pod inside the cluster (where it then hits 54321 directly).
- **Recommendation:** Change `inference/.env.uat:3` to `172.16.95.190:30321` (assuming node-IP) and confirm with `nc -vz 172.16.95.190 30321` from the docker host.

**3.3 EMA configured to never fire on the current model (MAJOR / by design but noteworthy).**
- **Where:** `inference/.env.uat:22` `EMA_TMIN=0.5`; `model/results.json` ensemble threshold=0.6787; expected benign ensemble score ≈ 0.05 per `.env.uat:17` comment
- **What:** Conditional EMA updates `T_{t+1} = max(T_min, β·T_t + (1-β)·A)` only on non-flagged windows. Since typical benign A ≈ 0.05 ≪ T_0=0.6787, every update yields `β·0.6787 + 0.1·0.05 ≈ 0.616 → 0.557 → 0.506 → 0.500 (floor)` so T converges to T_min=0.5 within ~10 windows and sits there. The `EmaThreshold.update` is correct, but T then stays at 0.5 forever (> 0.05 benign mean). Operators will see `desfam_threshold` saturate at 0.5 immediately and stay there.
- **Why it matters:** This is what the env-comment intends, but the dashboard's "Active threshold T over time" panel will look completely flat — and a sudden attack burst that drives a single benign window to e.g. 0.4 will *not* cause T to drop below 0.5 to that level. The EMA is therefore decorative on this score scale.
- **Recommendation:** If the goal is paper-faithful EMA behaviour, set `EMA_TMIN` lower (e.g. 0.05 or 0.10) so T can actually track benign drift. The current 0.5 is a useful safety floor for posterior-collapse VAE behaviour, but disables the paper's adaptive idea.

**3.4 Detector restart loop creates inflated Prometheus series count (MINOR).**
- **Where:** `metrics.py:72–77` `model_info` Gauge; restart loop spec
- **What:** Every boot increments label-set churn on `desfam_model_info{feature_mode, total_dims, has_temporal, …}`. With identical labels each boot this is fine, but the boot also resets all `Counter` values (the prometheus_client doesn't preserve counters across processes), so `desfam_events_total`, `desfam_attacks_total{pod=…}`, `desfam_ema_updates_total` all show as series-restarts in Grafana. PromQL `rate()` queries over the 03:22–03:24 UTC window will see 8 zero-resets in 2 minutes, producing spurious rate spikes.
- **Why it matters:** Dashboard panels that compute `rate(desfam_events_total[1m])` will look noisy. Bottom-anchored counters (Grafana "increase()" with a query interval crossing a restart) will report inflated values.
- **Recommendation:** Trivially fixed by `restart: on-failure:3` instead of `restart: unless-stopped` in `inference/docker-compose.yml:42`.

**3.5 Tetragon stub still imported but unused on this code path (informational).**
- **Where:** `inference/tetragon_grpc/` — present in the inference image but never reached because the gRPC channel can't connect.

---

### 4. Configuration drift

**4.1 Dashboard text describes a different threshold than is actually loaded (MAJOR).**
- **Where:** `inference/grafana/dashboards/desfam.json` lines 36, 392, 515, 519, 523, 618, 622
- **What:** Dashboard markdown asserts:
  > "**Threshold (0.0197):** … 99th-percentile of live dev-cluster scores during 2-minute calibration"
  But the live `desfam_threshold` gauge starts at `0.6787` (the trained T_0 from `results.json`), and the EMA floor is `0.5`. The threshold value `0.0197` appears 4 times in panel `thresholds.steps` arrays (gradient colour anchors) and twice in long-form description text.
- **Why it matters:** Anyone reading the dashboard during onboarding will believe the active threshold is 0.0197 when it is actually ~0.5. Severity colour-gradients (panel descriptions mention `0.0197` as a colour-anchor for "near threshold") are visually anchored to the old score scale.
- **Recommendation:** Either parameterise the panel description from a Grafana variable backed by `desfam_threshold`, or hard-code the current 0.5 (EMA floor) / 0.6787 (T_0) and refresh the description text. The `thresholds.steps` arrays should be rebased on the new score scale (`-0.5 … 1.0` instead of `-0.1 … 0.05`).

**4.2 detect.py `--tetragon-addr` default vs `.env.uat` override mismatch (MAJOR).** — see §3.2.

**4.3 `fe_report.json` no longer carries `feature_mode` but loader still publishes it (MINOR).**
- **Where:** `inference/src/loader.py:60` `mode = fe.get("feature_mode", "paper_plus")`; `model/fe_report.json` has no `feature_mode` key (only `feature_groups`, `total_dims`, …)
- **What:** Loader defaults to `"paper_plus"` unconditionally for newer artifacts. The `desfam_model_info` Gauge therefore always labels `feature_mode="paper_plus"` regardless of what was actually trained.
- **Why it matters:** Operator UX — the Grafana model-info stat appears correct only by coincidence (paper_plus is the only supported mode). If the project ever ships `paper_only` again, the label will silently lie.
- **Recommendation:** Either (a) make `train.py` write `feature_mode` back into `fe_report.json`, or (b) infer it from `total_dims` (12 → paper_only, 140 → paper_plus) in the loader so the dashboard cannot diverge.

**4.4 `kma-chuyen-de/CLAUDE.md` Tetragon docs use port 30321; `.env.uat` uses 54321 (informational, see §3.2).**

**4.5 EMA_TMIN comment vs actual score scale (MINOR).**
- **Where:** `inference/docker-compose.yml:22–24` description vs `.env.uat:17` comment
- **What:** docker-compose comment recommends "EMA_TMIN ≈ trained T_0 (see results.json -> models.ensemble.threshold)" — i.e. 0.6787; the `.env.uat` uses 0.5. Either choice is defensible (0.5 is paper-faithful, 0.6787 keeps T at the trained T_0), but the two comments push in opposite directions.
- **Recommendation:** Pick one canonical recommendation and update both files.

---

### 5. Documentation drift

**5.1 CRITICAL — `kma-chuyen-de/CLAUDE.md` still has the wrong paper numbers.**
- **Where:** `/Users/hoainn/Documents/Project/CaoHoc/kma-chuyen-de/CLAUDE.md:127`
- **What:** Line 127 reads:
  > "Paper Table 1 target values: IF AUC=0.646, VAE AUC=0.747, Ensemble AUC=0.656; F1≈0.883, Recall≈0.984."
  These are the WRONG numbers the user said had been corrected elsewhere. They have not been corrected here. File mtime is May 28 01:11 — well before the train.py fix at May 29 10:33.
- **Why it matters:** This file is the **project-canonical CLAUDE.md** loaded into every Claude Code session for this workspace. Future AI assistance will read these wrong numbers and propagate them.
- **Recommendation:** Update line 127 to `IF AUC=0.882, VAE AUC=0.960, Ensemble AUC=0.921; F1=0.932, Recall=0.996 (paper Table 3, test split, α=0.7 ensemble row).`

**5.2 `/tmp/desfam_retrain.log` archives the old paper numbers verbatim (informational).**
- **Where:** `/tmp/desfam_retrain.log:695, 773, 778`
- **What:** Log was generated by docker-compose retrain at 23:29 UTC+7 on May 28, **before** train.py was edited at 10:33 on May 29. So this log captures the pre-fix output. It is not a live regression — just a historical artefact.
- **Recommendation:** Delete or annotate when archiving for the writeup. The retrain itself never updated the model artifacts (those came from the 10:51 notebook execution).

**5.3 Notebook commentary mentions the gap explicitly (informational, OK).**
- **Where:** `train_dongting.ipynb:1594–1617`
- **What:** Vietnamese commentary acknowledges the gap, points out the previous (wrong) 0.646/0.747/0.656 numbers, and links to `paper/src/sections_en/05_evaluation.tex` as ground truth. Good.

---

### 6. Parity / correctness

**6.1 `tests/test_feature_parity.py` is in sync with current featurizer (OK).**
- **Where:** `training/tests/test_feature_parity.py`
- **What:** Exercises the full paper_plus vector path (`top_ids`, `disc_ids`, `top_bigrams`, `enable_stats=True`, cat_K, prefixspan_P) plus the temporal-off branch. Stubs TensorFlow when missing so it can run on the host without TF. The featurizer math (`inference/src/featurizer.py:transform`) tracks `train.py::build_features` group-by-group, including the per-group offset layout and stats_8 ordering.
- **Note:** Not run in this audit (no test runner output in any of the inspected logs). I cannot confirm it currently *passes*, only that the code paths look aligned.
- **Recommendation:** Add a pre-deploy hook that runs `python training/tests/test_feature_parity.py` before any `docker compose up` of the inference stack. The test exits non-zero on any drift.

**6.2 Loader / featurizer keyword-arg alignment with detect.py is correct (OK).** Spot-checked the `Featurizer(...)` call in `detect.py:228–242` against `featurizer.py:50–98` — all 12 kwargs match by name and type.

**6.3 Scorer returns the same `ens_score` formula the training Ensemble fitted (OK).** `ensemble_params.json` says `alpha=0.7 fitted=true`, and `Scorer.score()` delegates to `EnsembleScorer.score(vae, if)` (the same class used in training).

---

### 7. Empirical anomalies

**7.1 Cannot confirm `desfam_threshold` saturated at EMA_TMIN=0.5 — no events ever scored.**
- The detector boots loaded `T_0=0.6787` and set `metrics.threshold_gauge.set(0.6787)`. Until the first non-flagged window arrived, no EMA update fired, so the gauge would have stayed at 0.6787 for the entire ~13 s of each boot.
- The expected steady-state behaviour (T → 0.5) was never reached because no window was ever scored.

**7.2 Cannot confirm `desfam_ema_updates_total` was incrementing.** Same reason as 7.1 — no events.

**7.3 Per-component T_0 visible only at boot (informational).** `loader.py:158–167` parses `results.json.models.{vae,iforest,ensemble}.threshold` and `detect.py:219–220` publishes them as `desfam_component_threshold{model=…}`:
- vae: **1.41375** (from `results.json:54` — value `1.4137502908706665`)
- iforest: **0.000506** (`results.json:41` — `0.0005056147635406627`)
- ensemble: **0.6787** (`results.json:67` — `0.6787196487963363`)
These match `results.json` exactly. If the user's note says "vae~1.41 iforest~0.0005 ensemble~0.68" — that is **correct**, matches the live config.

**7.4 Prometheus TSDB blocks for the window exist and are intact.** Block IDs `01KSS4K1F…` (mint 05:51 UTC+7), `01KSS8ND6…` (07:02 UTC+7), `01KSSFMKS…` (09:04 UTC+7), `01KSSPKY5…` (11:06 UTC+7) all wrote successfully. The dashboard for the 10:22–10:24 UTC+7 window can be queried; it just will have zero per-pod data.

---

## Issues ranked by severity

**Critical**
1. **Detector cannot reach Tetragon** (§3.1) — 0 events in the window; production rollout blocked. Likely port-config issue (54321 vs 30321) — see §3.2.
2. **`kma-chuyen-de/CLAUDE.md:127` still records the wrong paper numbers** (§5.1) — propagates the bug we already fixed everywhere else.

**Major**
3. **VAE posterior collapse on 140-dim vector** (§1.1) — explains the ~0.13 paper gap.
4. **`p995` threshold strategy yields F1=0 on test for VAE & Ensemble** (§1.2) — detector at this T cannot fire on the paper's own test split.
5. **`.env.uat` `TETRAGON_ADDR=…:54321` vs `nodeport.yaml` `nodePort: 30321`** (§3.2) — root cause of §3.1.
6. **Grafana dashboard's panel descriptions and threshold colour-anchors hard-coded to 0.0197** (§4.1) — operators see stale threshold story.
7. **EMA configured to converge to floor immediately** (§3.3) — EMA is effectively static at T_min=0.5; not what the paper intends.

**Minor**
8. **`feature_mode` defaulted to "paper_plus" in loader regardless of fe_report** (§4.3).
9. **EMA_TMIN guidance is contradictory between docker-compose comment and .env.uat** (§4.5).
10. **Detector restart loop bloats Prometheus counter resets** (§3.4).

**Informational**
11. **Subsampling drops 94 % of train windows** (§1.4) — defensible.
12. **`/tmp/desfam_retrain.log` archives pre-fix paper numbers** (§5.2) — historical, not live.
13. **`feature_parity.py` is not auto-invoked before deploys** (§6.1).

---

## Recommended next actions (in order)

1. **Unblock Tetragon connectivity.** Change `inference/.env.uat:3` from `172.16.95.190:54321` to `172.16.95.190:30321` (node-IP + NodePort). Verify with `nc -vz 172.16.95.190 30321` before `docker compose up`. (Critical, < 1 minute.)
2. **Correct `kma-chuyen-de/CLAUDE.md:127`** to `IF AUC=0.882, VAE AUC=0.960, Ensemble AUC=0.921; F1=0.932 …`. (Critical, < 1 minute.)
3. **Switch to F1-optimal threshold for the live detector.** Either retrain with `THRESHOLD_STRATEGY=f1` or post-process `results.json` to compute and store a second `threshold_f1` per component. Update loader to read it. (Major, 30 minutes.)
4. **Fix Grafana dashboard descriptions.** Replace hard-coded `0.0197` strings with `$threshold` Grafana variable, or rebase to `~0.5`. Update the panel `thresholds.steps` colour anchors to the new score scale. (Major, 30 minutes.)
5. **Tighten the detector loop.** Add a connect-probe and switch `restart: unless-stopped` → `restart: on-failure:3` in `inference/docker-compose.yml:42`. (Minor, 5 minutes.)
6. **Address VAE posterior collapse** for the next training run: enable KL-annealing, raise hidden/latent dim, or add a denoising objective. (Major, half a day of experimentation.)
7. **Make `feature_parity.py` a pre-deploy gate** via a Makefile target or git pre-push hook. (Minor, 10 minutes.)
8. **Decide canonical EMA_TMIN.** If 0.5 is right, remove the docker-compose comment that recommends 0.6787; if 0.6787 is right, update .env.uat. (Minor, 2 minutes.)

---

## Evidence appendix (selected log excerpts with timestamps)

### A. Detector restart loop (UTC; add 7 h for UTC+7)

```text
03:22:14 [INFO] Loading artifacts from: /artifacts
03:22:14 [INFO]   paper_plus: freq=50 disc=30 stats=8 bigrams=40 cat=10 temporal=0 prefixspan=2  → input_dim=140
03:22:20 [INFO] Threshold loaded from results.json: 0.6787
03:22:20 [INFO] Effective threshold: 0.6787
03:22:20 [INFO] EMA threshold ON: T_0=0.6787  β=0.9  T_min=0.5  (updates only on non-flagged windows)
03:22:20 [INFO] Ready — window_len=15 stride=3 max_idle=30.0s flush_every=5.0s threshold=0.6787 kernel=6.17
03:22:20 [INFO] Connecting to Tetragon at 172.16.95.190:54321 ...
03:22:20 [INFO] Streaming (ns_filter=dev) ...
03:22:33 [ERROR] gRPC error: StatusCode.UNAVAILABLE — failed to connect to all addresses; last error: UNKNOWN: ipv4:172.16.95.190:54321: Failed to connect to remote host: Connection refused
…7 more identical iterations through 03:24:15…
docker inspect → ExitCode=137 RestartCount=7  StartedAt=03:22:11 FinishedAt=03:24:20
```

### B. VAE posterior collapse (seed 0 — seeds 1 & 2 identical to 4 dp)

```text
[07:17  ram= 6.1G]    VAE seed=0: fitting on 219,589 rows...
Epoch 1/80: 815/815 - 2s - 2ms/step - loss: 0.0879 - val_loss: 0.1299
Epoch 2/80: 815/815 - 1s - 1ms/step - loss: 0.0593 - val_loss: 0.1286
Epoch 3/80: 815/815 - 1s - 1ms/step - loss: 0.0584 - val_loss: 0.1282
Epoch 4/80: 815/815 - 1s - 988us/step - loss: 0.0582 - val_loss: 0.1281
Epoch 5/80: 815/815 - 1s - 1ms/step - loss: 0.0581 - val_loss: 0.1280
Epoch 6/80: 815/815 - 1s - 967us/step - loss: 0.0581 - val_loss: 0.1280
Epoch 7/80: 815/815 - 1s - 963us/step - loss: 0.0581 - val_loss: 0.1280
Epoch 8/80: 815/815 - 1s - 984us/step - loss: 0.0581 - val_loss: 0.1280
Epoch 9/80: 815/815 - 1s - 997us/step - loss: 0.0581 - val_loss: 0.1280
Epoch 10/80: 815/815 - 1s - 1ms/step - loss: 0.0581 - val_loss: 0.1280
Epoch 11/80: 815/815 - 1s - 1ms/step - loss: 0.0581 - val_loss: 0.1280
Epoch 12/80: 815/815 - 1s - 976us/step - loss: 0.0581 - val_loss: 0.1280
[07:50  ram= 6.6G]    seed=0  val_auc=0.7549  best_val_loss=0.1280  epochs=12
```

### C. p995-threshold zero-fire result (from `desfam_retrain.log:780–784`)

```text
Model                     AUC     AP     F1   Prec  Recall    FPR
----------------------------------------------------------------------
Isolation Forest        0.761  0.930  0.074  0.981   0.039  0.006
VAE                     0.824  0.951  0.000  0.000   0.000  0.003
Ensemble                0.754  0.923  0.000  0.000   0.000  0.003
```

### D. Notebook NameError (first rerun, fixed)

```text
File ".../notebook execute": NameError
Cell In[14], line 13
     11     ('Isolation Forest', if_auc_val,    test_if_auc,  PAPER['IF']),
     12     ('VAE (best seed)',  best_val_auc,  test_vae_auc, PAPER['VAE']),
---> 13     ('Ensemble (α=0.7)', ens_val_auc,   test_ens_auc, PAPER['Ensemble']),
NameError: name 'ens_val_auc' is not defined
```

Resolved by renaming the reference to `ens_auc_val` (consistent with the model-build cell). Second rerun (`/tmp/desfam_nbrerun2.log`) completed without error.

### E. Notebook scorecard cell output (after fix)

```text
Model               our val AUC  our test AUC  paper Table-3  Δ vs paper
Isolation Forest         0.6939        0.7610         0.8820    -0.1210
VAE (best seed)          0.7555        0.8235         0.9600    -0.1365
Ensemble (α=0.7)         0.7291        0.7541         0.9210    -0.1669
```

### F. CLAUDE.md stale paper-numbers reference

```text
/Users/hoainn/Documents/Project/CaoHoc/CLAUDE.md:127:
- Paper Table 1 target values: IF AUC=0.646, VAE AUC=0.747, Ensemble AUC=0.656; F1≈0.883, Recall≈0.984.
```

File `mtime`: `May 28 01:11:08 2026` (pre-fix).

### G. Dashboard hard-coded 0.0197 threshold

```text
inference/grafana/dashboards/desfam.json:36, 392 (description text)
inference/grafana/dashboards/desfam.json:515, 618 (thresholds.steps colour anchors)
```

Live `desfam_threshold` gauge actual value at startup: `0.6787` (then EMA-floored to `0.5`).
