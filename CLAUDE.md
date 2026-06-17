# CLAUDE.md — kma-chuyen-de

Reproducing the **DeSFAM** paper (eBPF + AI runtime security for cloud containers) on the **DongTing** dataset, with a live Kubernetes detector via Tetragon.

---

## Project Layout

```
kma-chuyen-de/
├── DeSFAM/
│   ├── training/          # Model training (Docker-based)
│   │   ├── train.py       # Single source of truth for all ML functions
│   │   ├── train_dongting.ipynb  # Academic notebook (imports from train.py)
│   │   ├── Dockerfile
│   │   └── docker-compose.pipeline.yml
│   ├── inference/         # Live detector service
│   │   ├── detect.py      # Orchestrator (~60 lines)
│   │   ├── src/           # Modular components (loader, extractor, windower, featurizer, scorer, alerter)
│   │   ├── tetragon_grpc/ # gRPC stubs
│   │   ├── syscalls_x86_64.py
│   │   ├── Dockerfile
│   │   └── docker-compose.yml   # detector + prometheus + grafana
│   ├── model/             # Saved artifacts (mounted by inference at runtime)
│   ├── Kubernetes/        # K8s manifests: Tetragon install + tracing policy + NodePort
│   └── diagram/           # desfam-architecture.excalidraw + .png
├── DongTing/
│   ├── DB/                # MongoDB dump zips (import with mongorestore)
│   ├── syscall_64.tbl     # Linux syscall table (x86-64)
│   └── Source Code Files/ # Original DongTing paper code
└── paper/
    ├── src/
    │   ├── main_en.tex
    │   ├── sections_en/   # 00–07 + appendix_a.tex  ← PAPER IS THE AUTHORITY
    │   └── references.bib
    └── build.sh           # Docker-based LaTeX build → build/en/main_en.pdf
```

---

## Training

```bash
cd DeSFAM/training

# Interactive (Jupyter Lab at http://localhost:8888)
docker compose -f docker-compose.pipeline.yml up jupyter

# Headless / fast run
docker compose -f docker-compose.pipeline.yml run --rm train
```

The two modes are kept in sync:
- **`train.py`** — single source of truth. All ML functions live here, no matplotlib.
- **`train_dongting.ipynb`** — imports everything from `train.py`, adds plots + academic commentary.
- To change the algorithm, **edit `train.py` only**. Notebook picks up changes on kernel restart.

**Output artifacts** (saved to `../model/`):

| File | Purpose |
|---|---|
| `feature_scaler.joblib` | RobustScaler(p1..p99) for 149-dim feature vector |
| `iforest.joblib` | Isolation Forest (300 trees, contamination=0.02) |
| `vae_encoder.weights.h5` | VAE encoder |
| `vae_decoder.weights.h5` | VAE decoder |
| `ensemble_vae_scaler.joblib` | Per-component score scaler |
| `ensemble_if_scaler.joblib` | Per-component score scaler |
| `ensemble_params.json` | alpha=0.7 + fitted flag |
| `fe_report.json` | Vocab: top_ids, disc_ids, bigrams, ver_cols |
| `results.json` | Metrics, latent_dim, hidden_dim, threshold |

---

## Inference

```bash
cd DeSFAM/inference

# Start detector + Prometheus + Grafana
docker compose up --build

# Detector only
docker compose up detector

# Key env vars (override in docker-compose.yml or .env)
TETRAGON_ADDR=10.200.118.136:30321   # Tetragon gRPC NodePort
TARGET_NAMESPACE=demo                 # K8s namespace to monitor
THRESHOLD=0.08                        # Anomaly score threshold
ARTIFACTS_DIR_HOST=../model           # Host path to model artifacts
```

Grafana: http://localhost:3000 (admin/admin)
Prometheus: http://localhost:9090

**Run inference with the live-calibrated threshold** (else the compose default
`THRESHOLD=0` uses the conservative trained T_0=0.81):
```bash
cd DeSFAM/inference
docker compose --env-file .env.uat up -d --build   # detector T_0=0.25, EMA_TMIN=0.20
```

**Tetragon logs in Grafana (Loki).** Loki runs in-cluster (ns `monitoring`,
NodePort 31100); Promtail tails the Tetragon `export-stdout` event stream + ns
demo pod logs and pushes to Loki; local Grafana adds it as the `Loki` datasource.
Dashboard panels: "Tetragon syscall events — ns demo / all" and "ns demo — workload logs".
LogQL selectors: `{namespace="kube-system", container="export-stdout"}` (events),
`{namespace="demo"}` (workloads).

---

## Kubernetes / Tetragon

```bash
cd DeSFAM/Kubernetes

# Install Tetragon + apply NodePort
bash tetragon/install.sh

# Apply syscall tracing policy
kubectl apply -f tetragon/tracing-policy.yaml

# Deploy demo workloads (normal + attack)
bash deploy.sh

# Loki + Promtail (Tetragon-log → Grafana via NodePort 31100)
kubectl apply -f loki/loki.yaml -f loki/promtail.yaml
# Teardown Loki:  kubectl delete ns monitoring

# Teardown
bash teardown.sh
```

---

## Paper

```bash
cd paper
bash build.sh   # Requires Docker → outputs build/en/main_en.pdf
```

**IMPORTANT — paper sections are read-only:**
- `paper/src/sections_en/` is the canonical truth for all claims, figures, and table values.
- **Never modify paper `.tex` files.** The notebook/code must match the paper, not the other way around.
- DeSFAM does **NOT** publish a per-model AUC/AP/F1 table for SyscallAD on DongTing — only aggregate headline (AUC≈0.94, F1≈0.92, precision≈94%, recall≈90%), itself internally inconsistent. Do NOT treat any per-model "published" AUC as a target.
- The `0.646/0.747/0.656` values some old notes/logs call "paper numbers" are a **mislabelled earlier run**, not DeSFAM's — do not use them.
- `paper/src/sections_en/05_evaluation.tex` Tab. `tab:dongting_reproduction` (IF 0.882 / VAE 0.960 / Ensemble 0.921) is **this project's own reproduction target**, not external published data.

---

## Feature Vector (sensor-alphabet, §IV.B.3)

Per sliding window (length 15, stride 3 — the canonical default):

```
[ freq_F | disc_D | stats_8 | cat_K | temporal_T? | prefixspan_P ]
```

**Sensor alphabet (the key alignment, 2026-05-30).** The live Tetragon policy
`syscall-ad-tracing` emits only **23 security-relevant syscalls** (execve, clone,
setuid/setgid/capset, openat, unlinkat, renameat2, mount, unshare, setns,
pivot_root, socket, connect, bind, mprotect, mmap, prctl, memfd_create, splice,
ptrace, init/finit_module, bpf — NO read/write/close/futex). `train.py`
(`SENSOR_ALPHABET=1`, default) filters DongTing to exactly this set before
windowing, so training and serving share one feature space. Without it, training
on the full syscall universe makes every live window degenerate → benign ≈ attack
(see `docs/auto-iter-log.md` iter-2). `load_syscall_table` also skips x32 ABI rows
so `execve`/`ptrace` use the canonical IDs the live extractor uses.

- `freq_F` (F=8): per-window frequency of the common alphabet syscalls.
- `disc_D` (D=15): **binary presence** of the rarer privileged syscalls
  (ptrace/splice/unshare/setns/mount/bpf/…). Dilution-robust: a lone `ptrace` in a
  15-`openat` window still flips the bit (plain frequency under-weights it).
- `stats_8`: entropy / unique-count / length / max-freq / … derived from freq.
- `cat_K` (K=10): normalised per-window frequency over functional syscall categories
  (process, file, memory, network, signal, time, ipc, security, io_event, other) —
  the paper's "Categorical Frequencies" behavioural signature.
- `temporal_T` (T=3, **conditional**): Δt mean/std/max; activates only with per-syscall
  timestamps. DongTing (`strace -v -f`) has none → T=0, `fe_report.json: has_temporal=false`.
- `prefixspan_P` (P=2): `[matched-flag, longest-match/L]` from matching the window against a
  PrefixSpan **Benign Pattern Database** mined on normal training windows (paper "Access List
  Pattern Matching"). DB persisted as `benign_patterns.json`.

DongTing default = **43 dims** (freq_8 + disc_15 + stats_8 + cat_10 + prefixspan_2).
Scaling: `RobustScaler(quantile_range=(1.0, 99.0))` fit on **normal training windows only**.
The vocab sizes are env-tunable (`TOP_K_FREQ/TOP_K_DISC/TOP_K_BIGRAMS/ENABLE_STATS`);
`bigrams` are off (noisy on a 24-symbol alphabet). **Data Augmentation** remains a
known gap (incompatible with normal-only IF/VAE).

Result (sensor-alphabet model): VAE val AUC 0.944 (posterior collapse resolved),
ensemble val 0.948, DongTing-test ensemble AUC 0.835 / F1 0.417. Live separation:
benign mongodb ≤0.19, demo attacks 0.278–0.930. Live T_0=0.25 in `inference/.env.uat`.

Data source: `train.py` reads the full DongTing release (`.log` files + `Baseline.xlsx`
splits) under `DATA_ROOT`. `train.py` and `featurizer.py` (inference) must produce
**bit-identical** vectors — guarded by `tests/test_feature_parity.py` (passing).

---

## DongTing Dataset (MongoDB)

Three collections in `syzbot_DB`:

| Collection | Content |
|---|---|
| `kernel_convert_baseline` | Master index: filename → label / split / kernel version |
| `kernel_syscall_normal_strace` | Normal sequences — pipe-separated integer syscall IDs |
| `kernel_syscallhook_bugpoc_trace_sum` | Attack sequences — pipe-separated syscall names |

Splits: `DTDS-train` / `DTDS-validation` / `DTDS-test`

To restore DB from zips:
```bash
cd DongTing/DB
for f in *.zip; do unzip -o "$f"; done
mongorestore --db syzbot_DB .
```

---

## Key Design Decisions

1. Normal-only training for both IF and VAE (no attack contamination in scaler/vocab/model)
2. Multi-seed VAE training (seeds 0,1,2) — keep best val-AUC seed
3. Ensemble: `0.7 × RobustScale(vae_error) + 0.3 × RobustScale(if_score)`
4. Threshold from validation set only (no test-set leakage) — stored in `results.json`
5. Inference reads threshold from `results.json` at startup via `loader.py`
