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
- Paper Table 1 target values: IF AUC=0.646, VAE AUC=0.747, Ensemble AUC=0.656; F1≈0.883, Recall≈0.984.

---

## Feature Vector (149 dims)

```
[ freq_60 | disc_40 | stats_8 | bigrams_40 | ver_1 ]
```

- `freq_60`: normalised frequency of top-60 syscalls by count
- `disc_40`: binary presence of next-40 discriminative syscalls
- `stats_8`: entropy, unique-count, log-length, max-freq, p75, std, coverage, raw-length/1000
- `bigrams_40`: normalised bigram frequency
- `ver_1`: one-hot kernel version (5.12 only in DongTing normal set → 1 column; paper uses 174 dims with 26 versions)

Scaling: `RobustScaler(quantile_range=(1.0, 99.0))` fit on **normal training sequences only**.

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
