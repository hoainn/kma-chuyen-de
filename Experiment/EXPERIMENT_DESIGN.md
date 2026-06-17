# Experiment Redesign — SyscallAD reproduction on the full DongTing dataset

**Object of study:** the SyscallAD (SysAD) anomaly-detection module of DeSFAM.
**Goal:** reproduce SyscallAD faithfully on the *full* DongTing release and produce
trustworthy, falsifiable evidence of whether the mechanism reproduces — addressing
every methodological finding from the peer-review panel (see report `ReportV2`, §6–§8).

This design **reuses the proven trainer** `DeSFAM/training/train.py` (single source of
truth for the model) and **adds a rigorous evaluation layer** (`rigorous_eval.py`) that
the original pipeline lacked. Nothing about the model/featurizer is reinvented.

---

## 1. Dataset (full release)

`DongTing_Official/` (Zenodo 6627050, ~87 GB raw):

| Artifact | Role |
|---|---|
| `Normal_data/` (4 suite zips) | 6,850 normal syscall-sequence `.log` files |
| `Abnormal_data/` (26 kernel-version zips) | 12,116 attack `.log` files |
| `Baseline.xlsx` (`kernel_convert_baseline`) | master index: label + split + kernel version per sequence |
| `syscall_64.tbl` | name↔ID map (kernel 5.17, x86-64) |

- Splits are the dataset-native `DTDS-train / -validation / -test` (80/10/10).
- `.log` members are read **streaming from the inner zips** (bounded by
  `DATA_MAX_SEQ_LEN=50000`) so the 85 GB is never materialised.
- **Encoding asymmetry (validity-relevant):** normal sequences are stored as integer
  IDs, attack sequences as syscall *names*. `load_syscall_table` (x32 ABI rows skipped)
  maps both onto the same ID space; the evaluator audits this (see §5.4).

---

## 2. Fixed pipeline configuration (pinned in `config.env`)

| Param | Value | Note |
|---|---|---|
| Sliding window (len, stride) | (15, 3) | canonical |
| Sensor alphabet | ON — 23 security syscalls | train/serve feature-space alignment |
| Feature vector | 43-dim `[freq8\|disc15\|stats8\|cat10\|prefixspan2]` | temporal OFF (no timestamps) |
| Scaler | `RobustScaler(1,99)` on **normal-train windows only** | leakage control |
| iForest | 300 trees, contamination 0.02 | normal-only |
| VAE | 32→8→32, SELU, dropout 0.2, L2 | normal-only, **seeds {0,1,2}** |
| Ensemble | `0.7·RS(VAE) + 0.3·RS(iForest)` | α adopted (not re-tuned) |
| Threshold | `p995` of benign-validation fused score | test never used for selection |

---

## 3. What is reproduced vs. adapted (honesty ledger)

| Item | Status |
|---|---|
| VAE+iForest ensemble, (15,3) windows, α=0.7, normal-only | **reproduced** |
| Temporal Δt features | adapted: OFF (DongTing has no timestamps) |
| Syscall alphabet | adapted: 23 security syscalls (matches Tetragon sensor) |
| Threshold | adapted: fixed (EMA presented but not used, for comparability) |
| DeSFAM Modules 1 & 3 | out of scope (not re-implemented) |

---

## 4. Primary protocol (proven trainer)

`train.py` on the full dataset produces, per model (iForest / VAE / Ensemble), the
**window-level** test metrics (AUC, AP, F1, Precision, Recall, FPR) + per-seed VAE
validation AUC → `results.json`. This is the baseline result table (report Table I).

---

## 5. Rigorous evaluation layer (the redesign — `rigorous_eval.py`)

Reloads the trained artifacts + dataset and computes what the original pipeline did
**not**, each item tracing to a specific reviewer finding:

### 5.1 Sequence-level metrics (review K2 — window-label inheritance)
Window labels are inherited from the sequence, so attack sequences contain many
benign-looking windows → window-level recall/F1 are biased. Fix: also aggregate to
**sequence level** (a sequence is flagged if **any** of its windows exceeds the
threshold; `seq_score = max(window scores)`), and report AUC/AP/F1 at both granularities.
Threshold for the sequence decision is selected on **validation sequence scores only**.

### 5.2 disc-only ablation (review K1, DA-CRITICAL — the confound)
Tests whether detection is genuine sequence-anomaly detection or merely
"a privileged syscall appeared". Computes, on **unscaled** features, the `disc` block
(binary-presence bits) and scores each window/sequence by the **count of disc bits set**.
Reports `disc-only` AUC/AP at window and sequence level **next to the full ensemble**.
- If `AUC(disc-only) ≈ AUC(ensemble)` → the confound is real; the report must retreat to
  "presence detection is reproducible; the learned model adds [measured] margin".
- If `AUC(ensemble) ≫ AUC(disc-only)` → the model uses sequence structure beyond presence.

### 5.3 disc-presence statistic (review K2)
Fraction of **attack windows** containing ≥1 privileged (`disc`) syscall. A low fraction
proves many attack-labeled windows are benign-looking (quantifies the label-noise bias).

### 5.4 Cross-encoding parity audit (review — methodology W3)
For each of the 23 alphabet syscalls: its ID, and its occurrence count in the
normal (ID-encoded) vs attack (name-encoded) streams. Asymmetric presence flags a
mapping bug that would set `disc` bits inconsistently across classes.

### 5.5 Seed stability (review — report mean±SD)
VAE validation AUC mean ± SD across seeds {0,1,2} (from `results.json:vae_seed_reports`).
Full per-seed **test** variance is enabled via `MULTISEED_TEST_EVAL=1` (set in
`config.env`): the trainer evaluates each seed's full ensemble on the test split and
writes `results.json:ensemble_seed_test_summary` (mean ± SD of AUC/AP/F1/Precision/Recall).
`rigorous_eval.py` surfaces it in `SUMMARY.md`.

### 5.6 Reproduction criterion (review K4)
**Important:** DeSFAM publishes **no per-model AUC/AP/F1 table** for SyscallAD on DongTing —
only aggregate headline numbers (AUC≈0.94, F1≈0.92, precision≈94%, recall≈90%), which are
themselves internally inconsistent. The earlier `0.646/0.747/0.656` "published" values were a
**mislabelled earlier run**, not paper numbers, and have been removed.
So there is no valid per-model published baseline to test against. The criterion is therefore
**qualitative**: (a) train↔serve feature parity holds; (b) ranking is high and seed-stable
(ensemble AUC ≥ 0.80); (c) precision orientation preserved (benign FPR ≤ target). Any mention of
DeSFAM is a *qualitative* reference to the aggregate headline, not a per-model pass/fail.

All of the above are written to `results/<run>/rigorous_metrics.json` and a
`SUMMARY.md` that maps 1:1 onto report `ReportV2` §7 tables.

---

## 6. Outputs → report mapping

| Output | Feeds report |
|---|---|
| `results.json: models.*` | §7 Table I (window-level) |
| `rigorous_metrics.json: sequence_level` | §7 new sequence-level row |
| `rigorous_metrics.json: ablation_disc_only` | §7 ablation row + §8.4 confound verdict |
| `rigorous_metrics.json: attack_window_disc_fraction` | §8.5 label-noise threat |
| `rigorous_metrics.json: cross_encoding_parity` | §6 parity note |
| `rigorous_metrics.json: success_criterion` | §7.2 honest comparison + verdict |
| `vae_seed_reports` mean±SD | §7 seed stability |

---

## 7. Reproduce

```bash
cd Experiment
bash prepare_data.sh        # verify DongTing_Official layout
bash run_experiment.sh      # full-dataset train + rigorous eval (Docker)
# → results/run_<timestamp>/{results.json, rigorous_metrics.json, SUMMARY.md, model/}
```

> First-run note: `rigorous_eval.py` has not yet been executed end-to-end against a
> trained artifact set; validate its output on the first full run and reconcile its
> recomputed window-level ensemble AUC against `results.json` (they must match).
