# Experiment — SyscallAD reproduction (full DongTing)

Redesigned, peer-review-driven experiment for the report `ReportV2`. It reproduces the
**SyscallAD** module of DeSFAM on the **full** DongTing release and produces *falsifiable*
evidence, fixing the methodological gaps the review panel raised.

See **`EXPERIMENT_DESIGN.md`** for the full protocol and the finding→fix mapping.

## Layout
| File | Role |
|---|---|
| `EXPERIMENT_DESIGN.md` | the redesigned protocol (read first) |
| `config.env` | pinned full-dataset config + published baselines + success thresholds |
| `prepare_data.sh` | verify/extract `DongTing_Official/` into the expected layout |
| `run_experiment.sh` | Docker: verify → train (full data) → rigorous eval |
| `rigorous_eval.py` | sequence-level metrics, disc-only ablation, parity, success criterion |
| `results/run_<ts>/` | per-run outputs (`results.json`, `rigorous_metrics.json`, `SUMMARY.md`, `model/`, logs) |

## Run
```bash
cd Experiment
bash run_experiment.sh
```
Requires Docker. Uses the dataset at `../DongTing_Official` (already unpacked) and the
proven trainer at `../DeSFAM/training/train.py` as the model engine.

## Engine vs. redesign
- **Engine (unchanged):** `DeSFAM/training/train.py` — model, featurizer, training. Produces
  window-level metrics in `results.json`.
- **Redesign (new, here):** `rigorous_eval.py` — reloads the trained artifacts + dataset and
  adds what the engine lacked:
  - **sequence-level** AUC/AP/F1 (not just window-level) — review K2;
  - **disc-only ablation** (presence-vs-sequence confound) — review K1 / DA-CRITICAL;
  - **attack-window disc-presence fraction** — review K2;
  - **cross-encoding parity audit** (normal=IDs, attack=names) — review W3;
  - **falsifiable success criterion** vs published AUC ±0.05 — review K4;
  - VAE seed mean±SD.

## Outputs → report `ReportV2` §7/§8
`SUMMARY.md` is written to map 1:1 onto the report's pending `\TODOexp{}` cells:
window vs sequence table, ablation row + confound verdict, parity note, success verdict.

> **First-run validation:** `rigorous_eval.py` recomputes the window-level ensemble AUC and
> logs it next to `results.json`'s value — they must match. If they don't, stop and
> reconcile before trusting the sequence/ablation numbers.
