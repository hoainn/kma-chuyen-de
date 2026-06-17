# Experiment SUMMARY (rigorous evaluation)

- experiment: `dongting_15_3`  window=(15, 3)
- **Success criterion:** FAIL (ensemble AUC 0.835 vs published 0.656, tol 0.05; FPR 0.008)

## Window vs sequence level (ensemble)
| level | AUC | AP | F1 | Precision | Recall |
|---|---|---|---|---|---|
| window | 0.835 | 0.990 | 0.417 | 0.999 | 0.264 |
| sequence | 0.787 | 0.871 | 0.502 | 0.977 | 0.338 |

## disc-only ablation (confound test)
- window AUC: disc-only **0.875** vs ensemble **0.835** → CONFOUND_LIKELY (disc-only ~ ensemble)
- sequence AUC: disc-only 0.678 vs ensemble 0.787
- attack-window disc-presence fraction: 0.835

## Seed stability
- VAE val AUC 0.943 ± 0.001  (seeds [0.9426124727993862, 0.9438259081525643, 0.9430260533502602])
- Ensemble **test** AUC 0.835 ± 0.000  (per-seed [0.835, 0.835, 0.836])

## Cross-encoding parity
- 2 of 24 alphabet syscalls flagged asymmetric: setgid, unlinkat
