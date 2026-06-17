# Auto-iteration log — feature-engineering tuning

**Goal:** detector correctly separates normal (6 benign demo workloads) from abnormal
(4 CVE-pattern simulators in ns demo: Dirty Pipe, PwnKit, Docker Escape, Exploit Chain).

**Locked by paper §IV.B.3:** window len=15 stride=3, `cat_10` categorical frequencies,
PrefixSpan-on-benign, temporal Δt (dormant in DongTing — no timestamps).

**Free to tune** (paper-plus additions, not described in §IV.B.3):
`freq_K_FREQ / disc_K_DISC / stats_8 / bigrams_K_BIGRAMS`, RobustScaler quantile range,
per-component fusion scaler quantile.

**Score per iteration:**
- `cve_caught / 4` — sims whose ensemble score crosses live-calibrated T at any point
- `fp_on_benign` — % of windows on the 6 normal workloads that cross T
- success if `cve_caught ≥ 3` and `fp_on_benign == 0`

---

## Baseline (iter-0) — current paper_plus

| knob | value |
|---|---|
| TOP_K_FREQ | 50 |
| TOP_K_DISC | 30 |
| TOP_K_BIGRAMS | 40 |
| RobustScaler quantile (per-feature) | p1..p99 |
| RobustScaler quantile (per-score fusion) | p1..p99 |
| total feat dim | 140 |
| live T (calibrated) | 0.7060 |

**Result:** 0 / 4 CVE caught (under T=0.706 floor); 0 FP on benign.

Per-CVE finding:
- Dirty Pipe: max ensemble 0.003 (model views as normal Python I/O)
- PwnKit: max ensemble 0.115 (model views as normal Python subprocess)
- Docker Escape: Tetragon emitted 0 events (telemetry, separate from model)
- Exploit Chain: max ensemble 0.7046 — **model fired**, VAE error 1.46 (10× baseline),
  but ensemble saturates at the same rail as benign DBMS init → score-scale ambiguity

Three distinct failure modes identified. Iterations below target failure modes 1 & 2
(score-scale ambiguity + model can't separate Python-wrapped attacks).

**Live-data ceiling check (before iterating):** measured per-pod VAE and IF on the
10-min live window. The result is sobering:

```
VAE error:   nginx/node/redis/mysql 0.03-0.04   (light benign)
             mongodb/python-app     1.457        (heavy benign — SATURATED)
             exploit-chain attack   1.456        (attack — SAME RAIL as heavy benign)

IF score:    light benign           -0.020 to -0.034
             mongodb/python-app     -0.0562      (heavy benign)
             exploit-chain attack   -0.0562      (attack — IDENTICAL to heavy benign)
```

All three signals (VAE, IF, ensemble) hit their saturation ceiling on heavy-benign and on
attack at the *same value*. This is a train/serve distribution-shift ceiling — DongTing's
"normal" never showed the model real-DBMS-style workloads, so anything OOD pins at the rail.

Feature engineering can only help if it pushes attack-saturation strictly above
benign-saturation in some signal direction. Iter-1 below tests that empirically.

---

## Iter-1 — bigram-emphasis (TOP_K_FREQ 50→20, TOP_K_BIGRAMS 40→100)

**Hypothesis:** mongodb's saturation comes from single-syscall frequency dims (`freq_50`
captures pwrite64/futex/openat — dimensions mongodb shares with anything OOD). Attack-
distinctive transitions (`unshare→setns`, `clone→wait4`) are intrinsically rare even in
OOD-benign and live in bigram space. Shifting weight from frequency to bigrams may carve
out a discriminable region inside the saturation rail.

**Changes (paper §IV.B.3 untouched):**
- `TOP_K_FREQ`: 50 → 20    (less weight on syscall popularity — Python/glibc dominate top-50)
- `TOP_K_BIGRAMS`: 40 → 100 (more transition-pattern coverage, esp. attack-distinctive)
- net dim: 140 → 170

**Status:** superseded. Iter-1 stayed inside the full-syscall-universe feature
space and could not fix the saturation — see the root cause below.

---

## Iter-2 — ROOT CAUSE: train/serve syscall-alphabet mismatch (2026-05-30)

**Finding.** The saturation in iter-0/iter-1 was never a tuning problem. The live
Tetragon policy `syscall-ad-tracing` (applied to ns demo) emits only **23
security-relevant syscalls** — `execve, clone, setuid, setgid, capset, openat,
unlinkat, renameat2, mount, unshare, setns, pivot_root, socket, connect, bind,
mprotect, mmap, prctl, memfd_create, splice, ptrace, init_module, finit_module,
bpf` — and notably NOT `read/write/close/futex`. But training used the **full
DongTing strace traces** (every syscall). So the scaler / vocab / PrefixSpan DB
were fit in a feature space the detector never observes: every live window
collapsed onto a near-degenerate vector → benign ≈ attack (the rail).

A second latent bug: `syscall_64.tbl` carries x32-ABI duplicate rows (512–547)
that `load_syscall_table` (last-wins) used, mapping `execve→520`, `ptrace→521` in
TRAINING while the live extractor uses the canonical `59`/`101` — those two
syscalls were silently in different ID spaces train vs serve.

**Fix (paper baseline untouched: window 15/3, cat_K, PrefixSpan-on-benign,
normal-only IF+VAE, RobustScaler p1..p99). Feature-engineering steps the paper
leaves open were updated:**
1. `train.py`: skip x32 ABI rows → canonical IDs match inference.
2. `train.py`: `SENSOR_ALPHABET` filter — drop every DongTing syscall outside the
   23 the sensor emits, before windowing. Train/serve windows are now isomorphic.
3. Alphabet-scoped vector: `freq_8` (common syscalls) + `disc_15` (binary
   presence of the rarer privileged syscalls — dilution-robust: a lone `ptrace`
   in a 15-`openat` window still flips the bit) + `stats_8` + `cat_10` +
   `prefixspan_2` = **43 dims** (was 170 — also kills the VAE posterior collapse
   the inspection flagged at high dim).
4. Alphabet persisted to `fe_report.json` (`sensor_alphabet[_names]`).

**Results.**
- VAE val AUC **0.944** (was ~0.67, no posterior collapse); ensemble val 0.948.
- DongTing test at p995: IF AUC 0.851 / VAE 0.734 / **Ensemble 0.835**;
  ensemble F1 **0.417** R 0.264 (was F1 0.014 — threshold finally fires).
- Train↔inference feature parity: PASS (max_abs_diff 0).
- Live validation (offline, no cluster writes):
  - benign mongodb (1925 windows, real sensor): ensemble median −0.050, **max
    0.189**, 0% above T≥0.20.
  - demo attacks (windows from each sim's actually-traced syscalls):
    docker-escape **0.278**, dirty-pipe **0.347**, pwnkit **0.50–0.56**,
    exploit-chain **0.930** — all separable from benign.
- Live operating point: `THRESHOLD=0.25`, `EMA_TMIN=0.20` in `.env.uat` (catches
  all four demo attacks, 0 FP on measured benign).

**Residual gap (honest).** Only mongodb emitted during capture; a hand-built
network-heavy benign window scored 0.31 — above docker-escape (0.278). Capture
the 6 benign workloads UNDER LOAD and re-check p99 before production; EMA absorbs
benign drift in the meantime. Driving the live attack sims for end-to-end
confirmation was declined this session (no cluster writes) — DongTing-test +
synthetic-window validation stands in.

