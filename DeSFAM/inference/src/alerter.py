"""
Alerter — consecutive-window filter, throttled output, calibration mode.

Output format (attack):
  [1393s] ATTACK  HIGH | demo/attack-miner          | ens= 0.076 vae= 0.260 if=-0.091 | consec=14 rate=100% | lat=12ms | mmap×8 execve×2
Output format (normal, throttled to 1 line per pod per 30s):
  [1400s] normal       | demo/nginx-web             | ens=-0.042 vae= 0.031 if=-0.115 | rate= 0%           | lat= 8ms
"""

from __future__ import annotations

import collections
import sys
import time

import numpy as np

try:
    from src import metrics as _prom
except (ImportError, Exception):
    _prom = None

_NR_SYSCALL: dict | None = None


def _get_nr_syscall() -> dict:
    global _NR_SYSCALL
    if _NR_SYSCALL is None:
        from syscalls_x86_64 import NR_SYSCALL
        _NR_SYSCALL = NR_SYSCALL
    return _NR_SYSCALL


def _severity(ens: float, threshold: float) -> str:
    margin = ens - threshold
    if margin >= 0.15:  return "CRIT"
    if margin >= 0.08:  return "HIGH"
    if margin >= 0.03:  return "MED "
    return                      "LOW "


class Alerter:
    # How often to reprint a line for the same pod (seconds)
    _ATTACK_THROTTLE_S = 5
    _NORMAL_THROTTLE_S = 30
    # Rolling window size for per-pod attack-rate tracking
    _RATE_WINDOW = 20

    def __init__(
        self,
        threshold: float,
        consecutive: int = 1,
        scores_only: bool = False,
        calibrate_secs: float = 0.0,
    ):
        self._threshold      = float(threshold)
        self._consecutive    = consecutive
        self._scores_only    = scores_only
        self._calibrate_secs = calibrate_secs
        # (group, idx_in_group, semantic_name) per feature dim, set by detect.py
        self._feature_dim_names: list[tuple[str, int, str]] = []
        # Per-pod last (group,idx,name) labelsets we set, so they can be cleared
        # on the next confirmed attack.
        self._last_feature_labels: dict[str, list[tuple[str, str, str]]] = {}

        # Rolling history: per-pod deque of recent attack records. Each entry is
        # (attack_id, [(group, summary), ...]) so the next attack at this slot
        # knows which labelsets to evict from the history gauges.
        self._history_max = 20
        self._attack_history: dict[str, collections.deque] = {}

        self._consec_attack: dict[str, int]           = collections.defaultdict(int)
        self._recent: dict[str, collections.deque]    = {}   # rolling is_attack bools
        self._last_print: dict[str, float]            = {}   # wall-time of last print
        self._first_attack: dict[str, bool]           = collections.defaultdict(lambda: True)
        self._calib_scores: list[float]               = []
        # Track last attack metric labels per pod so stale series can be removed
        self._last_attack_labels: dict[str, tuple[str, str]] = {}
        self._last_args_labels: dict[str, tuple[str, str]] = {}
        self._last_detail_labels: dict[str, tuple] = {}

    def set_threshold(self, t: float) -> None:
        """Mutate the cut-off used for severity bucketing — kept in sync with
        the Scorer when detect.py runs the EMA loop (paper §IV.B.3 Eq. 3)."""
        self._threshold = float(t)

    def set_feature_dim_names(self, names: list[tuple[str, int, str]]) -> None:
        """Per-dimension (group, idx, semantic_name) for the loaded vector.
        Used to label the desfam_last_attack_feature gauge when an attack fires."""
        self._feature_dim_names = list(names)

    def _rolling_rate(self, pod_key: str) -> int:
        """Return attack % over the last _RATE_WINDOW windows for this pod."""
        dq = self._recent.get(pod_key)
        if not dq:
            return 0
        return int(100 * sum(dq) / len(dq))

    def process(
        self,
        pod_key: str,
        result: dict,
        window: list[int],
        elapsed: int,
        args_window: list[dict] | None = None,
    ) -> bool:
        ens = result["ensemble"]

        # ── Prometheus metrics (always, before throttle) ──────────────────────
        if _prom is not None:
            _prom.ensemble_score.labels(pod=pod_key).set(ens)
            _prom.vae_score.labels(pod=pod_key).set(result["vae_raw"])
            _prom.iforest_score.labels(pod=pod_key).set(result["if_raw"])
            _prom.windows_total.labels(pod=pod_key).inc()
            _prom.latency_hist.observe(result.get("latency_ms", 0))

        # ── Calibration mode ─────────────────────────────────────────────────
        if self._calibrate_secs > 0:
            self._calib_scores.append(ens)
            if elapsed >= self._calibrate_secs:
                arr = np.array(self._calib_scores)
                print(f"\n=== Calibration ({len(arr)} windows, {elapsed:.0f}s) ===")
                print(f"  mean={arr.mean():.4f}  std={arr.std():.4f}")
                print(f"  p50={np.percentile(arr,50):.4f}  p90={np.percentile(arr,90):.4f}")
                print(f"  p95={np.percentile(arr,95):.4f}  p99={np.percentile(arr,99):.4f}")
                print(f"\nRecommended thresholds:")
                print(f"  Conservative (low FP): --threshold {np.percentile(arr,99):.4f}")
                print(f"  Balanced:              --threshold {np.percentile(arr,95):.4f}")
                print(f"  Sensitive (low FN):    --threshold {np.percentile(arr,90):.4f}")
                sys.exit(0)
            return False

        # ── Consecutive filter ────────────────────────────────────────────────
        is_attack = result["is_attack"]
        if is_attack:
            self._consec_attack[pod_key] += 1
        else:
            self._consec_attack[pod_key] = 0
        consec = self._consec_attack[pod_key]
        confirmed = is_attack and consec >= self._consecutive

        if _prom is not None:
            _prom.consecutive.labels(pod=pod_key).set(consec)
            if confirmed:
                _prom.attacks_total.labels(pod=pod_key).inc()
                nr_map = _get_nr_syscall()
                top5 = collections.Counter(window).most_common(5)
                top_str = " ".join(f"{nr_map.get(k, str(k))}×{v}" for k, v in top5)
                seq_len_str = str(len(window))
                # First 12 calls → deduplicate consecutive repeats → join with →
                _PREFIX_LEN = 12
                deduped: list[str] = []
                for nr in window:
                    name = nr_map.get(nr, str(nr))
                    if not deduped or deduped[-1] != name:
                        deduped.append(name)
                prefix_calls = deduped[:_PREFIX_LEN]
                seq_prefix = "→".join(prefix_calls) + ("…" if len(deduped) > _PREFIX_LEN else "")
                # Remove stale label-set before setting new one
                if pod_key in self._last_attack_labels:
                    old_top, old_prefix, old_len = self._last_attack_labels[pod_key]
                    try:
                        _prom.last_attack_score.remove(pod_key, old_top, old_prefix, old_len)
                    except Exception:
                        pass
                _prom.last_attack_score.labels(
                    pod=pod_key, top_syscalls=top_str, seq_prefix=seq_prefix, seq_len=seq_len_str
                ).set(ens)
                self._last_attack_labels[pod_key] = (top_str, seq_prefix, seq_len_str)

                # ── Argument context (paths + execs) ─────────────────────────
                _aw = args_window or []
                paths_list = list(dict.fromkeys(a["path"] for a in _aw if "path" in a))[:8]
                execs_list = list(dict.fromkeys(a["exec"] for a in _aw if "exec" in a))[:4]
                paths_str = " | ".join(paths_list) if paths_list else "(none)"
                execs_str = " | ".join(execs_list) if execs_list else "(none)"

                if pod_key in self._last_args_labels:
                    try:
                        _prom.last_attack_args.remove(pod_key, *self._last_args_labels[pod_key])
                    except Exception:
                        pass
                _prom.last_attack_args.labels(pod=pod_key, paths=paths_str, execs=execs_str).set(ens)
                self._last_args_labels[pod_key] = (paths_str, execs_str)

                # ── Combined detail metric (all triage info in one row) ───────
                sev_label = _severity(ens, self._threshold).strip()
                detail_key = (sev_label, top_str, seq_prefix, seq_len_str, paths_str, execs_str)
                if pod_key in self._last_detail_labels:
                    try:
                        _prom.last_attack_detail.remove(pod_key, *self._last_detail_labels[pod_key])
                    except Exception:
                        pass
                _prom.last_attack_detail.labels(
                    pod=pod_key, severity=sev_label,
                    top_syscalls=top_str, seq_prefix=seq_prefix, seq_len=seq_len_str,
                    paths=paths_str, execs=execs_str,
                ).set(ens)
                self._last_detail_labels[pod_key] = detail_key

                # ── Full feature vector that the model just flagged ───────────
                # Emit one series per dimension (scaled value the scorer saw).
                # Cardinality = feat_dim per pod — bounded; old labelsets are
                # cleared each time a new attack lands for that pod.
                feat = result.get("feature_vector")
                if feat and self._feature_dim_names:
                    for old in self._last_feature_labels.get(pod_key, []):
                        try:
                            _prom.last_attack_feature.remove(pod_key, *old)
                        except Exception:
                            pass
                    new_labels: list[tuple[str, str, str]] = []
                    for (group, idx, name), val in zip(self._feature_dim_names, feat):
                        idx_s = str(idx)
                        # Round to 3 dp at emission — the Grafana table renders
                        # arrays as JSON and has no per-element decimal control,
                        # so we trim the float noise here.
                        _prom.last_attack_feature.labels(
                            pod=pod_key, group=group, idx=idx_s, name=name
                        ).set(round(float(val), 3))
                        new_labels.append((group, idx_s, name))
                    self._last_feature_labels[pod_key] = new_labels

                    # ── Rolling history (last N attacks per pod) ─────────────
                    # Pack the whole attack into ONE series: the value =
                    # ensemble score; six labels carry the per-group
                    # "dim=val, dim=val" summaries covering ALL dims in that
                    # group, sorted by |scaled value| desc so the strongest
                    # contributors appear first. Grafana renders this as a flat
                    # table — no joins needed.
                    from collections import defaultdict
                    per_group: dict[str, list[tuple[str, float]]] = defaultdict(list)
                    for (group, _idx, name), val in zip(self._feature_dim_names, feat):
                        per_group[group].append((name, round(float(val), 3)))
                    summaries = {g: "—" for g in
                                 ('freq', 'disc', 'stats', 'bigrams', 'cat', 'prefixspan')}
                    for g, pairs in per_group.items():
                        if g not in summaries or not pairs:
                            continue
                        # Collapse the _x32 ABI suffix into the base kernel name,
                        # so e.g. `recvfrom`/`recvfrom_x32` (two distinct model
                        # dims) display as one entry, and bigrams like
                        # `ioctl_x32→ioctl_x32` collapse to `ioctl→ioctl`. The
                        # retained variant is the one with the larger |scaled
                        # value| so the cell order still reflects contribution.
                        merged: dict[str, float] = {}
                        for name, v in pairs:
                            base = name.replace('_x32', '')
                            prev = merged.get(base)
                            if prev is None or abs(v) > abs(prev):
                                merged[base] = v
                        flat = sorted(merged.items(),
                                      key=lambda nv: abs(nv[1]), reverse=True)
                        summaries[g] = ", ".join(n for n, _ in flat)
                    attack_id = time.strftime(
                        '%H:%M:%S', time.localtime(time.time()))
                    label_args = dict(pod=pod_key, attack_id=attack_id, **summaries)
                    _prom.attack_history.labels(**label_args).set(round(float(ens), 4))

                    # Evict the oldest entry if the buffer is full.
                    dq = self._attack_history.setdefault(
                        pod_key, collections.deque(maxlen=self._history_max + 1))
                    dq.append(label_args)
                    while len(dq) > self._history_max:
                        old = dq.popleft()
                        try:
                            _prom.attack_history.remove(
                                old['pod'], old['attack_id'],
                                old['freq'], old['disc'], old['stats'],
                                old['bigrams'], old['cat'], old['prefixspan'])
                        except Exception:
                            pass

        # Update rolling rate
        if pod_key not in self._recent:
            self._recent[pod_key] = collections.deque(maxlen=self._RATE_WINDOW)
        self._recent[pod_key].append(int(is_attack))
        if _prom is not None:
            _prom.attack_rate_pct.labels(pod=pod_key).set(self._rolling_rate(pod_key))

        # ── Decide whether to print ───────────────────────────────────────────
        now = time.monotonic()
        last = self._last_print.get(pod_key, 0.0)

        if confirmed:
            first = self._first_attack[pod_key]
            throttle = 0.0 if first else self._ATTACK_THROTTLE_S
        elif self._scores_only:
            throttle = self._NORMAL_THROTTLE_S
        else:
            return confirmed  # quiet mode: only print confirmed attacks immediately

        if now - last < throttle:
            return confirmed

        # ── Format and print ─────────────────────────────────────────────────
        nr_map   = _get_nr_syscall()
        seq_str  = " → ".join(nr_map.get(k, str(k)) for k in window)
        top      = collections.Counter(window).most_common(3)
        top_str  = " ".join(f"{nr_map.get(k, str(k))}×{v}" for k, v in top)
        rate     = self._rolling_rate(pod_key)
        lat      = result.get("latency_ms", 0)
        # Build human-readable args summary for the log line
        _aw = args_window or []
        _paths = list(dict.fromkeys(a["path"] for a in _aw if "path" in a))[:4]
        _execs = list(dict.fromkeys(a["exec"] for a in _aw if "exec" in a))[:3]
        args_str = ""
        if _execs:
            args_str += " execs[" + " ".join(_execs) + "]"
        if _paths:
            args_str += " paths[" + " | ".join(_paths) + "]"

        scores = (
            f"ensemble={ens:+.3f} "
            f"vae_err={result['vae_raw']:6.3f} "
            f"iforest={result['if_raw']:+.3f}"
        )

        if confirmed:
            sev = _severity(ens, self._threshold)
            tag = f"ATTACK {sev}"
            detail = f"consecutive={consec:<3d} attack_rate={rate:3d}%"
            self._first_attack[pod_key] = False
        else:
            tag = "normal      "
            detail = f"             attack_rate={rate:3d}%"

        print(
            f"[{elapsed:5d}s] {tag} | {pod_key:<32} | "
            f"{scores} | {detail} | latency={lat:3.0f}ms | top[{top_str}]{args_str} | seq[{seq_str}]",
            flush=True,
        )
        self._last_print[pod_key] = now
        return confirmed
