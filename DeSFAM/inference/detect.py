"""
DeSFAM live detector — thin orchestrator.

Connects to Tetragon gRPC, accumulates per-PID syscall traces, scores each
completed trace with the IF+VAE ensemble, and fires alerts when the ensemble
score exceeds the threshold.

Each "sequence" matches the training data format: one process execution trace,
bounded by exit_group or max_len (for long-running processes).

Usage:
  python detect.py \\
      --tetragon-addr 10.200.118.136:30321 \\
      --artifacts-dir /artifacts \\
      --namespace demo \\
      --kernel-ver 6.17 \\
      --threshold 0.55 \\
      --min-seq-len 8 \\
      --max-seq-len 5000 \\
      --max-idle-secs 30 \\
      --scores-only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import grpc

sys.path.insert(0, os.path.dirname(__file__))

import src.loader as loader
from src.extractor import Extractor
from src.windower import Windower
from src.featurizer import Featurizer
from src.scorer import Scorer
from src.alerter import Alerter
from src import metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeSFAM Tetragon gRPC syscall anomaly detector")
    p.add_argument("--tetragon-addr",  default=os.environ.get("TETRAGON_ADDR",  "10.200.118.136:30321"))
    p.add_argument("--artifacts-dir",  default=os.environ.get("ARTIFACTS_DIR",  "/artifacts"))
    p.add_argument("--namespace",      default=os.environ.get("TARGET_NAMESPACE", "demo"))
    p.add_argument("--kernel-ver",     default=os.environ.get("KERNEL_VER",      "6.17"))
    p.add_argument("--threshold",      type=float, default=float(os.environ.get("THRESHOLD", "0")))
    p.add_argument("--window-len",     type=int,   default=int(os.environ.get("WINDOW_LEN",    "15")))
    p.add_argument("--window-stride",  type=int,   default=int(os.environ.get("WINDOW_STRIDE", "3")))
    p.add_argument("--max-idle-secs",  type=float, default=float(os.environ.get("MAX_IDLE_SECS", "30")))
    p.add_argument("--flush-interval", type=float, default=float(os.environ.get("FLUSH_INTERVAL", "5")))
    p.add_argument("--alert-consecutive", type=int, default=int(os.environ.get("ALERT_CONSECUTIVE", "1")))
    p.add_argument("--calibrate-secs", type=float, default=float(os.environ.get("CALIBRATE_SECS", "0")))
    p.add_argument("--scores-only",    action="store_true", default=os.environ.get("SCORES_ONLY", "") == "1")
    p.add_argument("--log-features",   default=None, help="Path to write feature vectors as Parquet (optional)")
    # Online EMA threshold (paper §IV.B.3 Eq. 3, Algorithm 2 line 16):
    #   T_{t+1} = max(T_min, β·T_t + (1-β)·A(W_i))  ONLY on non-flagged windows
    # Opt-in via --ema or EMA_ENABLED=1. Defaults match the paper (β=0.9, T_min=0.5),
    # but T_min=0.5 was calibrated to a VAE-recon-error score scale; for our
    # RobustScaled ensemble the trained T_0 is usually a more sensible floor —
    # override with --ema-tmin / EMA_TMIN if the live scores drift below it.
    p.add_argument("--ema",            action="store_true", default=os.environ.get("EMA_ENABLED", "") == "1")
    p.add_argument("--ema-beta",       type=float, default=float(os.environ.get("EMA_BETA", "0.9")))
    p.add_argument("--ema-tmin",       type=float, default=float(os.environ.get("EMA_TMIN", "0.5")))
    p.add_argument("--verbose",        action="store_true")
    return p.parse_args()


def _load_id_to_name() -> dict[int, str]:
    """Read the kernel `syscall_64.tbl` bundled with the inference image (same
    table the model was trained against). Falls back to the hand-coded
    `SYSCALL_NR` from `syscalls_x86_64.py` if the file is missing. x32-ABI
    entries get a `_x32` suffix so they don't collide with the regular names
    (e.g. id 54 → `setsockopt`, id 541 → `setsockopt_x32`)."""
    tbl_path = os.path.join(os.path.dirname(__file__), 'syscall_64.tbl')
    id_to_name: dict[int, str] = {}
    if os.path.exists(tbl_path):
        with open(tbl_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 3 or not parts[0].isdigit():
                    continue
                num, abi, name = int(parts[0]), parts[1], parts[2]
                id_to_name[num] = f"{name}_x32" if abi == 'x32' else name
    if not id_to_name:
        from syscalls_x86_64 import SYSCALL_NR
        id_to_name = {v: k for k, v in SYSCALL_NR.items()}
    return id_to_name


def build_feature_dim_names(artifacts: dict) -> list[tuple[str, int, str]]:
    """Per-dimension (group, idx_in_group, semantic_name) for the loaded model's
    feature vector. Order matches train.py::build_features exactly so each tuple
    corresponds to one column of the scaled feature row the scorer sees.

    Syscall IDs are resolved to their kernel name via the bundled
    `syscall_64.tbl` — same table training used. Unknown IDs (shouldn't happen
    unless the vocab artefact is mismatched) keep the numeric fallback."""
    _nr_to_name = _load_id_to_name()
    def _name(sid: int) -> str:
        return _nr_to_name.get(int(sid), f"syscall_{sid}")

    names: list[tuple[str, int, str]] = []
    for i, sid in enumerate(artifacts.get("top_ids", []) or []):
        names.append(("freq", i, _name(sid)))
    for i, sid in enumerate(artifacts.get("disc_ids", []) or []):
        names.append(("disc", i, _name(sid)))
    if artifacts.get("enable_stats"):
        for i, n in enumerate(["entropy", "unique_count", "log_length",
                               "max_freq", "p75_nonzero", "std",
                               "coverage", "length_norm"]):
            names.append(("stats", i, n))
    for i, g in enumerate(artifacts.get("top_bigrams", []) or []):
        a, b = (g if len(g) == 2 else (g[0], g[1]))
        names.append(("bigrams", i, f"{_name(a)}→{_name(b)}"))
    for i, c in enumerate(artifacts.get("cat_cols", []) or []):
        names.append(("cat", i, c))
    if artifacts.get("has_temporal") and artifacts.get("temporal_dims", 0):
        for i, n in enumerate(["dt_mean", "dt_std", "dt_max"]):
            names.append(("temporal", i, n))
    if artifacts.get("ps_dims", 0):
        for i, n in enumerate(["matched_flag", "longest_match_norm"]):
            names.append(("prefixspan", i, n))
    return names


class EmaThreshold:
    """Paper §IV.B.3 Eq. 3 + Algorithm 2 line 16. Conditional update: T moves
    only when the window is NOT flagged, so attack scores cannot drag T upward
    and reduce sensitivity for what follows."""

    def __init__(self, t0: float, beta: float, t_min: float):
        self.t = float(t0)
        self.beta = float(beta)
        self.t_min = float(t_min)

    def update(self, score: float, is_attack: bool) -> float:
        if not is_attack:
            self.t = max(self.t_min, self.beta * self.t + (1 - self.beta) * float(score))
        return self.t


def _score_sequence(
    pod_key: str,
    seq: list[int],
    featurizer: Featurizer,
    scorer: Scorer,
    alerter: Alerter,
    kernel_ver: str,
    t_start: float,
    args_window: list[dict] | None = None,
) -> tuple[bool, dict]:
    t0 = time.perf_counter()
    feat = featurizer.transform(seq, kernel_ver)
    result = scorer.score(feat)
    result["latency_ms"] = (time.perf_counter() - t0) * 1000
    result["seq_len"] = len(seq)
    # The scaled vector the model just decided on — alerter exports it as
    # `desfam_last_attack_feature` when the window is flagged.
    result["feature_vector"] = feat[0].tolist()

    elapsed = int(time.time() - t_start)
    confirmed = alerter.process(pod_key, result, seq, elapsed, args_window)
    return confirmed, result


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("detector")

    log.info(f"Loading artifacts from: {args.artifacts_dir}")
    artifacts = loader.load(args.artifacts_dir, log)

    # THRESHOLD=0 means "use the trained T_0 from results.json"; any other value
    # (including negative — ensemble scores can go below zero) is an explicit
    # override.
    threshold = artifacts["threshold"] if args.threshold == 0 else args.threshold
    log.info(f"Effective threshold: {threshold:.4f}")

    metrics.start(8000)
    metrics.threshold_gauge.set(threshold)

    # Online EMA threshold (paper §IV.B.3 Eq. 3). Optional; opt-in via --ema.
    ema = EmaThreshold(threshold, args.ema_beta, args.ema_tmin) if args.ema else None
    if ema:
        log.info(f"EMA threshold ON: T_0={ema.t:.4f}  β={ema.beta}  T_min={ema.t_min}  "
                 f"(updates only on non-flagged windows)")
        if ema.t_min > ema.t:
            log.warning(f"EMA_TMIN ({ema.t_min}) > T_0 ({ema.t}): floor will dominate; "
                        "consider lowering EMA_TMIN to the trained score scale.")
        # Publish the static EMA parameters once; the per-step inputs are
        # updated below alongside each scorer.set_threshold(...) call.
        metrics.ema_beta.set(ema.beta)
        metrics.ema_t_min.set(ema.t_min)
        metrics.ema_t_prev.set(ema.t)        # T_t before the first update
        metrics.ema_last_score.set(float('nan'))
    # Surface loaded-model metadata + per-component T_0 for the Grafana panels.
    metrics.model_info.labels(
        feature_mode    = str(artifacts.get("feature_mode", "")),
        total_dims      = str(artifacts.get("input_dim", "")),
        has_temporal    = str(artifacts.get("has_temporal", False)),
        prefixspan_dims = str(artifacts.get("ps_dims", 0)),
        n_benign_patterns = str(sum(len(v) for v in artifacts.get("patterns_by_first", {}).values())),
        experiment      = str(artifacts.get("experiment", "")),
    ).set(1)
    for _name, _t in (artifacts.get("component_thresholds") or {}).items():
        metrics.component_threshold.labels(model=_name).set(_t)

    extractor  = Extractor(args.tetragon_addr, args.namespace, log)
    windower   = Windower(
        window_len   = args.window_len,
        window_stride= args.window_stride,
        max_idle_secs= args.max_idle_secs,
    )
    featurizer = Featurizer(
        cat_cols          = artifacts["cat_cols"],
        id_to_cat         = artifacts["id_to_cat"],
        patterns_by_first = artifacts["patterns_by_first"],
        ps_dims           = artifacts["ps_dims"],
        input_dim         = artifacts["input_dim"],
        has_temporal      = artifacts["has_temporal"],
        temporal_dims     = artifacts["temporal_dims"],
        top_ids           = artifacts.get("top_ids", []),
        disc_ids          = artifacts.get("disc_ids", []),
        top_bigrams       = artifacts.get("top_bigrams", []),
        enable_stats      = artifacts.get("enable_stats", False),
        feature_scaler    = artifacts["feature_scaler"],
        sensor_alphabet   = artifacts.get("sensor_alphabet"),
        log_path          = args.log_features,
    )
    scorer = Scorer(
        iforest     = artifacts["iforest"],
        encoder     = artifacts["encoder"],
        decoder     = artifacts["decoder"],
        threshold   = threshold,
        ensemble    = artifacts["ensemble"],
        legacy_norm = artifacts["legacy_norm"],
    )
    alerter = Alerter(
        threshold      = threshold,
        consecutive    = args.alert_consecutive,
        scores_only    = args.scores_only,
        calibrate_secs = args.calibrate_secs,
    )
    # Hand the alerter the per-dim names so it can label
    # `desfam_last_attack_feature` when an attack fires.
    alerter.set_feature_dim_names(build_feature_dim_names(artifacts))

    log.info(
        f"Ready — window_len={args.window_len} stride={args.window_stride} "
        f"max_idle={args.max_idle_secs}s flush_every={args.flush_interval}s "
        f"threshold={threshold:.4f} kernel={args.kernel_ver}"
    )

    event_cnt = score_cnt = alert_cnt = 0
    t_start = time.time()
    t_last_flush = t_start

    try:
        for pod_key, pid, nr, ts, ev_args in extractor.stream():
            event_cnt += 1
            metrics.events_total.inc()
            metrics.pod_events_total.labels(pod=pod_key).inc()

            result = windower.push(pod_key, pid, nr, ts, ev_args)
            if result is not None:
                seq_pod_key, seq, seq_args = result
                score_cnt += 1
                confirmed, score_result = _score_sequence(
                    seq_pod_key, seq, featurizer, scorer, alerter,
                    args.kernel_ver, t_start, seq_args,
                )
                if confirmed:
                    alert_cnt += 1
                if ema is not None:
                    # Publish the inputs to this EMA step so the dashboard can
                    # walk through the arithmetic.
                    metrics.ema_t_prev.set(ema.t)
                    metrics.ema_last_score.set(float(score_result["ensemble"]))
                    new_t = ema.update(score_result["ensemble"], score_result["is_attack"])
                    if not score_result["is_attack"]:
                        metrics.ema_updates_total.inc()
                    scorer.set_threshold(new_t)
                    alerter.set_threshold(new_t)
                    metrics.threshold_gauge.set(new_t)

            # Flush idle per-PID traces on a timer
            now = time.time()
            if now - t_last_flush >= args.flush_interval:
                t_last_flush = now
                for seq_pod_key, seq, seq_args in windower.flush_idle(now):
                    score_cnt += 1
                    confirmed, score_result = _score_sequence(
                        seq_pod_key, seq, featurizer, scorer, alerter,
                        args.kernel_ver, t_start, seq_args,
                    )
                    if confirmed:
                        alert_cnt += 1
                    if ema is not None:
                        metrics.ema_t_prev.set(ema.t)
                        metrics.ema_last_score.set(float(score_result["ensemble"]))
                        new_t = ema.update(score_result["ensemble"], score_result["is_attack"])
                        if not score_result["is_attack"]:
                            metrics.ema_updates_total.inc()
                        scorer.set_threshold(new_t)
                        alerter.set_threshold(new_t)
                        metrics.threshold_gauge.set(new_t)

            if event_cnt % 10_000 == 0:
                log.info(
                    f"events={event_cnt} scored={score_cnt} alerts={alert_cnt} "
                    f"active_pids={windower.active_pids} "
                    f"uptime={int(now - t_start)}s"
                )

    except grpc.RpcError as e:
        log.error(f"gRPC error: {e.code()} — {e.details()}")
        sys.exit(1)
    except KeyboardInterrupt:
        pass

    elapsed = int(time.time() - t_start)
    log.info(f"Done. events={event_cnt} scored={score_cnt} alerts={alert_cnt} uptime={elapsed}s")


if __name__ == "__main__":
    main()
