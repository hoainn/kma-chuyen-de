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
    p.add_argument("--verbose",        action="store_true")
    return p.parse_args()


def _score_sequence(
    pod_key: str,
    seq: list[int],
    featurizer: Featurizer,
    scorer: Scorer,
    alerter: Alerter,
    kernel_ver: str,
    t_start: float,
    args_window: list[dict] | None = None,
) -> bool:
    t0 = time.perf_counter()
    feat = featurizer.transform(seq, kernel_ver)
    result = scorer.score(feat)
    result["latency_ms"] = (time.perf_counter() - t0) * 1000
    result["seq_len"] = len(seq)

    elapsed = int(time.time() - t_start)
    return alerter.process(pod_key, result, seq, elapsed, args_window)


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

    threshold = args.threshold if args.threshold > 0 else artifacts["threshold"]
    log.info(f"Effective threshold: {threshold:.4f}")

    metrics.start(8000)
    metrics.threshold_gauge.set(threshold)

    extractor  = Extractor(args.tetragon_addr, args.namespace, log)
    windower   = Windower(
        window_len   = args.window_len,
        window_stride= args.window_stride,
        max_idle_secs= args.max_idle_secs,
    )
    featurizer = Featurizer(
        top_ids       = artifacts["top_ids"],
        disc_ids      = artifacts["disc_ids"],
        top_ngrams    = artifacts["top_ngrams"],
        ngram_n       = artifacts["ngram_n"],
        ver_cols      = artifacts["ver_cols"],
        cat_cols      = artifacts.get("cat_cols"),
        id_to_cat     = artifacts.get("id_to_cat"),
        input_dim     = artifacts["input_dim"],
        feature_scaler= artifacts["feature_scaler"],
        log_path      = args.log_features,
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
                confirmed = _score_sequence(
                    seq_pod_key, seq, featurizer, scorer, alerter,
                    args.kernel_ver, t_start, seq_args,
                )
                if confirmed:
                    alert_cnt += 1

            # Flush idle per-PID traces on a timer
            now = time.time()
            if now - t_last_flush >= args.flush_interval:
                t_last_flush = now
                for seq_pod_key, seq, seq_args in windower.flush_idle(now):
                    score_cnt += 1
                    confirmed = _score_sequence(
                        seq_pod_key, seq, featurizer, scorer, alerter,
                        args.kernel_ver, t_start, seq_args,
                    )
                    if confirmed:
                        alert_cnt += 1

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
