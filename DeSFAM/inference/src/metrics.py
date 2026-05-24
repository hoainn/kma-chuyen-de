"""Prometheus metrics for DeSFAM inference — call start() once from detect.py."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Per-pod gauges (labelled by pod name)
ensemble_score  = Gauge('desfam_ensemble_score',      'Ensemble anomaly score',         ['pod'])
vae_score       = Gauge('desfam_vae_score',           'VAE reconstruction error',       ['pod'])
iforest_score   = Gauge('desfam_iforest_score',       'Isolation Forest anomaly score', ['pod'])
attack_rate_pct = Gauge('desfam_attack_rate_pct',     'Rolling attack rate 0-100',      ['pod'])
consecutive     = Gauge('desfam_consecutive_attacks', 'Consecutive attack windows',     ['pod'])

# Counters
windows_total   = Counter('desfam_windows_total',      'Total windows scored',             ['pod'])
attacks_total   = Counter('desfam_attacks_total',      'Total confirmed attacks',          ['pod'])
events_total    = Counter('desfam_events_total',       'Total syscall events ingested')
pod_events_total = Counter('desfam_pod_events_total',  'Syscall events seen per pod',      ['pod'])

# Scoring latency histogram (milliseconds)
latency_hist = Histogram(
    'desfam_inference_latency_ms',
    'Scoring latency in milliseconds',
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500],
)

# Last confirmed attack trace per pod — labels carry syscall summary
# Only the most recent attack per pod is kept (old label-set removed on update)
last_attack_score = Gauge(
    'desfam_last_attack_score',
    'Ensemble score of last confirmed attack trace',
    ['pod', 'top_syscalls', 'seq_prefix', 'seq_len'],
)

# Syscall argument context for the last confirmed attack — human-readable paths/execs
last_attack_args = Gauge(
    'desfam_last_attack_args',
    'Argument context (file paths, binaries) of last confirmed attack trace',
    ['pod', 'paths', 'execs'],
)

# Combined full-detail metric: one row per pod, all triage info in labels
last_attack_detail = Gauge(
    'desfam_last_attack_detail',
    'Full triage context of last confirmed attack: severity, syscall trace, files, binaries',
    ['pod', 'severity', 'top_syscalls', 'seq_prefix', 'seq_len', 'paths', 'execs'],
)

# Active threshold value
threshold_gauge = Gauge('desfam_threshold', 'Active detection threshold')


def start(port: int = 8000) -> None:
    start_http_server(port)
    print(f"[metrics] Prometheus endpoint: http://0.0.0.0:{port}/metrics", flush=True)
