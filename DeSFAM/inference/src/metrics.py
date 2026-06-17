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

# Active ensemble threshold (the one used for the flag/no-flag decision).
threshold_gauge = Gauge('desfam_threshold', 'Active detection threshold')

# Per-component T_0 values (paper §IV.B.3 "initial threshold"). Useful with
# THRESHOLD_STRATEGY=p995 so operators can compare a live VAE/IF score against
# its trained T_0 instead of only the fused ensemble cut-off.
component_threshold = Gauge(
    'desfam_component_threshold',
    'Training-side T_0 per component',
    ['model'],          # 'vae' | 'iforest' | 'ensemble'
)

# Step-by-step EMA inputs surfaced for the "live calculator" dashboard panel,
# so operators can see exactly how T_{t+1} = max(T_min, β·T_t + (1-β)·A(W_i))
# is computed at every update (paper §IV.B.3 Eq. 3 / Algorithm 2 line 16).
ema_t_prev      = Gauge('desfam_ema_t_prev',      'T_t — active threshold before the latest EMA update')
ema_last_score  = Gauge('desfam_ema_last_score',  'A(W_i) — ensemble score of the most recent NON-flagged window (the only input that updates T)')
ema_beta        = Gauge('desfam_ema_beta',        'β — EMA decay parameter (env EMA_BETA)')
ema_t_min       = Gauge('desfam_ema_t_min',       'T_min — EMA floor (env EMA_TMIN)')
ema_updates_total = Counter('desfam_ema_updates_total', 'Number of EMA updates applied since start')

# Loaded-model metadata — value=1, info in labels. Drives the "Model Info" stat
# in Grafana so it is visible at a glance which variant is live (paper_only vs
# paper_plus, the feature dim, whether temporal is active, the pattern count).
model_info = Gauge(
    'desfam_model_info',
    'Loaded model metadata',
    ['feature_mode', 'total_dims', 'has_temporal',
     'prefixspan_dims', 'n_benign_patterns', 'experiment'],
)

# Per-dimension feature value of the most recent confirmed-attack window
# (scaled feature, i.e. exactly what the model saw and decided was anomalous).
# Cardinality is bounded by the feature dim: 140 for paper_plus, 12 for
# paper_only. Old label sets are wiped on every new attack so the gauge only
# ever exposes the latest vector.
last_attack_feature = Gauge(
    'desfam_last_attack_feature',
    'Scaled feature value of the last confirmed-attack window (per dimension)',
    ['pod', 'group', 'idx', 'name'],
)

# Rolling history (last N attacks per pod). ONE series per attack — the
# ensemble score lives in the metric value, the six per-group summaries live in
# labels. Grafana can then display this as a flat table with no joins:
#   attack_id | freq | disc | stats | bigrams | cat | prefixspan | Score
# Cardinality per pod ≤ N (default N=20). Each summary label carries
# "dim=val, dim=val, …" for that group's dims with |scaled| ≥ 0.5.
attack_history = Gauge(
    'desfam_attack_history',
    'Recent attack-flagged windows (rolling history of N attacks per pod). '
    'Value = ensemble anomaly score. Labels carry per-feature-group dim:value '
    'summaries.',
    ['pod', 'attack_id', 'freq', 'disc', 'stats', 'bigrams', 'cat', 'prefixspan'],
)


def start(port: int = 8000) -> None:
    start_http_server(port)
    print(f"[metrics] Prometheus endpoint: http://0.0.0.0:{port}/metrics", flush=True)
