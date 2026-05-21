"""
DeSFAM Real-Time Syscall Anomaly Detector — Tetragon gRPC Client (v2)

Streams events directly from Tetragon's FineGuidanceSensors.GetEvents gRPC API
and scores each pod's rolling syscall window using the model variant
selected at startup.

Supported variants (--model-variant):
    ensemble        v1 α=0.7·VAE + 0.3·IF (paper-style; default for compatibility)
    vae_dongting    v1 VAE alone, trained on DongTing
    vae_container   v2 VAE retrained on container baseline + attack data
    lstm            v2 supervised LSTM
    cnn1d           v2 supervised 1-D CNN (best per evaluation_v2.json)

Usage:
    python detect_tetra_grpc.py \
        --tetragon-addr tetragon-grpc.kube-system.svc:54321 \
        --namespace dev \
        --model-variant cnn1d \
        --window 200 --step 50 --kernel-ver 6.12
"""
import argparse
import collections
import json
import logging
import os
import sys
import time

import grpc
import joblib
import numpy as np
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tetragon_grpc'))
from tetragon import events_pb2, sensors_pb2_grpc  # noqa: E402

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='DeSFAM Tetragon gRPC detector (v2)')
parser.add_argument('--tetragon-addr', default='tetragon-grpc.kube-system.svc.cluster.local:54321')
parser.add_argument('--namespace', default='', help='K8s namespace filter (empty=all)')
parser.add_argument('--window', type=int, default=200)
parser.add_argument('--step', type=int, default=50)
parser.add_argument('--kernel-ver', default='6.12')
parser.add_argument('--threshold', type=float, default=None,
                    help='Override decision threshold; default loaded from model params')
parser.add_argument('--alpha', type=float, default=None,
                    help='Override α for ensemble variant only')
parser.add_argument('--model-variant', default='ensemble',
                    choices=['ensemble', 'vae_dongting', 'vae_container', 'lstm', 'cnn1d'])
parser.add_argument('--model-dir', default='/workspace/outputs')
parser.add_argument('--syscall-tbl', default='/workspace/data/syscall_64.tbl')
parser.add_argument('--scores-only', action='store_true')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('detect')
OUT = args.model_dir

# ── Syscall table ─────────────────────────────────────────────────────────────
log.info('Loading syscall table...')
syscall_name_to_id = {}
syscall_id_to_name = {}
with open(args.syscall_tbl) as f:
    for line in f:
        parts = line.strip().split()
        if parts and parts[0].isdigit():
            sid, sname = int(parts[0]), parts[2]
            syscall_name_to_id[sname] = sid
            syscall_id_to_name[sid] = sname
log.info(f'  {len(syscall_name_to_id)} syscalls in table')

# ── FE config + scaler ────────────────────────────────────────────────────────
log.info('Loading feature engineering config...')
with open(os.path.join(OUT, 'fe_report.json')) as f:
    fe_cfg = json.load(f)
TOP_IDS = list(fe_cfg['top_ids'])
DISC_IDS = list(fe_cfg['disc_ids'])
TOP_BIGRAMS = [tuple(b) for b in fe_cfg['top_bigrams']]
VER_COLS = fe_cfg['ver_cols']
log.info(f'  freq_60={len(TOP_IDS)} disc_40={len(DISC_IDS)} '
         f'bigrams={len(TOP_BIGRAMS)} ver={len(VER_COLS)}')

log.info('Loading StandardScaler...')
scaler = joblib.load(os.path.join(OUT, 'scaler_fe.pkl'))

log.info('Loading v1 model params...')
with open(os.path.join(OUT, 'model_params.json')) as f:
    mp_v1 = json.load(f)
INPUT_DIM = mp_v1['input_dim']
LATENT = mp_v1['latent_dim']


class _RobustNorm:
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def __call__(self, x):
        return np.clip((x - self.lo) / (self.hi - self.lo + 1e-9), 0.0, 1.0)


# ── TensorFlow / Keras model builders ─────────────────────────────────────────
log.info('Loading TensorFlow...')
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402


def build_vae_encoder():
    inp = keras.Input(shape=(INPUT_DIM,))
    x = layers.Dense(128, activation='selu')(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='selu')(x)
    x = layers.Dropout(0.2)(x)
    mu = layers.Dense(LATENT, name='mu')(x)
    lv = layers.Dense(LATENT, name='lv')(x)
    return keras.Model(inp, [mu, lv], name='encoder')


def build_vae_decoder():
    inp = keras.Input(shape=(LATENT,))
    x = layers.Dense(64, activation='selu')(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='selu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(INPUT_DIM)(x)
    return keras.Model(inp, out, name='decoder')


def build_lstm():
    inp = keras.Input(shape=(INPUT_DIM,))
    x = layers.Reshape((INPUT_DIM, 1))(inp)
    x = layers.LSTM(64)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    return keras.Model(inp, layers.Dense(1, activation='sigmoid')(x), name='lstm')


def build_cnn1d():
    inp = keras.Input(shape=(INPUT_DIM,))
    x = layers.Reshape((INPUT_DIM, 1))(inp)
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    return keras.Model(inp, layers.Dense(1, activation='sigmoid')(x), name='cnn1d')


def vae_recon_error(enc, dec, x_np, n_samples=5):
    x = tf.constant(x_np, dtype=tf.float32)
    errs = []
    for _ in range(n_samples):
        mu, lv = enc(x, training=False)
        lv = tf.clip_by_value(lv, -8, 8)
        z = mu + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(mu))
        xh = dec(z, training=False)
        errs.append(tf.reduce_sum(tf.square(x - xh), axis=1).numpy())
    return np.nan_to_num(np.mean(errs, axis=0), nan=1e6, posinf=1e6, neginf=0.0)


# ── Variant initialisation ────────────────────────────────────────────────────
variant = args.model_variant
log.info(f'Initialising variant: {variant}')

iso = None
norm_if = norm_vae = None
vae_enc = vae_dec = None
sup_model = None
DEFAULT_THRESHOLD = 0.5
ALPHA = args.alpha if args.alpha is not None else mp_v1.get('alpha', 0.7)

if variant == 'ensemble':
    iso = joblib.load(os.path.join(OUT, 'if_model.joblib'))
    norm_if = _RobustNorm(mp_v1['if_lo'], mp_v1['if_hi'])
    norm_vae = _RobustNorm(mp_v1['vae_lo'], mp_v1['vae_hi'])
    vae_enc = build_vae_encoder()
    vae_dec = build_vae_decoder()
    vae_enc.load_weights(os.path.join(OUT, 'vae_encoder.weights.h5'))
    vae_dec.load_weights(os.path.join(OUT, 'vae_decoder.weights.h5'))
    DEFAULT_THRESHOLD = mp_v1['ens_threshold']
elif variant == 'vae_dongting':
    norm_vae = _RobustNorm(mp_v1['vae_lo'], mp_v1['vae_hi'])
    vae_enc = build_vae_encoder()
    vae_dec = build_vae_decoder()
    vae_enc.load_weights(os.path.join(OUT, 'vae_encoder.weights.h5'))
    vae_dec.load_weights(os.path.join(OUT, 'vae_decoder.weights.h5'))
    DEFAULT_THRESHOLD = mp_v1['vae_threshold']
elif variant == 'vae_container':
    with open(os.path.join(OUT, 'model_params_v2_container.json')) as f:
        mp_v2 = json.load(f)
    norm_vae = _RobustNorm(mp_v2['vae_lo'], mp_v2['vae_hi'])
    vae_enc = build_vae_encoder()
    vae_dec = build_vae_decoder()
    vae_enc.load_weights(os.path.join(OUT, 'vae_encoder_v2_container.weights.h5'))
    vae_dec.load_weights(os.path.join(OUT, 'vae_decoder_v2_container.weights.h5'))
    DEFAULT_THRESHOLD = mp_v2.get('vae_threshold', 0.5)
elif variant == 'lstm':
    sup_model = build_lstm()
    sup_model.load_weights(os.path.join(OUT, 'lstm_v2.weights.h5'))
    # Threshold from train_supervised_report.json
    with open(os.path.join(OUT, 'train_supervised_report.json')) as f:
        sr = json.load(f)
    for m in sr['models']:
        if m['model'] == 'lstm_v2':
            DEFAULT_THRESHOLD = m['threshold']
elif variant == 'cnn1d':
    sup_model = build_cnn1d()
    sup_model.load_weights(os.path.join(OUT, 'cnn1d_v2.weights.h5'))
    with open(os.path.join(OUT, 'train_supervised_report.json')) as f:
        sr = json.load(f)
    for m in sr['models']:
        if m['model'] == 'cnn1d_v2':
            DEFAULT_THRESHOLD = m['threshold']

# Warm up Keras models
for m in (vae_enc, vae_dec, sup_model):
    if m is None:
        continue
    if m.input_shape[1] == LATENT:
        m(tf.zeros((1, LATENT)), training=False)
    else:
        m(tf.zeros((1, INPUT_DIM)), training=False)

THRESHOLD = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD
log.info(f'  default threshold={DEFAULT_THRESHOLD:.4f}  active threshold={THRESHOLD:.4f}')


# ── Feature engineering (shared) ──────────────────────────────────────────────
def build_features(seq_ids, kernel_ver):
    if not seq_ids:
        return np.zeros(INPUT_DIM, dtype=np.float32)
    arr = np.array(seq_ids, dtype=np.int32)
    counts = collections.Counter(seq_ids)
    total = len(seq_ids)
    freq = np.array([counts.get(sid, 0) / (total + 1e-9) for sid in TOP_IDS],
                    dtype=np.float32)
    s = set(seq_ids)
    disc = np.array([1.0 if sid in s else 0.0 for sid in DISC_IDS], dtype=np.float32)
    unique = len(s)
    freq_pr = np.bincount(arr[arr >= 0], minlength=548).astype(np.float32)
    freq_pr /= (freq_pr.sum() + 1e-9)
    ent = float(scipy_entropy(freq_pr + 1e-9))
    stats = np.array([
        np.log1p(total), np.log1p(unique), unique / (total + 1e-9), ent,
        np.log1p(total), np.log1p(total * 20),
        float(np.mean(arr)), float(np.std(arr)),
    ], dtype=np.float32)
    if total >= 2:
        bg_cnt = collections.Counter(zip(seq_ids[:-1], seq_ids[1:]))
        bg_tot = sum(bg_cnt.values()) + 1e-9
        bigrams = np.array([bg_cnt.get(bg, 0) / bg_tot for bg in TOP_BIGRAMS],
                           dtype=np.float32)
    else:
        bigrams = np.zeros(len(TOP_BIGRAMS), dtype=np.float32)
    ver_short = '.'.join(kernel_ver.split('.')[:2])
    ver_vec = np.array([1.0 if v == ver_short else 0.0 for v in VER_COLS],
                       dtype=np.float32)
    return np.concatenate([freq, disc, stats, bigrams, ver_vec])


# ── Scoring dispatchers per variant ───────────────────────────────────────────
def _score_ensemble(feat_scaled):
    if_raw = float(-iso.decision_function(feat_scaled)[0])
    vae_raw = float(vae_recon_error(vae_enc, vae_dec, feat_scaled)[0])
    if_n = float(norm_if(np.array([if_raw]))[0])
    vae_n = float(norm_vae(np.array([vae_raw]))[0])
    ens = ALPHA * vae_n + (1 - ALPHA) * if_n
    return {'score': ens, 'if_norm': if_n, 'vae_norm': vae_n}


def _score_vae(feat_scaled):
    vae_raw = float(vae_recon_error(vae_enc, vae_dec, feat_scaled)[0])
    vae_n = float(norm_vae(np.array([vae_raw]))[0])
    return {'score': vae_n, 'vae_norm': vae_n}


def _score_supervised(feat_scaled):
    proba = float(sup_model.predict(feat_scaled, verbose=0)[0, 0])
    return {'score': proba}


SCORERS = {
    'ensemble': _score_ensemble,
    'vae_dongting': _score_vae,
    'vae_container': _score_vae,
    'lstm': _score_supervised,
    'cnn1d': _score_supervised,
}


def score_window(seq_ids, kernel_ver):
    feat = build_features(seq_ids, kernel_ver).reshape(1, -1)
    feat_scaled = scaler.transform(feat).astype(np.float32)
    r = SCORERS[variant](feat_scaled)
    r['is_attack'] = bool(r['score'] >= THRESHOLD)
    return r


# ── Tetragon event parser (unchanged) ─────────────────────────────────────────
SYSCALL_FN_PREFIXES = ('__x64_sys_', '__arm64_sys_', 'sys_', '__se_sys_')


def _strip_syscall_prefix(fn_name):
    for p in SYSCALL_FN_PREFIXES:
        if fn_name.startswith(p):
            return fn_name[len(p):]
    return fn_name


def _pod_key(process_pb, default_ns):
    pod = getattr(process_pb, 'pod', None)
    if pod and pod.namespace:
        ns = pod.namespace
        name = pod.name or str(getattr(process_pb.pid, 'value', process_pb.pid))
    else:
        ns = default_ns or 'host'
        name = str(getattr(process_pb.pid, 'value', process_pb.pid))
    return f'{ns}/{name}'


def parse_event_pb(response):
    if response.HasField('process_kprobe'):
        kp = response.process_kprobe
        fn = _strip_syscall_prefix(kp.function_name or '')
        sid = syscall_name_to_id.get(fn)
        if sid is None:
            return None, None
        return _pod_key(kp.process, ''), sid
    if response.HasField('process_tracepoint'):
        tp = response.process_tracepoint
        if tp.event != 'sys_enter' and tp.subsys != 'raw_syscalls':
            return None, None
        if not tp.args:
            return None, None
        arg0 = tp.args[0]
        sid = None
        if arg0.HasField('long_arg'):
            sid = int(arg0.long_arg)
        elif arg0.HasField('int_arg'):
            sid = int(arg0.int_arg)
        if sid is None or sid < 0 or sid not in syscall_id_to_name:
            return None, None
        return _pod_key(tp.process, ''), sid
    if response.HasField('process_exec'):
        pe = response.process_exec
        return _pod_key(pe.process, ''), syscall_name_to_id.get('execve', 59)
    if response.HasField('process_exit'):
        pe = response.process_exit
        return _pod_key(pe.process, ''), syscall_name_to_id.get('exit_group', 231)
    return None, None


# ── gRPC streaming loop ───────────────────────────────────────────────────────
def stream_events(addr, ns_filter):
    log.info(f'Connecting to Tetragon gRPC at {addr}...')
    channel = grpc.insecure_channel(
        addr,
        options=[
            ('grpc.keepalive_time_ms', 120_000),
            ('grpc.keepalive_timeout_ms', 20_000),
            ('grpc.max_receive_message_length', 16 * 1024 * 1024),
        ],
    )
    stub = sensors_pb2_grpc.FineGuidanceSensorsStub(channel)
    request = events_pb2.GetEventsRequest()
    if ns_filter:
        flt = request.allow_list.add()
        flt.namespace.append(ns_filter)
    log.info(f'Streaming events (namespace_filter={ns_filter or "<all>"})...')
    return stub.GetEvents(request)


def main():
    WINDOW = args.window
    STEP = args.step if args.step > 0 else WINDOW
    NS = args.namespace
    log.info(f'Ready. variant={variant!r}  Window={WINDOW} step={STEP} '
             f'threshold={THRESHOLD:.4f} kernel={args.kernel_ver}')

    windows = collections.defaultdict(collections.deque)
    step_cnts = collections.defaultdict(int)
    event_cnt = 0
    alert_cnt = 0
    t_start = time.time()

    try:
        for response in stream_events(args.tetragon_addr, NS):
            pod_key, sid = parse_event_pb(response)
            if pod_key is None:
                continue
            event_cnt += 1
            w = windows[pod_key]
            w.append(sid)
            while len(w) > WINDOW:
                w.popleft()
            step_cnts[pod_key] += 1
            if len(w) < WINDOW or step_cnts[pod_key] < STEP:
                continue
            step_cnts[pod_key] = 0

            t0 = time.perf_counter()
            res = score_window(list(w), args.kernel_ver)
            latency = (time.perf_counter() - t0) * 1000
            elapsed = int(time.time() - t_start)
            verdict = 'ATTACK' if res['is_attack'] else 'normal'
            if res['is_attack']:
                alert_cnt += 1

            if res['is_attack'] or args.scores_only:
                top = collections.Counter(w).most_common(5)
                top_str = ' '.join(f'{syscall_id_to_name.get(k, str(k))}×{v}' for k, v in top)
                # Extra fields when ensemble (preserve old format)
                extras = ''
                if variant == 'ensemble':
                    extras = f' vae={res["vae_norm"]:.3f} if={res["if_norm"]:.3f}'
                print(
                    f'[{elapsed:5d}s] {verdict:6s} | {pod_key:<45} | '
                    f'score={res["score"]:.3f}{extras} '
                    f'| lat={latency:.0f}ms | top: {top_str}',
                    flush=True,
                )
            if event_cnt % 10000 == 0:
                log.info(f'events={event_cnt} alerts={alert_cnt} '
                         f'pods={len(windows)} uptime={elapsed}s')
    except grpc.RpcError as e:
        log.error(f'gRPC error: {e.code()} — {e.details()}')
        sys.exit(1)
    except KeyboardInterrupt:
        pass

    elapsed = int(time.time() - t_start)
    log.info(f'Done. events={event_cnt} alerts={alert_cnt} '
             f'pods={len(windows)} uptime={elapsed}s')


if __name__ == '__main__':
    main()
