"""
DeSFAM Real-Time Syscall Anomaly Detector — Tetragon Integration

Reads Tetragon JSON events from stdin (piped from tetra getevents --output json)
and scores each pod's syscall sequence using the trained IF + VAE ensemble.

Usage (inside container, receiving tetra stream from host):
    tetra getevents \
        --server-address localhost:54321 \
        --namespace dev \
        --output json | \
    docker exec -i dongting_jupyter \
        python /workspace/detect_tetra.py \
        --namespace dev \
        --window 200 \
        --step 50 \
        --kernel-ver 5.15 \
        --verbose

Event format supported:
  process_kprobe:     function_name = "__x64_sys_read"
  process_tracepoint: event=sys_enter, args[0].long_arg = <syscall_nr>
"""
import sys, os, json, argparse, time, collections, logging
import numpy as np
from scipy.stats import entropy as scipy_entropy

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='DeSFAM Tetragon anomaly detector')
parser.add_argument('--namespace',  default='dev',    help='K8s namespace to monitor')
parser.add_argument('--window',     type=int, default=200, help='Syscall window size')
parser.add_argument('--step',       type=int, default=50,  help='Slide step (0=tumbling)')
parser.add_argument('--kernel-ver', default='5.15',   help='Node kernel version (major.minor)')
parser.add_argument('--threshold',  type=float, default=None,
                    help='Override ensemble threshold (0-1). Default: loaded from model_params.json')
parser.add_argument('--alpha',      type=float, default=None,
                    help='Override VAE weight alpha (0-1). Default: from model_params.json')
parser.add_argument('--scores-only',action='store_true',
                    help='Print all scores (not just alerts)')
parser.add_argument('--dry-run',   action='store_true',
                    help='Load models and parse events but skip scoring (latency test)')
parser.add_argument('--verbose',   action='store_true')
args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('detect')

OUT = os.path.join(os.path.dirname(__file__), 'outputs')
DATA = os.path.join(os.path.dirname(__file__), 'data', 'dongting')

# ── Load syscall table ────────────────────────────────────────────────────────
log.info('Loading syscall table...')
syscall_name_to_id = {}
syscall_id_to_name = {}
tbl_path = os.path.join(DATA, 'syscall_64.tbl')
with open(tbl_path) as f:
    for line in f:
        parts = line.strip().split()
        if parts and parts[0].isdigit():
            sid, sname = int(parts[0]), parts[2]
            syscall_name_to_id[sname] = sid
            syscall_id_to_name[sid]   = sname

log.info(f'  {len(syscall_name_to_id)} syscalls in table')

# ── Load FE config ────────────────────────────────────────────────────────────
log.info('Loading feature engineering config...')
with open(os.path.join(OUT, 'fe_report.json')) as f:
    fe_cfg = json.load(f)

TOP_IDS    = list(fe_cfg['top_ids'])       # 60 most frequent syscall IDs
DISC_IDS   = list(fe_cfg['disc_ids'])      # 40 discriminative IDs
TOP_BIGRAMS= [tuple(b) for b in fe_cfg['top_bigrams']]  # 40 bigram pairs
VER_COLS   = fe_cfg['ver_cols']            # 26 kernel version strings

log.info(f'  freq_60={len(TOP_IDS)} disc_40={len(DISC_IDS)} '
         f'bigrams={len(TOP_BIGRAMS)} ver={len(VER_COLS)}')

# ── Load scaler ───────────────────────────────────────────────────────────────
log.info('Loading StandardScaler...')
import joblib
scaler = joblib.load(os.path.join(OUT, 'scaler_fe.pkl'))

# ── Load model params ─────────────────────────────────────────────────────────
log.info('Loading model params...')
with open(os.path.join(OUT, 'model_params.json')) as f:
    mparams = json.load(f)

ALPHA     = args.alpha     if args.alpha     is not None else mparams['alpha']
THRESHOLD = args.threshold if args.threshold is not None else mparams['ens_threshold']
LATENT    = mparams['latent_dim']
INPUT_DIM = mparams['input_dim']

log.info(f'  alpha={ALPHA}  threshold={THRESHOLD:.4f}  latent={LATENT}  input={INPUT_DIM}')

# ── Load IF model ─────────────────────────────────────────────────────────────
log.info('Loading Isolation Forest...')
iso = joblib.load(os.path.join(OUT, 'if_model.joblib'))

class _Normalizer:
    def __init__(self, lo, hi): self.lo, self.hi = lo, hi
    def __call__(self, x):
        return np.clip((x - self.lo) / (self.hi - self.lo + 1e-9), 0.0, 1.0)

norm_if  = _Normalizer(mparams['if_lo'],  mparams['if_hi'])
norm_vae = _Normalizer(mparams['vae_lo'], mparams['vae_hi'])

# ── Load VAE ──────────────────────────────────────────────────────────────────
log.info('Loading VAE encoder/decoder...')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_encoder(input_dim, latent_dim):
    inp = keras.Input(shape=(input_dim,))
    x   = layers.Dense(128, activation='selu')(inp)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(64,  activation='selu')(x)
    x   = layers.Dropout(0.2)(x)
    mu  = layers.Dense(latent_dim, name='mu')(x)
    lv  = layers.Dense(latent_dim, name='lv')(x)
    return keras.Model(inp, [mu, lv], name='encoder')

def build_decoder(latent_dim, output_dim):
    inp = keras.Input(shape=(latent_dim,))
    x   = layers.Dense(64,  activation='selu')(inp)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(128, activation='selu')(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(output_dim)(x)
    return keras.Model(inp, out, name='decoder')

encoder = build_encoder(INPUT_DIM, LATENT)
decoder = build_decoder(LATENT, INPUT_DIM)
encoder.load_weights(os.path.join(OUT, 'vae_encoder.weights.h5'))
decoder.load_weights(os.path.join(OUT, 'vae_decoder.weights.h5'))
# Warm up
_dummy = tf.zeros((1, INPUT_DIM))
encoder(_dummy, training=False); decoder(tf.zeros((1, LATENT)), training=False)
log.info('  VAE loaded and warmed up')

def vae_recon_error(x_np, n_samples=5):
    x = tf.constant(x_np, dtype=tf.float32)
    errors = []
    for _ in range(n_samples):
        mu, lv = encoder(x, training=False)
        lv     = tf.clip_by_value(lv, -8, 8)
        z      = mu + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(mu))
        xh     = decoder(z, training=False)
        errors.append(tf.reduce_sum(tf.square(x - xh), axis=1).numpy())
    return np.nan_to_num(np.mean(errors, axis=0), nan=1e6, posinf=1e6, neginf=0.0)

# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(seq_ids, kernel_ver):
    """Build 174-dim feature vector from a window of syscall IDs."""
    if not seq_ids:
        return np.zeros(INPUT_DIM, dtype=np.float32)

    arr    = np.array(seq_ids, dtype=np.int32)
    counts = collections.Counter(seq_ids)
    total  = len(seq_ids)

    # freq_60
    freq = np.array([counts.get(sid, 0) / (total + 1e-9) for sid in TOP_IDS],
                    dtype=np.float32)

    # disc_40
    s    = set(seq_ids)
    disc = np.array([1.0 if sid in s else 0.0 for sid in DISC_IDS], dtype=np.float32)

    # stats_8 — approximate log_count & log_size from window length
    unique  = len(s)
    freq_pr = np.bincount(arr[arr >= 0], minlength=548).astype(np.float32)
    freq_pr /= (freq_pr.sum() + 1e-9)
    ent  = float(scipy_entropy(freq_pr + 1e-9))
    stats = np.array([
        np.log1p(total),          # log_len
        np.log1p(unique),         # log_unique
        unique / (total + 1e-9),  # diversity
        ent,                      # entropy
        np.log1p(total),          # log_count  (approximate: window size)
        np.log1p(total * 20),     # log_size   (approximate: ~20 bytes per syscall line)
        float(np.mean(arr)),      # mean_id
        float(np.std(arr)),       # std_id
    ], dtype=np.float32)

    # bigram_40
    if total >= 2:
        bg_cnt = collections.Counter(zip(seq_ids[:-1], seq_ids[1:]))
        bg_tot = sum(bg_cnt.values()) + 1e-9
        bigrams = np.array([bg_cnt.get(bg, 0) / bg_tot for bg in TOP_BIGRAMS],
                           dtype=np.float32)
    else:
        bigrams = np.zeros(len(TOP_BIGRAMS), dtype=np.float32)

    # ver_onehot — match on major.minor prefix
    ver_short = '.'.join(kernel_ver.split('.')[:2])
    ver_vec   = np.array([1.0 if v == ver_short else 0.0 for v in VER_COLS],
                         dtype=np.float32)

    return np.concatenate([freq, disc, stats, bigrams, ver_vec])

# ── Scoring ───────────────────────────────────────────────────────────────────
def score_window(seq_ids, kernel_ver):
    feat = build_features(seq_ids, kernel_ver).reshape(1, -1)
    feat_scaled = scaler.transform(feat).astype(np.float32)

    if_raw  = float(-iso.decision_function(feat_scaled)[0])
    vae_raw = float(vae_recon_error(feat_scaled)[0])

    if_norm  = float(norm_if(np.array([if_raw]))[0])
    vae_norm = float(norm_vae(np.array([vae_raw]))[0])
    ens_score = ALPHA * vae_norm + (1 - ALPHA) * if_norm

    return {
        'if_raw':   round(if_raw, 4),
        'vae_raw':  round(vae_raw, 2),
        'if_norm':  round(if_norm, 4),
        'vae_norm': round(vae_norm, 4),
        'ensemble': round(ens_score, 4),
        'is_attack': bool(ens_score >= THRESHOLD),
    }

# ── Parse Tetragon events ─────────────────────────────────────────────────────
def parse_event(raw):
    """
    Returns (pod_key, syscall_id) or (None, None) if not a syscall event.
    pod_key: "<namespace>/<pod-name>" or "<namespace>/<pid>" if no pod.
    """
    try:
        ev = json.loads(raw)
    except json.JSONDecodeError:
        return None, None

    # ── process_kprobe: function_name = "__x64_sys_read" / "sys_read"
    if 'process_kprobe' in ev:
        kp   = ev['process_kprobe']
        func = kp.get('function_name', '')
        # strip common prefixes
        for prefix in ('__x64_sys_', '__arm64_sys_', 'sys_', '__se_sys_'):
            if func.startswith(prefix):
                func = func[len(prefix):]
                break
        sid = syscall_name_to_id.get(func)
        if sid is None:
            return None, None
        proc  = kp.get('process', {})
        pod   = proc.get('pod', {})
        ns    = pod.get('namespace') or ev.get('node_name', 'host')
        pname = pod.get('name') or str(proc.get('pid', 'unknown'))
        return f'{ns}/{pname}', sid

    # ── process_tracepoint: sys_enter with syscall number in args[0]
    if 'process_tracepoint' in ev:
        tp = ev['process_tracepoint']
        if tp.get('event') != 'sys_enter':
            return None, None
        args_list = tp.get('args', [])
        if not args_list:
            return None, None
        sid = int(args_list[0].get('long_arg', -1))
        if sid < 0 or sid not in syscall_id_to_name:
            return None, None
        proc  = tp.get('process', {})
        pod   = proc.get('pod', {})
        ns    = pod.get('namespace') or ev.get('node_name', 'host')
        pname = pod.get('name') or str(proc.get('pid', 'unknown'))
        return f'{ns}/{pname}', sid

    # ── process_exec: execve (syscall 59) — captured by Tetragon for every exec
    if 'process_exec' in ev:
        pe   = ev['process_exec']
        proc = pe.get('process', {})
        pod  = proc.get('pod', {})
        ns   = pod.get('namespace') or ev.get('node_name', 'host')
        pname= pod.get('name') or str(proc.get('pid', 'unknown'))
        return f'{ns}/{pname}', syscall_name_to_id.get('execve', 59)

    # ── process_exit: exit_group (syscall 231)
    if 'process_exit' in ev:
        pe   = ev['process_exit']
        proc = pe.get('process', {})
        pod  = proc.get('pod', {})
        ns   = pod.get('namespace') or ev.get('node_name', 'host')
        pname= pod.get('name') or str(proc.get('pid', 'unknown'))
        return f'{ns}/{pname}', syscall_name_to_id.get('exit_group', 231)

    return None, None

# ── Main streaming loop ───────────────────────────────────────────────────────
log.info(f'Ready. Window={args.window} step={args.step} '
         f'threshold={THRESHOLD:.4f} kernel={args.kernel_ver}')
log.info('Streaming events from stdin...\n')

WINDOW   = args.window
STEP     = args.step if args.step > 0 else WINDOW   # tumbling if step=0
NS_FILTER= args.namespace

windows   = collections.defaultdict(collections.deque)  # pod → deque[syscall_id]
step_cnts = collections.defaultdict(int)                 # pod → events since last score

event_cnt = 0
alert_cnt = 0
t_start   = time.time()

try:
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue

        pod_key, sid = parse_event(raw)
        if pod_key is None:
            continue

        # Namespace filter
        ns_part = pod_key.split('/')[0]
        if NS_FILTER and ns_part != NS_FILTER:
            continue

        event_cnt += 1
        w = windows[pod_key]
        w.append(sid)

        # Keep window bounded
        while len(w) > WINDOW:
            w.popleft()

        step_cnts[pod_key] += 1

        # Score every STEP new syscalls once we have a full window
        if len(w) < WINDOW or step_cnts[pod_key] < STEP:
            continue
        step_cnts[pod_key] = 0

        if args.dry_run:
            log.debug(f'dry-run: {pod_key} window full ({WINDOW})')
            continue

        t0     = time.perf_counter()
        result = score_window(list(w), args.kernel_ver)
        latency= (time.perf_counter() - t0) * 1000

        elapsed = int(time.time() - t_start)
        verdict = 'ATTACK' if result['is_attack'] else 'normal'

        if result['is_attack']:
            alert_cnt += 1

        if result['is_attack'] or args.scores_only:
            top_syscalls = collections.Counter(w).most_common(5)
            top_str = ' '.join(f'{syscall_id_to_name.get(k,str(k))}×{v}'
                               for k, v in top_syscalls)
            print(
                f'[{elapsed:5d}s] {verdict:6s} | {pod_key:<45} | '
                f'ens={result["ensemble"]:.3f} '
                f'vae={result["vae_norm"]:.3f} '
                f'if={result["if_norm"]:.3f} '
                f'| lat={latency:.0f}ms | top: {top_str}',
                flush=True,
            )

        if event_cnt % 10000 == 0:
            log.info(f'events={event_cnt}  alerts={alert_cnt}  pods={len(windows)}  '
                     f'uptime={elapsed}s')

except KeyboardInterrupt:
    pass

elapsed = int(time.time() - t_start)
log.info(f'\nDone. events={event_cnt}  alerts={alert_cnt}  pods={len(windows)}  '
         f'uptime={elapsed}s')
