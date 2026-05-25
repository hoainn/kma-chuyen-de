"""
DeSFAM — DongTing Training Pipeline (script mode)
Sync target: train_dongting.ipynb — keep function bodies identical in both files.
"""
from __future__ import annotations

import json
import os
import random
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from pymongo import MongoClient
from scipy.stats import entropy as scipy_entropy
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Environment ────────────────────────────────────────────────────────────
MONGO_URI   = os.environ.get('MONGO_URI',   'mongodb://mongo:27017/')
MONGO_DB    = os.environ.get('MONGO_DB',    'syzbot_DB')
SYSCALL_TBL = os.environ.get('SYSCALL_TBL', '/data/dongting_repo/syscall_64.tbl')
OUTPUT_DIR  = os.environ.get('OUTPUT_DIR',  '/model')

# Sliding-window training (paper §IV.B.1: length 15, stride 3). Opt-in so the
# default pipeline still trains on full per-process traces.
USE_SLIDING_WINDOW = os.environ.get('USE_SLIDING_WINDOW', '0') == '1'
WINDOW_LEN    = int(os.environ.get('WINDOW_LEN',    '15'))
WINDOW_STRIDE = int(os.environ.get('WINDOW_STRIDE', '3'))

# Categorical syscall frequencies (paper §IV.B.1, "Categorical Frequencies"):
# functional-category one-hot frequency per window. Adds n_cat dims after the
# version one-hot. Default ON because it is the paper-claimed signature feature.
USE_CATEGORIES = os.environ.get('USE_CATEGORIES', '1') == '1'

# Cap on windows PER SPLIT. DongTing's long attack traces expand to ~127 M
# train windows at stride=3 — that alone would need ~80 GB. We subsample each
# split independently (random per-trace, sorted to keep temporal locality).
# 2 M × 159 × 4 B ≈ 1.3 GB per split; train+val+test peak < 4 GB.
MAX_WINDOWS_PER_SPLIT = int(
    os.environ.get('MAX_WINDOWS_PER_SPLIT',
                   os.environ.get('MAX_TRAIN_WINDOWS', '2000000'))
)


# ── Status logging helpers (white-box progress) ───────────────────────────
import time as _time
_T0 = _time.monotonic()

def _rss_gib() -> float:
    """Best-effort current RSS in GiB. Returns 0 if unavailable. Inside the
    Linux training container ru_maxrss is in KB; on macOS it is in bytes."""
    try:
        import resource, sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024 * 1024) if sys.platform == 'darwin' else rss / (1024 * 1024)
    except Exception:
        return 0.0

def _log(msg: str) -> None:
    """Timestamped print with elapsed seconds + RSS so every long-running
    stage surfaces wall-clock and memory headroom."""
    elapsed = _time.monotonic() - _T0
    mm, ss = divmod(int(elapsed), 60)
    print(f'[{mm:02d}:{ss:02d}  ram={_rss_gib():4.1f}G]  {msg}', flush=True)

# Paper feature components NOT implemented here:
#   • Temporal features (Δt mean/std/max) — DongTing stores syscall sequences as
#     pipe-separated IDs only; no timestamps are recoverable from the dataset.
#   • PrefixSpan Access List pattern matching — deferred; high engineering cost
#     and orthogonal to the main detection model.
#   • Synthetic data augmentation of attack sequences — incompatible with the
#     normal-only training paradigm (IF and VAE never see attack data).
# These gaps are documented in the replication report; do not silently ignore.

# ── Hyperparameters ────────────────────────────────────────────────────────
CFG = {
    'top_k_freq':    60,
    'top_k_disc':    40,
    'top_k_ngrams':  40,
    'ngram_n':       2,
    'latent_dim':    8,
    'hidden_dim':    32,
    'dropout':       0.2,
    'l2':            1e-4,
    'lr':            1e-3,
    'epochs':        80,
    'batch_size':    256,
    'seeds':         [0, 1, 2],
    'n_estimators':  300,
    'contamination': 0.02,
    'alpha':         0.7,
    'global_seed':   42,
    'threshold':     'f1',
}

GLOBAL_SEED = CFG['global_seed']


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ── Step 1: Syscall table ──────────────────────────────────────────────────

def load_syscall_table(path: str) -> dict[str, int]:
    name_to_id: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                name_to_id[parts[2]] = int(parts[0])
    return name_to_id


# ── Syscall functional categories (paper §IV.B.1) ─────────────────────────
# Hand-curated x86-64 mapping. Categories follow common groupings used in syscall
# IDS research (e.g., NIS, ADFA-LD): process control, file I/O, memory, network,
# signal, time, IPC, security/privilege, and I/O event multiplexing.
# Syscalls not listed fall into 'other'.

_SYSCALL_CATEGORY_MAP: dict[str, str] = {
    # process control
    **dict.fromkeys([
        'fork', 'vfork', 'clone', 'clone3', 'execve', 'execveat',
        'exit', 'exit_group', 'wait4', 'waitid', 'kill', 'tkill', 'tgkill',
        'getpid', 'getppid', 'gettid', 'getpgrp', 'getpgid', 'setpgid',
        'getsid', 'setsid', 'getpriority', 'setpriority',
        'sched_yield', 'sched_setparam', 'sched_getparam',
        'sched_setscheduler', 'sched_getscheduler',
        'sched_setaffinity', 'sched_getaffinity',
        'pause', 'arch_prctl', 'set_tid_address', 'set_robust_list',
        'getrlimit', 'setrlimit', 'prlimit64', 'getrusage',
    ], 'process'),
    # file I/O
    **dict.fromkeys([
        'open', 'openat', 'openat2', 'creat', 'close', 'close_range',
        'read', 'write', 'pread64', 'pwrite64', 'readv', 'writev',
        'preadv', 'pwritev', 'preadv2', 'pwritev2',
        'lseek', 'ftruncate', 'truncate',
        'access', 'faccessat', 'faccessat2',
        'stat', 'fstat', 'lstat', 'newfstatat', 'statx', 'statfs', 'fstatfs',
        'unlink', 'unlinkat', 'rename', 'renameat', 'renameat2',
        'mkdir', 'mkdirat', 'rmdir', 'link', 'linkat', 'symlink', 'symlinkat',
        'readlink', 'readlinkat',
        'chmod', 'fchmod', 'fchmodat', 'fchmodat2',
        'chown', 'fchown', 'lchown', 'fchownat',
        'umask', 'sync', 'fsync', 'fdatasync', 'syncfs',
        'getdents', 'getdents64', 'getcwd', 'chdir', 'fchdir', 'chroot',
        'mount', 'umount2', 'pivot_root',
        'pipe', 'pipe2', 'dup', 'dup2', 'dup3',
        'fcntl', 'ioctl', 'flock',
        'sendfile', 'splice', 'tee', 'vmsplice', 'copy_file_range',
        'fallocate', 'fadvise64', 'sync_file_range',
        'inotify_init', 'inotify_init1', 'inotify_add_watch', 'inotify_rm_watch',
        'fanotify_init', 'fanotify_mark',
        'utime', 'utimes', 'utimensat', 'futimesat',
        'name_to_handle_at', 'open_by_handle_at',
        'quotactl', 'quotactl_fd',
    ], 'file'),
    # memory
    **dict.fromkeys([
        'brk', 'mmap', 'mmap2', 'munmap', 'mremap',
        'mprotect', 'madvise', 'mincore', 'msync',
        'pkey_alloc', 'pkey_free', 'pkey_mprotect',
        'mlock', 'mlock2', 'munlock', 'mlockall', 'munlockall',
        'memfd_create', 'memfd_secret',
        'mbind', 'set_mempolicy', 'get_mempolicy',
        'migrate_pages', 'move_pages',
        'process_madvise', 'process_vm_readv', 'process_vm_writev',
        'shmget', 'shmat', 'shmdt', 'shmctl',
    ], 'memory'),
    # network
    **dict.fromkeys([
        'socket', 'socketpair', 'bind', 'listen',
        'accept', 'accept4', 'connect',
        'shutdown', 'sendto', 'recvfrom',
        'sendmsg', 'recvmsg', 'sendmmsg', 'recvmmsg',
        'setsockopt', 'getsockopt', 'getsockname', 'getpeername',
    ], 'network'),
    # signal
    **dict.fromkeys([
        'rt_sigaction', 'rt_sigprocmask', 'rt_sigreturn',
        'rt_sigpending', 'rt_sigtimedwait', 'rt_sigqueueinfo',
        'rt_sigsuspend', 'rt_tgsigqueueinfo',
        'sigaltstack', 'signalfd', 'signalfd4',
        'pidfd_open', 'pidfd_send_signal', 'pidfd_getfd',
        'restart_syscall',
    ], 'signal'),
    # time
    **dict.fromkeys([
        'gettimeofday', 'settimeofday', 'time',
        'clock_gettime', 'clock_settime', 'clock_getres',
        'clock_nanosleep', 'nanosleep',
        'alarm', 'setitimer', 'getitimer',
        'timer_create', 'timer_delete', 'timer_settime',
        'timer_gettime', 'timer_getoverrun',
        'timerfd_create', 'timerfd_settime', 'timerfd_gettime',
        'adjtimex', 'clock_adjtime', 'times',
    ], 'time'),
    # IPC
    **dict.fromkeys([
        'msgget', 'msgsnd', 'msgrcv', 'msgctl',
        'semget', 'semop', 'semtimedop', 'semctl',
        'futex', 'futex_waitv', 'futex_wake', 'futex_wait', 'futex_requeue',
        'mq_open', 'mq_unlink', 'mq_timedsend', 'mq_timedreceive',
        'mq_notify', 'mq_getsetattr',
        'eventfd', 'eventfd2',
    ], 'ipc'),
    # security / privilege / namespaces
    **dict.fromkeys([
        'setuid', 'setgid', 'setreuid', 'setregid',
        'setresuid', 'setresgid', 'setfsuid', 'setfsgid',
        'getuid', 'getgid', 'geteuid', 'getegid',
        'getresuid', 'getresgid', 'setgroups', 'getgroups',
        'capset', 'capget',
        'prctl', 'seccomp', 'ptrace',
        'keyctl', 'add_key', 'request_key',
        'unshare', 'setns',
        'landlock_create_ruleset', 'landlock_add_rule', 'landlock_restrict_self',
    ], 'security'),
    # I/O multiplexing / event notification
    **dict.fromkeys([
        'select', 'pselect6', 'poll', 'ppoll',
        'epoll_create', 'epoll_create1', 'epoll_wait', 'epoll_pwait',
        'epoll_pwait2', 'epoll_ctl',
        'io_setup', 'io_destroy', 'io_getevents',
        'io_submit', 'io_cancel', 'io_pgetevents',
        'io_uring_setup', 'io_uring_enter', 'io_uring_register',
    ], 'io_event'),
}

# Stable column order for the categorical-frequency feature group.
SYSCALL_CATEGORIES: list[str] = [
    'process', 'file', 'memory', 'network', 'signal',
    'time', 'ipc', 'security', 'io_event', 'other',
]


def build_id_to_category(name_to_id: dict[str, int]) -> dict[int, str]:
    """Invert SYSCALL_CATEGORY_MAP and merge with the syscall_64.tbl ID space.
    Unknown syscalls fall back to 'other'."""
    return {
        sid: _SYSCALL_CATEGORY_MAP.get(name, 'other')
        for name, sid in name_to_id.items()
    }


# ── Step 2: Load sequences from MongoDB ───────────────────────────────────

def load_sequences(name_to_id: dict[str, int]):
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
    db = client[MONGO_DB]

    _log('  Indexing normal sequences from Mongo...')
    normal_idx: dict[str, str] = {}
    for doc in db.kernel_syscall_normal_strace.find(
            {}, {'kns_normal_file_name': 1, 'kns_normal_mlseq_list': 1}):
        normal_idx[doc['kns_normal_file_name']] = doc['kns_normal_mlseq_list']

    _log('  Indexing attack sequences from Mongo...')
    attack_idx: dict[str, str] = {}
    for doc in db.kernel_syscallhook_bugpoc_trace_sum.find(
            {}, {'kshs_poclog_name': 1, 'kshs_bugpoc_syscall_list': 1}):
        attack_idx[doc['kshs_poclog_name']] = doc['kshs_bugpoc_syscall_list']

    _log(f'  Indexed normal={len(normal_idx):,}  attack={len(attack_idx):,}')
    _log('  Reading baseline index...')
    seqs, labels, splits, ver_list = [], [], [], []
    missing = 0
    split_map = {'DTDS-train': 'train', 'DTDS-validation': 'val', 'DTDS-test': 'test'}

    for doc in db.kernel_convert_baseline.find(
            {}, {'kcb_bug_name': 1, 'kcb_seq_lables': 1,
                 'kcb_seq_class': 1, 'kcb_master_line_ver': 1}):
        name  = doc['kcb_bug_name']
        label = 0 if doc['kcb_seq_lables'] == 'Normal' else 1
        split = split_map.get(doc['kcb_seq_class'], 'train')
        ver   = str(doc.get('kcb_master_line_ver', '')).strip()

        if label == 0:
            raw = normal_idx.get(name)
            if raw is None:
                missing += 1; continue
            ids = [int(x) for x in raw.split('|') if x.strip().isdigit()]
        else:
            raw = attack_idx.get(name)
            if raw is None:
                missing += 1; continue
            ids = [name_to_id[nm.strip()] for nm in raw.split('|')
                   if nm.strip() in name_to_id]

        if len(ids) < 2:
            missing += 1; continue

        seqs.append(ids); labels.append(label)
        splits.append(split); ver_list.append(ver)

    client.close()
    print(f'  Loaded {len(seqs)} sequences ({missing} skipped)')
    return seqs, labels, splits, ver_list


# ── Step 2b: Sliding window expansion (opt-in, streaming) ─────────────────
# We never materialise the list of all windows — at this dataset scale that
# would consume tens of GB of Python list/int objects and OOM the trainer.
# Instead we count windows up front (cheap), then stream them straight into
# the pre-allocated feature matrix in build_features_windowed.

def sliding_window(seq: list[int], length: int, stride: int) -> list[list[int]]:
    """Eager helper kept for unit tests; NOT used in main pipeline."""
    if len(seq) < length:
        return []
    return [seq[i: i + length] for i in range(0, len(seq) - length + 1, stride)]


def count_windows(seq_len: int, length: int, stride: int) -> int:
    if seq_len < length:
        return 0
    return (seq_len - length) // stride + 1


def iter_windows(seqs, ver, y, length: int, stride: int):
    """Yield (window_view, ver, label) without copying the underlying list.
    The window is a list[int] of exactly `length` elements taken as a slice;
    callers must not mutate it (build_features_windowed only reads)."""
    for s, v, yy in zip(seqs, ver, y):
        n_win = count_windows(len(s), length, stride)
        for i in range(n_win):
            start = i * stride
            yield s[start: start + length], v, int(yy)


def iter_window_arrays(seqs_arr, ver, y, length: int, stride: int):
    """Vectorised variant: yields numpy *views* (no copy) into pre-converted
    int32 trace arrays. The window has dtype int32 and length exactly `length`."""
    for s_arr, v, yy in zip(seqs_arr, ver, y):
        n_win = count_windows(len(s_arr), length, stride)
        for i in range(n_win):
            start = i * stride
            yield s_arr[start: start + length], v, int(yy)


# ── Step 3: Feature engineering ───────────────────────────────────────────

def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0: return 0.0
    p = counts / total
    return float(scipy_entropy(p[p > 0], base=2))


def _ver_short(v: str) -> str:
    s = v.strip().lower()
    for pfx in ('linux-', 'kernel-', 'v'):
        if s.startswith(pfx): s = s[len(pfx):]
    parts = s.split('.')
    clean = []
    for p in parts:
        p2 = p.split('-')[0].split('+')[0]
        if p2.isdigit(): clean.append(p2)
        else: break
    if len(clean) >= 2: return f'{clean[0]}.{clean[1]}'
    return '.'.join(clean) if clean else ''


def fit_vocab(seqs_train, ver_train, top_k_freq=60, top_k_disc=40, top_k_ngrams=40, ngram_n=2):
    all_sc: Counter = Counter()
    all_ng: Counter = Counter()
    for seq in seqs_train:
        all_sc.update(seq)
        if ngram_n >= 2:
            for i in range(len(seq) - ngram_n + 1):
                all_ng[tuple(seq[i: i + ngram_n])] += 1
    top_ids    = [sc for sc, _ in all_sc.most_common(top_k_freq)]
    cands      = [sc for sc, _ in all_sc.most_common(top_k_freq + top_k_disc)]
    disc_ids   = cands[top_k_freq: top_k_freq + top_k_disc]
    top_ngrams = [g for g, _ in all_ng.most_common(top_k_ngrams if ngram_n >= 2 else 0)]
    ver_cols   = sorted(set(s for v in ver_train if v and (s := _ver_short(v))))
    return top_ids, disc_ids, top_ngrams, ver_cols


def build_features_windowed(seqs, ver_list, y_in,
                            top_ids, disc_ids, top_ngrams, ver_cols,
                            length: int, stride: int, ngram_n: int = 2,
                            cat_cols: list[str] | None = None,
                            id_to_cat: dict[int, str] | None = None,
                            max_windows: int | None = None,
                            rng_seed: int = 0):
    """Per-trace batched featurisation. Returns (X, y, n_dropped).

    All windows of one trace are featurised in a single numpy call using
    sliding_window_view + reused workspace buffers. Compared with the legacy
    per-window loop this drops allocator churn by ~100× (one buffer reset per
    trace, not per window) and eliminates the Python-int allocations the
    bigram block used to generate via .tolist().

    Math is intentionally bit-identical to the per-window formulation so the
    inference-side Featurizer (which still uses the per-window form) produces
    the same vectors.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    # Convert each trace to a contiguous int32 array once; slices become views.
    seqs_arr: list[np.ndarray] = [np.asarray(s, dtype=np.int32) for s in seqs]
    y_arr = np.asarray(y_in, dtype=np.int32)

    n_freq  = len(top_ids)
    n_disc  = len(disc_ids)
    n_ng    = len(top_ngrams)
    n_ver   = len(ver_cols)
    n_cat   = len(cat_cols) if cat_cols else 0
    n_feat  = n_freq + n_disc + 8 + n_ng + n_ver + n_cat
    ver_idx = {v: i for i, v in enumerate(ver_cols)}

    # Window-count pass also gives us max_n_win for buffer sizing.
    n_per_trace: list[int] = []
    n_total   = 0
    n_dropped = 0
    max_n_win = 0
    for s_arr in seqs_arr:
        n = count_windows(len(s_arr), length, stride)
        n_per_trace.append(n)
        if n == 0: n_dropped += 1
        if n > max_n_win: max_n_win = n
        n_total += n

    # Optional per-trace subsampling so the train X matrix fits in memory.
    # DongTing's long attack traces would otherwise yield ~10 GB matrices. We
    # cap globally and apportion to each trace by its share of the total
    # (random pick within each trace, sorted to keep temporal locality).
    sampled_idx_per_trace: list[np.ndarray] | None = None
    if max_windows is not None and n_total > max_windows:
        ratio = max_windows / n_total
        rng = np.random.default_rng(rng_seed)
        sampled_idx_per_trace = []
        new_total = 0
        for n in n_per_trace:
            if n == 0:
                sampled_idx_per_trace.append(np.empty(0, dtype=np.int32))
                continue
            k = max(1, int(np.floor(n * ratio)))
            k = min(k, n)
            idx = rng.choice(n, size=k, replace=False)
            idx.sort()
            sampled_idx_per_trace.append(idx.astype(np.int32, copy=False))
            new_total += k
        print(f'  Subsampling windows: {n_total:,} → {new_total:,} '
              f'(cap={max_windows:,}, ratio={ratio:.3f})')
        n_total = new_total
        # Recompute max_n_win on the sampled counts (smaller is fine).
        max_n_win = max((arr.size for arr in sampled_idx_per_trace), default=0)

    # ── Precompute lookup arrays ──────────────────────────────────────────
    top_ids_arr  = np.asarray(top_ids,  dtype=np.int64) if n_freq else np.empty(0, np.int64)
    disc_ids_arr = np.asarray(disc_ids, dtype=np.int64) if n_disc else np.empty(0, np.int64)

    max_id = 0
    if n_freq:  max_id = max(max_id, int(top_ids_arr.max()))
    if n_disc:  max_id = max(max_id, int(disc_ids_arr.max()))
    if id_to_cat: max_id = max(max_id, max(id_to_cat.keys()))
    for s_arr in seqs_arr:
        if s_arr.size:
            v = int(s_arr.max())
            if v > max_id: max_id = v
    n_ids = max_id + 1

    if n_cat > 0 and id_to_cat is not None:
        cat_idx_map = {c: i for i, c in enumerate(cat_cols)}
        other_i = cat_idx_map.get('other', 0)
        cat_of_id = np.full(n_ids, other_i, dtype=np.int16)
        for sid, name in id_to_cat.items():
            ci = cat_idx_map.get(name)
            if ci is not None and 0 <= sid < n_ids:
                cat_of_id[sid] = ci
    else:
        cat_of_id = None

    # Bigram lookup: sorted int64 codes + parallel column array → searchsorted.
    if n_ng > 0 and ngram_n == 2:
        pairs = sorted(((int(a) << 16) | int(b), i) for i, (a, b) in enumerate(top_ngrams))
        sorted_codes = np.fromiter((c for c, _ in pairs), dtype=np.int64, count=len(pairs))
        code_to_col  = np.fromiter((i for _, i in pairs), dtype=np.int32, count=len(pairs))
    else:
        sorted_codes = np.empty(0, dtype=np.int64)
        code_to_col  = np.empty(0, dtype=np.int32)

    ver_short_per_trace = [_ver_short(v) for v in ver_list]

    # ── Output + reused per-trace workspaces ──────────────────────────────
    X = np.zeros((n_total, n_feat), dtype=np.float32)
    y = np.zeros((n_total,),        dtype=np.int32)

    if max_n_win == 0:
        return X, y, n_dropped

    buf_counts = np.zeros((max_n_win, n_ids), dtype=np.int32)
    buf_cat    = np.zeros((max_n_win, n_cat), dtype=np.int32) if n_cat   else None
    buf_bg     = np.zeros((max_n_win, n_ng),  dtype=np.int32) if n_ng    else None

    off      = n_freq + n_disc
    ng_off   = off + 8
    ver_off  = ng_off + n_ng
    cat_off  = ver_off + n_ver
    total    = length
    n_win_ng = max(total - ngram_n + 1, 1)
    row_idx_col = np.arange(max_n_win)[:, None]

    # ── Per-trace featurisation ───────────────────────────────────────────
    n_traces_total = len(seqs_arr)
    progress_every = max(1, n_traces_total // 20)  # ~5% increments
    row_off = 0
    for ti, s_arr in enumerate(seqs_arr):
        if ti and ti % progress_every == 0:
            _log(f'  featurise: trace {ti:,}/{n_traces_total:,} '
                 f'({100*ti/n_traces_total:.0f}%)  rows={row_off:,}/{n_total:,}')
        n_win = n_per_trace[ti]
        if n_win == 0: continue
        # (n_win, length) view — zero copy
        all_windows = sliding_window_view(s_arr, window_shape=length)[::stride]
        assert all_windows.shape[0] == n_win
        if sampled_idx_per_trace is not None:
            keep = sampled_idx_per_trace[ti]
            if keep.size == 0: continue
            # Fancy index → contiguous (k, length) int32 copy. Small.
            windows = all_windows[keep]
            n_win = keep.size
        else:
            windows = all_windows

        # Per-window full histogram, scatter into reused buffer
        counts_view = buf_counts[:n_win]
        counts_view.fill(0)
        np.add.at(counts_view, (row_idx_col[:n_win], windows), 1)

        X_block = X[row_off: row_off + n_win]

        # freq_60
        if n_freq:
            X_block[:, :n_freq] = counts_view[:, top_ids_arr] / total
        # disc_40
        if n_disc:
            X_block[:, n_freq:n_freq + n_disc] = (counts_view[:, disc_ids_arr] > 0).astype(np.float32)

        # stats_8 (math identical to per-window form)
        raw = counts_view[:, top_ids_arr].astype(np.float32) if n_freq else np.zeros((n_win, 0), np.float32)
        if n_freq:
            totals = raw.sum(axis=1, keepdims=True)
            safe_t = np.where(totals > 0, totals, 1.0)
            p = raw / safe_t
            with np.errstate(divide='ignore', invalid='ignore'):
                log_p = np.log2(p, where=(p > 0))
            log_p = np.where(p > 0, log_p, 0.0)
            H = -(p * log_p).sum(axis=1)
            X_block[:, off + 0] = np.where(totals.squeeze(-1) > 0, H, 0.0)
            X_block[:, off + 3] = raw.max(axis=1) / total
            raw_nan = np.where(raw > 0, raw, np.nan)
            with np.errstate(invalid='ignore'):
                p75 = np.nanpercentile(raw_nan, 75, axis=1)
            X_block[:, off + 4] = np.where(np.isnan(p75), 0.0, p75).astype(np.float32)
            X_block[:, off + 5] = raw.std(axis=1)
            X_block[:, off + 6] = (raw > 0).sum(axis=1).astype(np.float32) / max(n_freq, 1)
        X_block[:, off + 1] = (counts_view > 0).sum(axis=1).astype(np.float32)
        X_block[:, off + 2] = np.log1p(total)
        X_block[:, off + 7] = total / 1000.0

        # bigrams via searchsorted
        if sorted_codes.size and length >= 2:
            a = windows[:, :-1].astype(np.int64)
            b = windows[:,  1:].astype(np.int64)
            codes = (a << 16) | b                            # (n_win, length-1)
            pos = np.searchsorted(sorted_codes, codes)
            pos_clip = np.minimum(pos, sorted_codes.size - 1)
            hit = sorted_codes[pos_clip] == codes
            cols = code_to_col[pos_clip]
            bg_view = buf_bg[:n_win]
            bg_view.fill(0)
            rows = np.broadcast_to(row_idx_col[:n_win], cols.shape)
            np.add.at(bg_view, (rows[hit], cols[hit]), 1)
            X_block[:, ng_off:ng_off + n_ng] = bg_view / n_win_ng

        # ver_onehot
        vs = ver_short_per_trace[ti]
        if vs and vs in ver_idx:
            X_block[:, ver_off + ver_idx[vs]] = 1.0

        # cat_10
        if cat_of_id is not None:
            ci = cat_of_id[windows]                          # (n_win, length)
            cat_view = buf_cat[:n_win]
            cat_view.fill(0)
            np.add.at(cat_view, (row_idx_col[:n_win], ci), 1)
            X_block[:, cat_off:cat_off + n_cat] = cat_view / total

        # labels are uniform across this trace's windows
        y[row_off: row_off + n_win] = int(y_arr[ti])
        row_off += n_win

    return X, y, n_dropped


def build_features(seqs, ver_list, top_ids, disc_ids, top_ngrams, ver_cols, ngram_n=2,
                   cat_cols: list[str] | None = None,
                   id_to_cat: dict[int, str] | None = None) -> np.ndarray:
    n_freq  = len(top_ids)
    n_disc  = len(disc_ids)
    n_ng    = len(top_ngrams)
    n_ver   = len(ver_cols)
    n_cat   = len(cat_cols) if cat_cols else 0
    n_feat  = n_freq + n_disc + 8 + n_ng + n_ver + n_cat
    top_idx  = {sc: i for i, sc in enumerate(top_ids)}
    disc_idx = {sc: i for i, sc in enumerate(disc_ids)}
    ng_idx   = {g: i  for i, g  in enumerate(top_ngrams)}
    ver_idx  = {v: i  for i, v  in enumerate(ver_cols)}
    cat_idx  = {c: i  for i, c  in enumerate(cat_cols or [])}
    X = np.zeros((len(seqs), n_feat), dtype=np.float32)
    for i, (seq, ver) in enumerate(zip(seqs, ver_list)):
        if not seq: continue
        total = len(seq)
        for sc in seq:
            if sc in top_idx: X[i, top_idx[sc]] += 1
        X[i, :n_freq] /= max(total, 1)
        for sc in set(seq):
            if sc in disc_idx: X[i, n_freq + disc_idx[sc]] = 1.0
        off = n_freq + n_disc
        raw = X[i, :n_freq] * total
        X[i, off+0] = _entropy(raw)
        X[i, off+1] = float(len(set(seq)))
        X[i, off+2] = float(np.log1p(total))
        X[i, off+3] = float(raw.max()) / max(total, 1)
        nz = raw[raw > 0]
        X[i, off+4] = float(np.percentile(nz, 75)) if len(nz) else 0.0
        X[i, off+5] = float(np.std(raw))
        X[i, off+6] = float((raw > 0).sum()) / max(n_freq, 1)
        X[i, off+7] = float(total) / 1000.0
        ng_off = off + 8
        if n_ng > 0 and total >= ngram_n:
            n_win = max(total - ngram_n + 1, 1)
            for j in range(total - ngram_n + 1):
                gram = tuple(seq[j: j + ngram_n])
                if gram in ng_idx: X[i, ng_off + ng_idx[gram]] += 1
            X[i, ng_off: ng_off + n_ng] /= n_win
        ver_off = ng_off + n_ng
        vs = _ver_short(ver)
        if vs in ver_idx: X[i, ver_off + ver_idx[vs]] = 1.0
        if n_cat > 0 and id_to_cat is not None:
            cat_off = ver_off + n_ver
            for sc in seq:
                c = id_to_cat.get(sc, 'other')
                ci = cat_idx.get(c)
                if ci is not None: X[i, cat_off + ci] += 1
            X[i, cat_off: cat_off + n_cat] /= max(total, 1)
    return X


# ── Step 4 / 5: Models ────────────────────────────────────────────────────

def build_vae(input_dim: int, latent_dim: int = 8, hidden_dim: int = 32,
              dropout: float = 0.2, l2: float = 1e-4):
    reg = keras.regularizers.l2(l2)

    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            z_lv = tf.clip_by_value(z_log_var, -10.0, 10.0)
            return z_mean + tf.exp(0.5 * z_lv) * tf.random.normal(tf.shape(z_mean))

    class KLLoss(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            z_lv = tf.clip_by_value(z_log_var, -10.0, 10.0)
            kl = -0.5 * tf.reduce_mean(1 + z_lv - tf.square(z_mean) - tf.exp(z_lv))
            self.add_loss(kl)
            return inputs

    x_in = keras.Input(shape=(input_dim,), name='encoder_input')
    x = layers.Dense(hidden_dim, activation='selu', kernel_regularizer=reg)(x_in)
    x = layers.Dropout(dropout)(x)
    z_mean    = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling(name='z')([z_mean, z_log_var])
    [z_mean, z_log_var] = KLLoss(name='kl_loss')([z_mean, z_log_var])
    encoder = keras.Model(x_in, [z_mean, z_log_var, z], name='encoder')

    z_in = keras.Input(shape=(latent_dim,), name='decoder_input')
    x = layers.Dense(hidden_dim, activation='selu', kernel_regularizer=reg)(z_in)
    x = layers.Dropout(dropout)(x)
    x_out = layers.Dense(input_dim, name='decoder_output')(x)
    decoder = keras.Model(z_in, x_out, name='decoder')

    vae = keras.Model(x_in, decoder(encoder(x_in)[2]), name='vae')
    return encoder, decoder, vae


def recon_error(X, encoder, decoder) -> np.ndarray:
    _, _, z = encoder(X, training=False)
    x_hat = decoder(z, training=False)
    return np.mean((X - x_hat.numpy()) ** 2, axis=1)


# ── Step 6: Ensemble ──────────────────────────────────────────────────────

class EnsembleScorer:
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self._vae_scaler = RobustScaler(quantile_range=(1.0, 99.0))
        self._if_scaler  = RobustScaler(quantile_range=(1.0, 99.0))

    def fit(self, vae_scores, if_scores):
        self._vae_scaler.fit(vae_scores.reshape(-1, 1))
        self._if_scaler.fit(if_scores.reshape(-1, 1))
        return self

    def score(self, vae_scores, if_scores) -> np.ndarray:
        v = self._vae_scaler.transform(vae_scores.reshape(-1, 1)).ravel()
        f = self._if_scaler.transform(if_scores.reshape(-1, 1)).ravel()
        return self.alpha * v + (1 - self.alpha) * f

    def save(self, out_dir: Path):
        joblib.dump(self._vae_scaler, out_dir / 'ensemble_vae_scaler.joblib')
        joblib.dump(self._if_scaler,  out_dir / 'ensemble_if_scaler.joblib')
        (out_dir / 'ensemble_params.json').write_text(
            json.dumps({'alpha': self.alpha, 'fitted': True}, indent=2))


# ── Step 7: Evaluation helpers ────────────────────────────────────────────

def select_threshold(y_val, scores, strategy='f1') -> float:
    """Pick a decision threshold by strategy.

    For 'f1' we used to loop over every roc_curve threshold and call sklearn's
    f1_score — O(thresholds × N) ≈ 1e11 ops on 2M-window val splits, blowing up
    to tens of minutes per call. The vectorised path below uses cumulative
    sums on the descending-sorted scores: O(N log N) for the sort and O(N) for
    everything else, regardless of unique-threshold count.
    """
    y_val  = np.asarray(y_val)
    scores = np.asarray(scores)
    if strategy == 'p99':
        return float(np.percentile(scores[y_val == 0], 99))
    if strategy == 'fpr5':
        fpr_a, tpr_a, thresholds = roc_curve(y_val, scores)
        valid = np.where(fpr_a <= 0.05)[0]
        idx = valid[np.argmax(tpr_a[valid])] if len(valid) else 0
        return float(thresholds[idx])
    # 'f1' vectorised sweep
    n_pos = int((y_val == 1).sum())
    if n_pos == 0:
        return 0.5
    order = np.argsort(-scores, kind='stable')   # descending
    y_sorted  = y_val[order].astype(np.int32)
    s_sorted  = scores[order]
    tp_cum    = np.cumsum(y_sorted)
    n_predpos = np.arange(1, len(y_sorted) + 1, dtype=np.int64)
    precision = tp_cum / np.maximum(n_predpos, 1)
    recall    = tp_cum / n_pos
    denom     = precision + recall
    f1_curve  = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    best_idx  = int(np.argmax(f1_curve))
    return float(s_sorted[best_idx])


def compute_metrics(y_true, scores, threshold) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'threshold':  float(threshold),
        'auc_roc':    float(roc_auc_score(y_true, scores)),
        'ap':         float(average_precision_score(y_true, scores)),
        'f1':         float(f1_score(y_true, y_pred, zero_division=0)),
        'precision':  float(precision_score(y_true, y_pred, zero_division=0)),
        'recall_tpr': float(recall_score(y_true, y_pred, zero_division=0)),
        'fpr':        float(fp / max(fp + tn, 1)),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


# ── Main training pipeline ────────────────────────────────────────────────

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(OUTPUT_DIR)
    set_seeds(GLOBAL_SEED)

    print(f'TensorFlow {tf.__version__}  NumPy {np.__version__}')
    print(f'Output dir: {OUTPUT_DIR}')

    # Step 1
    name_to_id = load_syscall_table(SYSCALL_TBL)
    print(f'{len(name_to_id)} syscalls loaded')

    # Step 2
    seqs, labels, splits, ver_list = load_sequences(name_to_id)
    idx_train = [i for i, s in enumerate(splits) if s == 'train']
    idx_val   = [i for i, s in enumerate(splits) if s == 'val']
    idx_test  = [i for i, s in enumerate(splits) if s == 'test']

    seqs_train = [seqs[i] for i in idx_train]
    ver_train  = [ver_list[i] for i in idx_train]
    y_train    = np.array([labels[i] for i in idx_train], dtype=np.int32)
    seqs_val   = [seqs[i] for i in idx_val]
    ver_val    = [ver_list[i] for i in idx_val]
    y_val      = np.array([labels[i] for i in idx_val], dtype=np.int32)
    seqs_test  = [seqs[i] for i in idx_test]
    ver_test   = [ver_list[i] for i in idx_test]
    y_test     = np.array([labels[i] for i in idx_test], dtype=np.int32)

    print(f'train={len(seqs_train):,}  val={len(seqs_val):,}  test={len(seqs_test):,}')

    # Step 3a — vocab fit on FULL normal training traces (length-agnostic,
    # stable counts). This deliberately runs before any sliding-window expansion
    # so the vocab is independent of windowing.
    normal_train_seqs = [seqs_train[i] for i in range(len(seqs_train)) if y_train[i] == 0]
    normal_train_ver  = [ver_train[i]  for i in range(len(seqs_train)) if y_train[i] == 0]
    top_ids, disc_ids, top_ngrams, ver_cols = fit_vocab(
        normal_train_seqs, normal_train_ver,
        CFG['top_k_freq'], CFG['top_k_disc'], CFG['top_k_ngrams'], CFG['ngram_n'],
    )
    n_ng     = len(top_ngrams)
    n_ver    = len(ver_cols)

    # Build syscall→category mapping; persisted so inference produces the same
    # categorical-frequency columns from the live syscall stream.
    if USE_CATEGORIES:
        cat_cols  = list(SYSCALL_CATEGORIES)
        id_to_cat = build_id_to_category(name_to_id)
        n_cat     = len(cat_cols)
    else:
        cat_cols, id_to_cat, n_cat = [], None, 0

    feat_dim = CFG['top_k_freq'] + CFG['top_k_disc'] + 8 + n_ng + n_ver + n_cat
    print(f'freq={len(top_ids)}  disc={len(disc_ids)}  bigrams={n_ng}  '
          f'ver={n_ver}  cat={n_cat}  total_dim={feat_dim}')

    # Step 3b — Build feature matrices. With USE_SLIDING_WINDOW=1 we stream
    # windows directly into a pre-allocated X (never materialise the list of
    # windows) and labels are inherited from each parent trace.
    if USE_SLIDING_WINDOW:
        print(f'Sliding window: length={WINDOW_LEN} stride={WINDOW_STRIDE}'
              f'  cap_per_split={MAX_WINDOWS_PER_SPLIT:,}')
        before = (len(seqs_train), len(seqs_val), len(seqs_test))
        print('Building windowed feature matrices...')
        # All splits get the same cap so metrics on val/test remain
        # representative under random per-trace subsampling.
        X_train_full, y_train, d_tr = build_features_windowed(
            seqs_train, ver_train, y_train, top_ids, disc_ids, top_ngrams, ver_cols,
            WINDOW_LEN, WINDOW_STRIDE, CFG['ngram_n'], cat_cols, id_to_cat,
            max_windows=MAX_WINDOWS_PER_SPLIT, rng_seed=GLOBAL_SEED)
        del seqs_train, ver_train
        X_val,        y_val,   d_va = build_features_windowed(
            seqs_val,   ver_val,   y_val,   top_ids, disc_ids, top_ngrams, ver_cols,
            WINDOW_LEN, WINDOW_STRIDE, CFG['ngram_n'], cat_cols, id_to_cat,
            max_windows=MAX_WINDOWS_PER_SPLIT, rng_seed=GLOBAL_SEED + 1)
        del seqs_val, ver_val
        X_test,       y_test,  d_te = build_features_windowed(
            seqs_test,  ver_test,  y_test,  top_ids, disc_ids, top_ngrams, ver_cols,
            WINDOW_LEN, WINDOW_STRIDE, CFG['ngram_n'], cat_cols, id_to_cat,
            max_windows=MAX_WINDOWS_PER_SPLIT, rng_seed=GLOBAL_SEED + 2)
        del seqs_test, ver_test
        # The originals from load_sequences are now redundant.
        del seqs, ver_list, labels, splits
        print(f'  before traces:   train={before[0]:,}  val={before[1]:,}  test={before[2]:,}')
        print(f'  after windows:   train={X_train_full.shape[0]:,}  val={X_val.shape[0]:,}  test={X_test.shape[0]:,}')
        print(f'  dropped (shorter than window): train={d_tr}  val={d_va}  test={d_te}')
        X_val  = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
    else:
        print('Building feature matrices...')
        X_train_full = build_features(seqs_train, ver_train, top_ids, disc_ids, top_ngrams, ver_cols,
                                      CFG['ngram_n'], cat_cols, id_to_cat)
        X_val        = build_features(seqs_val,   ver_val,   top_ids, disc_ids, top_ngrams, ver_cols,
                                      CFG['ngram_n'], cat_cols, id_to_cat).astype(np.float32)
        X_test       = build_features(seqs_test,  ver_test,  top_ids, disc_ids, top_ngrams, ver_cols,
                                      CFG['ngram_n'], cat_cols, id_to_cat).astype(np.float32)

    # Step 3c
    # Fit scaler on the normal-only subset, then drop X_train_full to free ~1 GB
    # before any transforms allocate float64 temporaries.
    X_train_norm = X_train_full[y_train == 0].astype(np.float32)
    del X_train_full
    feature_scaler = RobustScaler(quantile_range=(1.0, 99.0))
    feature_scaler.fit(X_train_norm)

    # Cast scaler params to float32 so subsequent arithmetic stays in float32
    # (sklearn would otherwise upcast to float64 — doubling peak memory). Then
    # do (X − center) / scale IN-PLACE on each split so no temporaries are
    # allocated. Math is identical to feature_scaler.transform().
    feature_scaler.center_ = feature_scaler.center_.astype(np.float32)
    feature_scaler.scale_  = feature_scaler.scale_.astype(np.float32)

    def _scale_inplace(X):
        X -= feature_scaler.center_
        X /= feature_scaler.scale_
        return X
    X_train = _scale_inplace(X_train_norm)   # rename — same array, now scaled
    X_val   = _scale_inplace(X_val)
    X_test  = _scale_inplace(X_test)

    joblib.dump(feature_scaler, out / 'feature_scaler.joblib')
    input_dim = feat_dim
    fe_report: dict = {
        'feature_groups': {
            f'freq_{len(top_ids)}': len(top_ids),
            f'disc_{len(disc_ids)}': len(disc_ids),
            'stats_8': 8,
            f'ngram_{n_ng}': n_ng,
            'ver_onehot': n_ver,
        },
        'total_dims': feat_dim,
        'ngram_n': CFG['ngram_n'],
        'top_ids': top_ids,
        'disc_ids': disc_ids,
        'top_ngrams': [list(g) for g in top_ngrams],
        'ver_cols': ver_cols,
    }
    if n_cat > 0:
        fe_report['feature_groups'][f'cat_{n_cat}'] = n_cat
        fe_report['cat_cols']         = cat_cols
        # Serialise as {syscall_id: category}; keys are stringified for JSON.
        fe_report['syscall_id_to_cat'] = {str(k): v for k, v in (id_to_cat or {}).items()}
    (out / 'fe_report.json').write_text(json.dumps(fe_report, indent=2))
    _log('feature_scaler.joblib + fe_report.json saved')

    # Step 4: Isolation Forest
    _log(f'Training IsolationForest: {CFG["n_estimators"]} trees, n_train={X_train.shape[0]:,}  rows...')
    iforest = IsolationForest(
        n_estimators=CFG['n_estimators'],
        contamination=CFG['contamination'],
        random_state=GLOBAL_SEED,
        n_jobs=-1,
        verbose=1,   # 1 line per (n_estimators/10) trees built
    )
    iforest.fit(X_train)
    _log('IF: fit done; scoring train...')
    if_train = -iforest.decision_function(X_train)
    _log('IF: scoring val...')
    if_val   = -iforest.decision_function(X_val)
    _log('IF: scoring test...')
    if_test  = -iforest.decision_function(X_test)
    _log(f'IF val AUC: {roc_auc_score(y_val, if_val):.4f}  (paper: 0.646)')
    joblib.dump(iforest, out / 'iforest.joblib')

    # Step 5: VAE
    _log(f'Training VAE with seeds {CFG["seeds"]}  (epochs ≤ {CFG["epochs"]}, batch={CFG["batch_size"]})')
    seed_reports = []
    best_val_auc, best_enc_w, best_dec_w = -1.0, None, None
    best_seed = CFG['seeds'][0]

    for seed in CFG['seeds']:
        _log(f'  VAE seed={seed}: fitting on {X_train.shape[0]:,} rows...')
        set_seeds(seed)
        encoder, decoder, vae = build_vae(
            input_dim, CFG['latent_dim'], CFG['hidden_dim'], CFG['dropout'], CFG['l2'])
        vae.compile(optimizer=keras.optimizers.Adam(CFG['lr']), loss='mse')
        hist = vae.fit(
            X_train, X_train,
            epochs=CFG['epochs'], batch_size=CFG['batch_size'],
            validation_split=0.05,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=2,   # 1 line per epoch with loss / val_loss
        )
        val_auc    = float(roc_auc_score(y_val, recon_error(X_val, encoder, decoder)))
        best_loss  = float(min(hist.history['val_loss']))
        epochs_run = len(hist.history['loss'])
        _log(f'  seed={seed}  val_auc={val_auc:.4f}  best_val_loss={best_loss:.4f}  epochs={epochs_run}')
        seed_reports.append({'seed': seed, 'val_auc': val_auc,
                             'epochs': epochs_run, 'best_val_loss': best_loss})
        if val_auc > best_val_auc:
            best_val_auc = val_auc; best_seed = seed
            best_enc_w = encoder.get_weights(); best_dec_w = decoder.get_weights()

    set_seeds(best_seed)
    encoder, decoder, _ = build_vae(
        input_dim, CFG['latent_dim'], CFG['hidden_dim'], CFG['dropout'], CFG['l2'])
    encoder.set_weights(best_enc_w); decoder.set_weights(best_dec_w)
    _log(f'Best seed={best_seed}  val_auc={best_val_auc:.4f}  (paper: 0.747)')

    _log('VAE: scoring train (recon error)...')
    vae_train = recon_error(X_train, encoder, decoder)
    _log('VAE: scoring val...')
    vae_val   = recon_error(X_val,   encoder, decoder)
    _log('VAE: scoring test...')
    vae_test  = recon_error(X_test,  encoder, decoder)

    # Step 6: Ensemble
    _log('Fitting EnsembleScorer (α sweep)...')
    ensemble = EnsembleScorer(alpha=CFG['alpha'])
    ensemble.fit(vae_train, if_train)
    ens_val  = ensemble.score(vae_val,  if_val)
    ens_test = ensemble.score(vae_test, if_test)
    _log(f'Ensemble val AUC: {roc_auc_score(y_val, ens_val):.4f}  (paper: 0.656)')

    # Step 7: Threshold + metrics
    _log(f'Selecting thresholds (strategy={CFG["threshold"]}) and computing test metrics...')
    strategy = CFG['threshold']
    t_if  = select_threshold(y_val, if_val,  strategy)
    t_vae = select_threshold(y_val, vae_val, strategy)
    t_ens = select_threshold(y_val, ens_val, strategy)
    m_if  = compute_metrics(y_test, if_test,  t_if)
    m_vae = compute_metrics(y_test, vae_test, t_vae)
    m_ens = compute_metrics(y_test, ens_test, t_ens)

    hdr = f'{"Model":<22} {"AUC":>6} {"AP":>6} {"F1":>6} {"Prec":>6} {"Recall":>7} {"FPR":>6}'
    print(hdr); print('-' * 70)
    for lbl, m in [('Isolation Forest', m_if), ('VAE', m_vae), ('Ensemble', m_ens)]:
        print(f'{lbl:<22} {m["auc_roc"]:>6.3f} {m["ap"]:>6.3f} {m["f1"]:>6.3f} '
              f'{m["precision"]:>6.3f} {m["recall_tpr"]:>7.3f} {m["fpr"]:>6.3f}')

    # Step 8: Save artifacts
    joblib.dump(iforest, out / 'iforest.joblib')
    encoder.save_weights(str(out / 'vae_encoder.weights.h5'))
    decoder.save_weights(str(out / 'vae_decoder.weights.h5'))
    ensemble.save(out)

    results = {
        'experiment':       (f'dongting_sliding_window_{WINDOW_LEN}_{WINDOW_STRIDE}'
                             if USE_SLIDING_WINDOW else 'dongting_build_and_train'),
        'use_sliding_window': bool(USE_SLIDING_WINDOW),
        'window_len':       WINDOW_LEN if USE_SLIDING_WINDOW else None,
        'window_stride':    WINDOW_STRIDE if USE_SLIDING_WINDOW else None,
        'n_train':          int(X_train.shape[0]),
        'n_val':            int(X_val.shape[0]),
        'n_test':           int(X_test.shape[0]),
        'feat_dim':         input_dim,
        'latent_dim':       CFG['latent_dim'],
        'hidden_dim':       CFG['hidden_dim'],
        'contamination':    CFG['contamination'],
        'ngram_n':          CFG['ngram_n'],
        'vae_seed_reports': seed_reports,
        'best_vae_seed':    best_seed,
        'models': {
            'isolation_forest': m_if,
            'vae':              m_vae,
            'ensemble':         m_ens,
        },
    }
    (out / 'results.json').write_text(json.dumps(results, indent=2))

    print(f'\nArtifacts in {OUTPUT_DIR}:')
    for f in sorted(out.iterdir()):
        print(f'  {f.name:<48} {f.stat().st_size // 1024:>6} KB')


if __name__ == '__main__':
    main()
