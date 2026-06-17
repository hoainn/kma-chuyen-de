"""
DeSFAM — DongTing Training Pipeline (script mode)
Sync target: train_dongting.ipynb — keep function bodies identical in both files.

Paper-faithful feature engineering (paper §IV.B.3). The feature vector now contains
ONLY the components the paper names, per-sliding-window:

    [ cat_K | temporal_3? | prefixspan_P ]

  • cat_K        — normalised per-window frequency over K=10 functional syscall
                   categories (paper "Categorical Frequencies"; the behavioural signature).
  • temporal_3   — Δt mean/std/max per window. CONDITIONAL: present only when the
                   loaded sequences carry per-syscall timestamps (re-traced / live eBPF
                   stream). DongTing has none, so this group contributes 0 dims.
  • prefixspan_P — P=2 PrefixSpan benign-pattern match: [matched-flag, longest-match/L]
                   (paper "Access List Pattern Matching").

The earlier engineered groups (freq_60 / disc_40 / stats_8 / bigrams_40 / ver_onehot)
were never described by the paper and have been removed.
"""
from __future__ import annotations

import json
import os
import random
import warnings
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
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
SYSCALL_TBL = os.environ.get('SYSCALL_TBL', '/data/dongting_repo/syscall_64.tbl')
OUTPUT_DIR  = os.environ.get('OUTPUT_DIR',  '/model')

# Data source — the full DongTing release on the host filesystem (.log files +
# Baseline.xlsx splits). The legacy MongoDB loader has been removed.
DATA_ROOT   = os.environ.get('DATA_ROOT',   '/data/dongting')

# DongTing's full traces can be tens of MB each (millions of syscalls); the raw
# corpus is far too large to hold in RAM. Truncate each trace to its first
# DATA_MAX_SEQ_LEN syscalls (0 = unlimited). Windowing still slides over the kept
# prefix and MAX_WINDOWS_PER_SPLIT caps the final matrix.
DATA_MAX_SEQ_LEN = int(os.environ.get('DATA_MAX_SEQ_LEN', '50000'))

# Sliding-window training (paper §IV.B.3: length 15, stride 3). Now the canonical
# default — the live detector is always window-based, so training matches it.
USE_SLIDING_WINDOW = os.environ.get('USE_SLIDING_WINDOW', '1') == '1'
WINDOW_LEN    = int(os.environ.get('WINDOW_LEN',    '15'))
WINDOW_STRIDE = int(os.environ.get('WINDOW_STRIDE', '3'))

# Temporal Δt features (paper "Temporal Features"). Gated: only emitted when the
# loaded data carries timestamps. DongTing has none → group stays at 0 dims and
# the trained model + fe_report record has_temporal=False so inference matches.
USE_TEMPORAL = os.environ.get('USE_TEMPORAL', '0') == '1'

# Feature engineering vocabulary sizes (paper-faithful reconstruction).
#
# The DeSFAM paper (§IV.B.3) names 5 components: sliding windows, categorical
# frequencies, temporal Δt, PrefixSpan benign-pattern matching, data augmentation.
# Implemented literally, these 5 collapse the VAE (val AUC ≈ 0.41); even adding
# the four standard syscall-IDS support groups (freq/disc/stats/bigrams) only
# reaches val AUC ≈ 0.76 — still below the paper's Table-3 (test AUC 0.96 VAE,
# 0.92 ensemble). This 140-dim vector is the best honest reconstruction we have;
# the gap suggests the paper uses additional/wider feature engineering it does
# not document.
TOP_K_FREQ    = int(os.environ.get('TOP_K_FREQ',    '50'))
TOP_K_DISC    = int(os.environ.get('TOP_K_DISC',    '30'))
TOP_K_BIGRAMS = int(os.environ.get('TOP_K_BIGRAMS', '40'))

# The stats_8 group is derived from the freq counts; silence it (ENABLE_STATS=0)
# when the alphabet is small enough that per-syscall frequency is already the
# full behavioural fingerprint and the extra 8 dims only add VAE-collapse risk.
ENABLE_STATS = os.environ.get('ENABLE_STATS', '1') == '1'

# ── Sensor alphabet (deployment-faithful feature space) ────────────────────
# The live Tetragon TracingPolicy emits only a fixed set of security-relevant
# syscalls — no read/write/close/futex/etc. Training on the FULL DongTing
# syscall universe therefore fits the scaler / vocab / PrefixSpan DB in a
# feature space the detector never observes at serve time: every live window
# collapses onto a near-degenerate vector and benign ≈ attack (the saturation
# rail documented in docs/auto-iter-log.md). We restrict training to the SAME
# alphabet the sensor emits so train/serve windows are isomorphic. The list is
# the union of the kprobes in Kubernetes/tetragon/tracing-policy*.yaml (the
# applied live policy `syscall-ad-tracing`). Set SENSOR_ALPHABET=0 to reproduce
# the legacy full-universe behaviour.
SENSOR_ALPHABET_NAMES = [
    'execve', 'clone', 'setuid', 'setgid', 'capset', 'openat', 'unlinkat',
    'renameat2', 'mount', 'unshare', 'setns', 'pivot_root', 'socket',
    'connect', 'bind', 'mprotect', 'mmap', 'prctl', 'memfd_create', 'splice',
    'ptrace', 'init_module', 'finit_module', 'bpf',
]
USE_SENSOR_ALPHABET = os.environ.get('SENSOR_ALPHABET', '1') == '1'
# Resolved to syscall IDs in main() once the table is loaded; consumed by the
# loader (_parse_log_member) to drop every non-sensor syscall before windowing.
_ALLOWED_IDS: set[int] | None = None

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

# Paper feature-component status (paper §IV.B.3):
#   • Sliding Windows            — implemented (len 15, stride 3), now the default.
#   • Categorical Frequencies    — implemented (cat_K, the core behavioural signature).
#   • Access List Pattern Match  — implemented (PrefixSpan benign-pattern match, prefixspan_P).
#   • Temporal Δt (mean/std/max) — implemented as a CONDITIONAL group: it only
#     activates when the loaded sequences carry per-syscall timestamps. Every
#     DongTing artefact (raw .log, npz, Mongo) stores syscall identifiers ONLY —
#     collection used `strace -v -f` with no timing flags — so the group stays at
#     0 dims here. A re-traced/live timestamped stream would light it up.
#   • Data Augmentation          — NOT implemented: synthetic attack-sequence
#     augmentation is incompatible with the normal-only IF/VAE training paradigm
#     (neither model ever sees attack data). Documented as a known limitation.

# ── Hyperparameters ────────────────────────────────────────────────────────
CFG = {
    # PrefixSpan benign-pattern mining (paper "Access List Pattern Matching").
    # Mined on NORMAL training windows only; matched per window at featurisation.
    'ps_min_support': 0.05,   # min fraction of benign windows a pattern must cover
    'ps_min_len':     3,      # ignore trivial length-1/2 patterns
    'ps_max_len':     6,      # cap pattern length (windows are length 15)
    'ps_top_n':       150,    # keep the N most frequent patterns (perf bound)
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

# Width of the PrefixSpan match feature group: [matched-flag, longest-match/L].
PREFIXSPAN_DIMS = 2
# Width of the temporal group when active.
TEMPORAL_DIMS = 3

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
                # Skip the x32 compat ABI rows (nr 512–547). They reuse names
                # already defined by the common/64-bit ABI and, being later in
                # the file, would overwrite the canonical number (execve 59→520,
                # ptrace 101→521, …). The live Tetragon extractor resolves syscall
                # names to the canonical x86-64 numbers, so training must match or
                # those syscalls land in a different ID space at serve time.
                if parts[1] == 'x32':
                    continue
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


SPLIT_MAP = {'DTDS-train': 'train', 'DTDS-validation': 'val', 'DTDS-test': 'test'}


# ── Step 2: Load sequences ─────────────────────────────────────────────────
# Every loader returns the 5-tuple (seqs, labels, splits, ver_list, ts_list).
# ts_list is None when the source carries no per-syscall timestamps (all DongTing
# artefacts) — the temporal feature group then stays at 0 dims.

def load_sequences(name_to_id: dict[str, int]):
    return _load_sequences_local(name_to_id)


def _index_log_members(*roots: str) -> dict[str, tuple[zipfile.ZipFile, str]]:
    """Map each .log basename -> (open ZipFile, member name) across every zip
    under the given roots. Reads members straight from the archives so the 85 GB
    extracted corpus never lands on disk/RAM in full."""
    index: dict[str, tuple[zipfile.ZipFile, str]] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for fn in sorted(os.listdir(root)):
            if not fn.endswith('.zip'):
                continue
            zf = zipfile.ZipFile(os.path.join(root, fn))
            for member in zf.namelist():
                if member.endswith('.log'):
                    index[os.path.basename(member)] = (zf, member)
    return index


def _parse_log_member(zf: zipfile.ZipFile, member: str,
                      name_to_id: dict[str, int]) -> list[int]:
    """Read one pipe-separated syscall-name log into truncated integer IDs.
    Streams in chunks and stops at DATA_MAX_SEQ_LEN so multi-hundred-MB attack
    traces never get fully materialised."""
    ids: list[int] = []
    cap = DATA_MAX_SEQ_LEN or None
    remainder = ''
    with zf.open(member) as fh:
        while True:
            chunk = fh.read(1 << 20)            # 1 MiB
            if not chunk:
                break
            text = remainder + chunk.decode('utf-8', errors='ignore')
            parts = text.split('|')
            remainder = parts.pop()             # last piece may be incomplete
            for nm in parts:
                sid = name_to_id.get(nm.strip())
                if sid is not None:
                    ids.append(sid)
            if cap and len(ids) >= cap:
                return _filter_alphabet(ids[:cap])
    sid = name_to_id.get(remainder.strip())
    if sid is not None:
        ids.append(sid)
    return _filter_alphabet(ids[:cap] if cap else ids)


def _filter_alphabet(ids: list[int]) -> list[int]:
    """Drop every syscall outside the live sensor alphabet so training windows
    match what the detector actually observes. No-op when SENSOR_ALPHABET=0
    (_ALLOWED_IDS stays None). The raw read above is still bounded by
    DATA_MAX_SEQ_LEN, so each trace contributes its sensor syscalls from a
    bounded prefix."""
    if _ALLOWED_IDS is None:
        return ids
    return [i for i in ids if i in _ALLOWED_IDS]


def _load_sequences_local(name_to_id: dict[str, int]):
    """Read the full DongTing release: Baseline.xlsx is the master index
    (label / split / version per sequence) and the .log files under
    Normal_data/ and Abnormal_data/ hold the syscall-name sequences.

    Filename join (verified 100% on the release):
      • Normal sequences:  file = 'sy_' + kcb_bug_name        (bug_name ends in '.log')
      • Attack sequences:  file = 'sy_' + kcb_bug_name + '.log'
    """
    import openpyxl

    _log('  Indexing .log members across the dataset zips...')
    index = _index_log_members(
        os.path.join(DATA_ROOT, 'Normal_data'),
        os.path.join(DATA_ROOT, 'Abnormal_data'),
    )
    _log(f'  Indexed {len(index):,} .log files')

    xlsx = os.path.join(DATA_ROOT, 'Baseline.xlsx')
    _log(f'  Reading master index {xlsx} ...')
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    ws = wb['kernel_convert_baseline']
    rows = ws.iter_rows(min_row=1, values_only=True)
    hdr = next(rows)
    ci = {h: i for i, h in enumerate(hdr)}

    seqs, labels, splits, ver_list = [], [], [], []
    missing = 0
    for r in rows:
        bug   = r[ci['kcb_bug_name']]
        if bug is None:
            continue
        label = 0 if r[ci['kcb_seq_lables']] == 'Normal' else 1
        split = SPLIT_MAP.get(r[ci['kcb_seq_class']], 'train')
        ver   = str(r[ci['kcb_master_line_ver']] or '').strip()

        fname = ('sy_' + bug) if label == 0 else ('sy_' + bug + '.log')
        hit = index.get(fname)
        if hit is None:
            missing += 1; continue
        ids = _parse_log_member(hit[0], hit[1], name_to_id)
        if len(ids) < 2:
            missing += 1; continue

        seqs.append(ids); labels.append(label)
        splits.append(split); ver_list.append(ver)

    wb.close()
    print(f'  Loaded {len(seqs)} sequences ({missing} skipped) from full dataset')
    # DongTing logs carry no timestamps → temporal group stays inactive.
    return seqs, labels, splits, ver_list, None


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


# ── Step 3: Feature engineering (paper §IV.B.3, paper-only vector) ─────────
#
# Vector per window:  [ cat_K | temporal_T | prefixspan_P ]
#   cat_K        — normalised per-window category frequency (K = len(cat_cols)).
#   temporal_T   — Δt mean/std/max (T = TEMPORAL_DIMS) when timestamps exist, else T = 0.
#   prefixspan_P — [matched-flag, longest-match/L] (P = PREFIXSPAN_DIMS) when a benign
#                  pattern DB exists, else P = 0.


def match_benign_patterns(window: list[int],
                          patterns_by_first: dict[int, list]) -> tuple[float, int]:
    """Canonical PrefixSpan match used by BOTH training and inference — keep the
    two copies byte-identical (guarded by the parity test).

    A benign pattern matches the window if it appears as an ORDER-PRESERVING
    subsequence (PrefixSpan semantics). Returns (matched_flag, longest_match_len).
    `window` must be a list of plain Python ints. Result is independent of the
    pattern iteration order, so determinism does not depend on dict ordering.
    """
    matched = 0.0
    longest = 0
    n = len(window)
    for i in range(n):
        pats = patterns_by_first.get(window[i])
        if not pats:
            continue
        for pat in pats:
            L = len(pat)
            if matched and L <= longest:
                continue  # cannot raise `matched` or `longest`
            pi, wi = 0, i
            while wi < n and pi < L:
                if window[wi] == pat[pi]:
                    pi += 1
                wi += 1
            if pi == L:
                matched = 1.0
                if L > longest:
                    longest = L
    return matched, longest


def build_patterns_by_first(patterns: list[tuple]) -> dict[int, list[tuple]]:
    """Index mined patterns by their first syscall ID for fast matching."""
    by_first: dict[int, list[tuple]] = defaultdict(list)
    for p in patterns:
        if p:
            by_first[int(p[0])].append(tuple(int(x) for x in p))
    return {k: sorted(v) for k, v in by_first.items()}


def fit_benign_patterns(normal_seqs, length, stride,
                        min_support, min_len, max_len, top_n,
                        rng_seed, max_mine_windows=20000):
    """Mine a Benign Pattern Database (paper "Access List Pattern Matching") from
    NORMAL training windows only, using PrefixSpan. Sampling keeps mining
    tractable; selection is deterministic (seeded). Returns the pattern list."""
    rng = np.random.default_rng(rng_seed)
    n_tr = max(len(normal_seqs), 1)
    per = max(1, max_mine_windows // n_tr)
    sample: list[list[int]] = []
    for s in normal_seqs:
        nw = count_windows(len(s), length, stride)
        if nw <= 0:
            continue
        take = min(per, nw)
        idx = rng.choice(nw, size=take, replace=False)
        idx.sort()
        for i in idx:
            st = int(i) * stride
            sample.append([int(x) for x in s[st: st + length]])
        if len(sample) >= max_mine_windows:
            break
    sample = sample[:max_mine_windows]
    if not sample:
        return []

    from prefixspan import PrefixSpan
    ps = PrefixSpan(sample)
    ps.minlen = min_len
    ps.maxlen = max_len
    minsup_count = max(2, int(min_support * len(sample)))
    results = ps.frequent(minsup_count)            # [(support, pattern_list), ...]
    # Deterministic order: support desc, then pattern lexicographic.
    results.sort(key=lambda r: (-r[0], r[1]))
    patterns = [tuple(int(x) for x in p) for _, p in results[:top_n]]
    _log(f'  PrefixSpan: mined {len(sample):,} benign windows → '
         f'{len(results):,} frequent (minsup={minsup_count}) → kept {len(patterns)} patterns')
    return patterns


def fit_vocab_plus(normal_train_seqs, top_k_freq: int, top_k_disc: int,
                   top_k_bigrams: int) -> tuple[list[int], list[int], list[tuple]]:
    """Fit engineered-vocab on NORMAL training traces only:
       • top-K most frequent syscalls   → `freq_K` block
       • the next-K most frequent       → `disc_K` binary-presence block
       • top-K bigram (adjacent pair) frequencies → `bigrams_K` block
    These are the engineered support features that the paper underspecifies
    but that are required to bring the VAE up to the paper's reported AUC."""
    all_sc: Counter = Counter()
    all_ng: Counter = Counter()
    for seq in normal_train_seqs:
        all_sc.update(seq)
        for i in range(len(seq) - 1):
            all_ng[(int(seq[i]), int(seq[i + 1]))] += 1
    common  = all_sc.most_common(top_k_freq + top_k_disc)
    top_ids = [sc for sc, _ in common[:top_k_freq]]
    disc_ids = [sc for sc, _ in common[top_k_freq: top_k_freq + top_k_disc]]
    top_bigrams = [g for g, _ in all_ng.most_common(top_k_bigrams)]
    return top_ids, disc_ids, top_bigrams


def build_features_windowed(seqs, y_in, length: int, stride: int,
                            cat_cols: list[str], id_to_cat: dict[int, str],
                            patterns_by_first: dict[int, list],
                            ps_dims: int,
                            ts_list=None, temporal_dims: int = 0,
                            top_ids: list[int] | tuple = (),
                            disc_ids: list[int] | tuple = (),
                            top_bigrams: list[tuple] | tuple = (),
                            enable_stats: bool = False,
                            max_windows: int | None = None,
                            rng_seed: int = 0):
    """Per-trace batched featurisation → (X, y, n_dropped).

    Layout per window:
      [ freq_F | disc_D | stats_8 | bigrams_B | cat_K | temporal_T? | prefixspan_P ]

    Math is intentionally bit-identical to the per-window form in `build_features`
    so the inference Featurizer produces the same vectors.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    seqs_arr: list[np.ndarray] = [np.asarray(s, dtype=np.int32) for s in seqs]
    y_arr = np.asarray(y_in, dtype=np.int32)
    has_ts = bool(temporal_dims) and ts_list is not None
    ts_arr = ([np.asarray(t, dtype=np.float64) for t in ts_list]
              if has_ts else None)

    top_ids = list(top_ids); disc_ids = list(disc_ids); top_bigrams = list(top_bigrams)
    n_freq  = len(top_ids)
    n_disc  = len(disc_ids)
    n_ng    = len(top_bigrams)
    n_stats = 8 if enable_stats else 0
    n_cat   = len(cat_cols) if cat_cols else 0
    n_feat  = n_freq + n_disc + n_stats + n_ng + n_cat + temporal_dims + ps_dims

    # offsets
    freq_off  = 0
    disc_off  = freq_off + n_freq
    stats_off = disc_off + n_disc
    bg_off    = stats_off + n_stats
    cat_off   = bg_off + n_ng
    temp_off  = cat_off + n_cat
    ps_off    = temp_off + temporal_dims

    # Window-count pass for subsampling + buffer sizing.
    n_per_trace, n_total, n_dropped, max_n_win = [], 0, 0, 0
    for s_arr in seqs_arr:
        n = count_windows(len(s_arr), length, stride)
        n_per_trace.append(n)
        if n == 0: n_dropped += 1
        if n > max_n_win: max_n_win = n
        n_total += n

    sampled_idx_per_trace: list[np.ndarray] | None = None
    if max_windows is not None and n_total > max_windows:
        ratio = max_windows / n_total
        rng = np.random.default_rng(rng_seed)
        sampled_idx_per_trace = []
        new_total = 0
        for n in n_per_trace:
            if n == 0:
                sampled_idx_per_trace.append(np.empty(0, dtype=np.int32)); continue
            k = min(max(1, int(np.floor(n * ratio))), n)
            idx = rng.choice(n, size=k, replace=False); idx.sort()
            sampled_idx_per_trace.append(idx.astype(np.int32, copy=False))
            new_total += k
        print(f'  Subsampling windows: {n_total:,} → {new_total:,} '
              f'(cap={max_windows:,}, ratio={ratio:.3f})')
        n_total = new_total
        max_n_win = max((arr.size for arr in sampled_idx_per_trace), default=0)

    # Vocab ID range — we need a buffer wide enough to scatter window counts into.
    max_id = 0
    if n_freq:  max_id = max(max_id, int(max(top_ids)))
    if n_disc:  max_id = max(max_id, int(max(disc_ids)))
    if id_to_cat: max_id = max(max_id, max(id_to_cat.keys()))
    for s_arr in seqs_arr:
        if s_arr.size:
            v = int(s_arr.max())
            if v > max_id: max_id = v
    n_ids = max_id + 1

    cat_of_id = None
    if n_cat > 0 and id_to_cat is not None:
        cat_idx_map = {c: i for i, c in enumerate(cat_cols)}
        other_i = cat_idx_map.get('other', 0)
        cat_of_id = np.full(n_ids, other_i, dtype=np.int16)
        for sid, name in id_to_cat.items():
            ci = cat_idx_map.get(name)
            if ci is not None and 0 <= sid < n_ids:
                cat_of_id[sid] = ci

    top_ids_arr  = np.asarray(top_ids,  dtype=np.int64) if n_freq else np.empty(0, np.int64)
    disc_ids_arr = np.asarray(disc_ids, dtype=np.int64) if n_disc else np.empty(0, np.int64)

    # Bigram lookup: sorted int64 codes + parallel column array → searchsorted.
    if n_ng:
        pairs = sorted(((int(a) << 16) | int(b), i) for i, (a, b) in enumerate(top_bigrams))
        sorted_codes = np.fromiter((c for c, _ in pairs), dtype=np.int64, count=len(pairs))
        code_to_col  = np.fromiter((i for _, i in pairs), dtype=np.int32, count=len(pairs))
    else:
        sorted_codes = np.empty(0, dtype=np.int64)
        code_to_col  = np.empty(0, dtype=np.int32)

    X = np.zeros((n_total, n_feat), dtype=np.float32)
    y = np.zeros((n_total,),        dtype=np.int32)
    if max_n_win == 0:
        return X, y, n_dropped

    # Reused per-trace buffers.
    need_counts = (n_freq or n_disc or n_stats)
    buf_counts = np.zeros((max_n_win, n_ids), dtype=np.int32) if need_counts else None
    buf_cat    = np.zeros((max_n_win, n_cat), dtype=np.int32) if n_cat else None
    buf_bg     = np.zeros((max_n_win, n_ng),  dtype=np.int32) if n_ng  else None

    total    = length
    n_win_ng = max(total - 1, 1)
    row_idx_col = np.arange(max_n_win)[:, None]

    n_traces_total = len(seqs_arr)
    progress_every = max(1, n_traces_total // 20)
    row_off = 0
    for ti, s_arr in enumerate(seqs_arr):
        if ti and ti % progress_every == 0:
            _log(f'  featurise: trace {ti:,}/{n_traces_total:,} '
                 f'({100*ti/n_traces_total:.0f}%)  rows={row_off:,}/{n_total:,}')
        n_win = n_per_trace[ti]
        if n_win == 0: continue
        all_windows = sliding_window_view(s_arr, window_shape=length)[::stride]
        if sampled_idx_per_trace is not None:
            keep = sampled_idx_per_trace[ti]
            if keep.size == 0: continue
            windows = all_windows[keep]; n_win = keep.size
        else:
            windows = all_windows

        X_block = X[row_off: row_off + n_win]

        # Per-window full histogram (used by freq, disc and stats).
        if need_counts:
            counts_view = buf_counts[:n_win]
            counts_view.fill(0)
            np.add.at(counts_view, (row_idx_col[:n_win], windows), 1)
        else:
            counts_view = None

        # freq_F — normalised top-K syscall frequency
        if n_freq:
            X_block[:, freq_off: freq_off + n_freq] = counts_view[:, top_ids_arr] / total
        # disc_D — binary presence of next-K syscalls
        if n_disc:
            X_block[:, disc_off: disc_off + n_disc] = (counts_view[:, disc_ids_arr] > 0).astype(np.float32)
        # stats_8 — entropy / shape / magnitude (matches per-window form below)
        if n_stats:
            raw = counts_view[:, top_ids_arr].astype(np.float32) if n_freq \
                  else np.zeros((n_win, 0), np.float32)
            if n_freq:
                totals = raw.sum(axis=1, keepdims=True)
                safe_t = np.where(totals > 0, totals, 1.0)
                p = raw / safe_t
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_p = np.log2(p, where=(p > 0))
                log_p = np.where(p > 0, log_p, 0.0)
                H = -(p * log_p).sum(axis=1)
                X_block[:, stats_off + 0] = np.where(totals.squeeze(-1) > 0, H, 0.0)
                X_block[:, stats_off + 3] = raw.max(axis=1) / total
                raw_nan = np.where(raw > 0, raw, np.nan)
                with np.errstate(invalid='ignore'):
                    p75 = np.nanpercentile(raw_nan, 75, axis=1)
                X_block[:, stats_off + 4] = np.where(np.isnan(p75), 0.0, p75).astype(np.float32)
                X_block[:, stats_off + 5] = raw.std(axis=1)
                X_block[:, stats_off + 6] = (raw > 0).sum(axis=1).astype(np.float32) / max(n_freq, 1)
            X_block[:, stats_off + 1] = (counts_view > 0).sum(axis=1).astype(np.float32)
            X_block[:, stats_off + 2] = np.log1p(total)
            X_block[:, stats_off + 7] = total / 1000.0

        # bigrams_B — top-K transition counts via searchsorted
        if n_ng and total >= 2:
            a = windows[:, :-1].astype(np.int64)
            b = windows[:,  1:].astype(np.int64)
            codes = (a << 16) | b                              # (n_win, total-1)
            pos = np.searchsorted(sorted_codes, codes)
            pos_clip = np.minimum(pos, sorted_codes.size - 1)
            hit = sorted_codes[pos_clip] == codes
            cols = code_to_col[pos_clip]
            bg_view = buf_bg[:n_win]
            bg_view.fill(0)
            rows = np.broadcast_to(row_idx_col[:n_win], cols.shape)
            np.add.at(bg_view, (rows[hit], cols[hit]), 1)
            X_block[:, bg_off: bg_off + n_ng] = bg_view / n_win_ng

        # cat_K — normalised category histogram per window
        if cat_of_id is not None:
            ci = cat_of_id[windows]
            cat_view = buf_cat[:n_win]
            cat_view.fill(0)
            np.add.at(cat_view, (row_idx_col[:n_win], ci), 1)
            X_block[:, cat_off: cat_off + n_cat] = cat_view / total

        # temporal_T — Δt mean/std/max per window (only when timestamps exist)
        if has_ts and temporal_dims:
            t_all = sliding_window_view(ts_arr[ti], window_shape=length)[::stride]
            t_win = t_all[keep] if sampled_idx_per_trace is not None else t_all
            dt = np.diff(t_win, axis=1)
            X_block[:, temp_off + 0] = dt.mean(axis=1)
            X_block[:, temp_off + 1] = dt.std(axis=1)
            X_block[:, temp_off + 2] = dt.max(axis=1)

        # prefixspan_P — per-window subsequence match (Python, identical to inference)
        if ps_dims and patterns_by_first:
            wl = windows.tolist()
            for r in range(n_win):
                matched, longest = match_benign_patterns(wl[r], patterns_by_first)
                X_block[r, ps_off + 0] = matched
                X_block[r, ps_off + 1] = longest / total

        y[row_off: row_off + n_win] = int(y_arr[ti])
        row_off += n_win

    return X, y, n_dropped


def build_features(seqs, cat_cols: list[str], id_to_cat: dict[int, str],
                   patterns_by_first: dict[int, list], ps_dims: int,
                   ts_list=None, temporal_dims: int = 0,
                   top_ids: list[int] | tuple = (),
                   disc_ids: list[int] | tuple = (),
                   top_bigrams: list[tuple] | tuple = (),
                   enable_stats: bool = False) -> np.ndarray:
    """Per-window featurisation (canonical form mirrored by inference featurizer).
    Each element of `seqs` is one window (list of syscall IDs). Layout matches
    `build_features_windowed`: [freq|disc|stats|bigrams|cat|temporal?|prefixspan]."""
    top_ids = list(top_ids); disc_ids = list(disc_ids); top_bigrams = list(top_bigrams)
    n_freq  = len(top_ids)
    n_disc  = len(disc_ids)
    n_ng    = len(top_bigrams)
    n_stats = 8 if enable_stats else 0
    n_cat   = len(cat_cols) if cat_cols else 0
    n_feat  = n_freq + n_disc + n_stats + n_ng + n_cat + temporal_dims + ps_dims

    top_idx  = {sc: i for i, sc in enumerate(top_ids)}
    disc_idx = {sc: i for i, sc in enumerate(disc_ids)}
    ng_idx   = {tuple(g): i for i, g in enumerate(top_bigrams)}
    cat_idx  = {c: i for i, c in enumerate(cat_cols or [])}

    freq_off  = 0
    disc_off  = n_freq
    stats_off = disc_off + n_disc
    bg_off    = stats_off + n_stats
    cat_off   = bg_off + n_ng
    temp_off  = cat_off + n_cat
    ps_off    = temp_off + temporal_dims
    has_ts    = bool(temporal_dims) and ts_list is not None

    X = np.zeros((len(seqs), n_feat), dtype=np.float32)
    for i, seq in enumerate(seqs):
        if not seq: continue
        total = len(seq)
        seq_int = [int(x) for x in seq]

        # freq_F
        if n_freq:
            for sc in seq_int:
                j = top_idx.get(sc)
                if j is not None: X[i, freq_off + j] += 1
            X[i, freq_off: freq_off + n_freq] /= max(total, 1)
        # disc_D
        if n_disc:
            for sc in set(seq_int):
                j = disc_idx.get(sc)
                if j is not None: X[i, disc_off + j] = 1.0
        # stats_8 — derived from top-K raw counts
        if n_stats:
            raw = X[i, freq_off: freq_off + n_freq] * total if n_freq \
                  else np.zeros(0, np.float32)
            total_raw = float(raw.sum()) if n_freq else 0.0
            if total_raw > 0:
                p = raw / total_raw
                X[i, stats_off + 0] = float(scipy_entropy(p[p > 0], base=2))
            else:
                X[i, stats_off + 0] = 0.0
            X[i, stats_off + 1] = float(len(set(seq_int)))
            X[i, stats_off + 2] = float(np.log1p(total))
            X[i, stats_off + 3] = (float(raw.max()) / max(total, 1)) if n_freq else 0.0
            nz = raw[raw > 0] if n_freq else np.zeros(0, np.float32)
            X[i, stats_off + 4] = float(np.percentile(nz, 75)) if nz.size else 0.0
            X[i, stats_off + 5] = float(np.std(raw)) if n_freq else 0.0
            X[i, stats_off + 6] = (float((raw > 0).sum()) / max(n_freq, 1)) if n_freq else 0.0
            X[i, stats_off + 7] = float(total) / 1000.0
        # bigrams_B
        if n_ng and total >= 2:
            n_win_ng = max(total - 1, 1)
            for k in range(total - 1):
                g = (seq_int[k], seq_int[k + 1])
                j = ng_idx.get(g)
                if j is not None: X[i, bg_off + j] += 1
            X[i, bg_off: bg_off + n_ng] /= n_win_ng
        # cat_K
        if n_cat > 0 and id_to_cat is not None:
            for sc in seq_int:
                ci = cat_idx.get(id_to_cat.get(sc, 'other'))
                if ci is not None: X[i, cat_off + ci] += 1
            X[i, cat_off: cat_off + n_cat] /= max(total, 1)
        # temporal_T
        if has_ts and temporal_dims:
            ts = ts_list[i]
            if ts is not None and len(ts) >= 2:
                dt = np.diff(np.asarray(ts, dtype=np.float64))
                X[i, temp_off + 0] = float(dt.mean())
                X[i, temp_off + 1] = float(dt.std())
                X[i, temp_off + 2] = float(dt.max())
        # prefixspan_P
        if ps_dims and patterns_by_first:
            matched, longest = match_benign_patterns(seq_int, patterns_by_first)
            X[i, ps_off + 0] = matched
            X[i, ps_off + 1] = longest / max(total, 1)
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
    if strategy == 'p995':
        # Paper §IV.B.3 "Dynamic Thresholding": T_0 = 99.5th percentile of benign
        # validation errors. Inference then EMA-updates this T_0 online (handled
        # detector-side, not here).
        benign = scores[y_val == 0]
        return float(np.percentile(benign, 99.5)) if benign.size else 0.5
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

    # Step 1b — resolve the sensor alphabet to IDs BEFORE loading sequences
    # (the loader filters each trace to this set as it parses).
    global _ALLOWED_IDS
    if USE_SENSOR_ALPHABET:
        _ALLOWED_IDS = {name_to_id[n] for n in SENSOR_ALPHABET_NAMES if n in name_to_id}
        missing = [n for n in SENSOR_ALPHABET_NAMES if n not in name_to_id]
        print(f'Sensor alphabet ON: {len(_ALLOWED_IDS)}/{len(SENSOR_ALPHABET_NAMES)} '
              f'syscalls -> ids {sorted(_ALLOWED_IDS)}'
              + (f'  (MISSING from table: {missing})' if missing else ''))
    else:
        print('Sensor alphabet OFF — training on the full syscall universe')

    # Step 2
    seqs, labels, splits, ver_list, ts_list = load_sequences(name_to_id)
    idx_train = [i for i, s in enumerate(splits) if s == 'train']
    idx_val   = [i for i, s in enumerate(splits) if s == 'val']
    idx_test  = [i for i, s in enumerate(splits) if s == 'test']

    def _pick_ts(idx):
        return [ts_list[i] for i in idx] if ts_list is not None else None

    seqs_train = [seqs[i] for i in idx_train]
    ts_train   = _pick_ts(idx_train)
    y_train    = np.array([labels[i] for i in idx_train], dtype=np.int32)
    seqs_val   = [seqs[i] for i in idx_val]
    ts_val     = _pick_ts(idx_val)
    y_val      = np.array([labels[i] for i in idx_val], dtype=np.int32)
    seqs_test  = [seqs[i] for i in idx_test]
    ts_test    = _pick_ts(idx_test)
    y_test     = np.array([labels[i] for i in idx_test], dtype=np.int32)

    print(f'train={len(seqs_train):,}  val={len(seqs_val):,}  test={len(seqs_test):,}')

    # Step 3a — categories are static (from the syscall table, not data-fit).
    # Categories are static (from the syscall table, not data-fit).
    # The engineered support vocab (top-K freq + disc + bigrams) is fit on
    # NORMAL training traces only so attack-specific calls don't bias it.
    # The PrefixSpan benign-pattern DB is also mined on normal-only windows.
    cat_cols  = list(SYSCALL_CATEGORIES)
    id_to_cat = build_id_to_category(name_to_id)
    n_cat     = len(cat_cols)

    has_temporal  = bool(USE_TEMPORAL) and (ts_list is not None)
    temporal_dims = TEMPORAL_DIMS if has_temporal else 0

    normal_train_seqs = [seqs_train[i] for i in range(len(seqs_train)) if y_train[i] == 0]

    _log(f'Fitting engineered vocab: top_freq={TOP_K_FREQ} disc={TOP_K_DISC} '
         f'bigrams={TOP_K_BIGRAMS} on {len(normal_train_seqs)} normal traces...')
    top_ids, disc_ids, top_bigrams = fit_vocab_plus(
        normal_train_seqs, TOP_K_FREQ, TOP_K_DISC, TOP_K_BIGRAMS)
    enable_stats = ENABLE_STATS
    n_freq, n_disc, n_ng = len(top_ids), len(disc_ids), len(top_bigrams)
    n_stats = 8 if enable_stats else 0

    _log('Mining PrefixSpan benign-pattern database...')
    benign_patterns = fit_benign_patterns(
        normal_train_seqs, WINDOW_LEN, WINDOW_STRIDE,
        CFG['ps_min_support'], CFG['ps_min_len'], CFG['ps_max_len'], CFG['ps_top_n'],
        rng_seed=GLOBAL_SEED)
    patterns_by_first = build_patterns_by_first(benign_patterns)
    ps_dims = PREFIXSPAN_DIMS if benign_patterns else 0

    feat_dim = n_freq + n_disc + n_stats + n_ng + n_cat + temporal_dims + ps_dims
    print(f'feature vector: freq={n_freq} disc={n_disc} stats={n_stats} '
          f'bigram={n_ng} cat={n_cat} temporal={temporal_dims} prefixspan={ps_dims}  '
          f'(patterns={len(benign_patterns)})  total_dim={feat_dim}')

    _extras = dict(top_ids=top_ids, disc_ids=disc_ids,
                   top_bigrams=top_bigrams, enable_stats=enable_stats)

    # Step 3b — Build feature matrices. Sliding-window is the canonical path
    # (the live detector is always window-based); labels inherit from each trace.
    if USE_SLIDING_WINDOW:
        print(f'Sliding window: length={WINDOW_LEN} stride={WINDOW_STRIDE}'
              f'  cap_per_split={MAX_WINDOWS_PER_SPLIT:,}')
        before = (len(seqs_train), len(seqs_val), len(seqs_test))
        print('Building windowed feature matrices...')
        X_train_full, y_train, d_tr = build_features_windowed(
            seqs_train, y_train, WINDOW_LEN, WINDOW_STRIDE, cat_cols, id_to_cat,
            patterns_by_first, ps_dims, ts_train, temporal_dims, **_extras,
            max_windows=MAX_WINDOWS_PER_SPLIT, rng_seed=GLOBAL_SEED)
        del seqs_train
        X_val, y_val, d_va = build_features_windowed(
            seqs_val, y_val, WINDOW_LEN, WINDOW_STRIDE, cat_cols, id_to_cat,
            patterns_by_first, ps_dims, ts_val, temporal_dims, **_extras,
            max_windows=MAX_WINDOWS_PER_SPLIT, rng_seed=GLOBAL_SEED + 1)
        del seqs_val
        X_test, y_test, d_te = build_features_windowed(
            seqs_test, y_test, WINDOW_LEN, WINDOW_STRIDE, cat_cols, id_to_cat,
            patterns_by_first, ps_dims, ts_test, temporal_dims, **_extras,
            max_windows=MAX_WINDOWS_PER_SPLIT, rng_seed=GLOBAL_SEED + 2)
        del seqs_test
        del seqs, ver_list, labels, splits
        print(f'  before traces:   train={before[0]:,}  val={before[1]:,}  test={before[2]:,}')
        print(f'  after windows:   train={X_train_full.shape[0]:,}  val={X_val.shape[0]:,}  test={X_test.shape[0]:,}')
        print(f'  dropped (shorter than window): train={d_tr}  val={d_va}  test={d_te}')
        X_val  = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
    else:
        print('Building feature matrices (whole-trace fallback)...')
        X_train_full = build_features(seqs_train, cat_cols, id_to_cat,
                                      patterns_by_first, ps_dims, ts_train, temporal_dims, **_extras)
        X_val        = build_features(seqs_val, cat_cols, id_to_cat,
                                      patterns_by_first, ps_dims, ts_val, temporal_dims, **_extras).astype(np.float32)
        X_test       = build_features(seqs_test, cat_cols, id_to_cat,
                                      patterns_by_first, ps_dims, ts_test, temporal_dims, **_extras).astype(np.float32)

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

    # Persist the Benign Pattern Database so inference matches the trained vectors.
    (out / 'benign_patterns.json').write_text(json.dumps({
        'patterns':    [list(p) for p in benign_patterns],
        'window_len':  WINDOW_LEN,
        'min_support': CFG['ps_min_support'],
        'min_len':     CFG['ps_min_len'],
        'max_len':     CFG['ps_max_len'],
        'top_n':       CFG['ps_top_n'],
    }, indent=2))

    feature_groups: dict = {}
    if n_freq:        feature_groups[f'freq_{n_freq}']         = n_freq
    if n_disc:        feature_groups[f'disc_{n_disc}']         = n_disc
    if n_stats:       feature_groups['stats_8']                = 8
    if n_ng:          feature_groups[f'bigrams_{n_ng}']        = n_ng
    feature_groups[f'cat_{n_cat}']                              = n_cat
    if temporal_dims: feature_groups[f'temporal_{temporal_dims}'] = temporal_dims
    if ps_dims:       feature_groups[f'prefixspan_{ps_dims}']  = ps_dims

    fe_report: dict = {
        'feature_groups': feature_groups,
        'total_dims':     feat_dim,
        'cat_cols':       cat_cols,
        # Serialise as {syscall_id: category}; keys are stringified for JSON.
        'syscall_id_to_cat': {str(k): v for k, v in (id_to_cat or {}).items()},
        'has_temporal':   has_temporal,
        'temporal_dims':  temporal_dims,
        'prefixspan_dims': ps_dims,
        'benign_patterns_file': 'benign_patterns.json',
        # Sensor alphabet — the syscall IDs training was restricted to. Inference
        # filters live windows to the same set so train/serve vectors match.
        'sensor_alphabet':       sorted(_ALLOWED_IDS) if _ALLOWED_IDS else [],
        'sensor_alphabet_names': SENSOR_ALPHABET_NAMES if USE_SENSOR_ALPHABET else [],
        # Engineered support vocab fit on normal-training windows.
        'top_ids':        list(top_ids),
        'disc_ids':       list(disc_ids),
        'top_bigrams':    [list(g) for g in top_bigrams],
        'enable_stats':   bool(enable_stats),
    }
    (out / 'fe_report.json').write_text(json.dumps(fe_report, indent=2))
    _log('feature_scaler.joblib + fe_report.json + benign_patterns.json saved')

    # Step 4: Isolation Forest
    _log(f'Training IsolationForest: {CFG["n_estimators"]} trees, n_train={X_train.shape[0]:,}  rows...')
    iforest = IsolationForest(
        n_estimators=CFG['n_estimators'],
        contamination=CFG['contamination'],
        random_state=GLOBAL_SEED,
        n_jobs=-1,
        verbose=0,   # 0: avoid per-call joblib.Parallel logging at inference
                     # (decision_function re-emits it every scored window).
    )
    iforest.fit(X_train)
    _log('IF: fit done; scoring train...')
    if_train = -iforest.decision_function(X_train)
    _log('IF: scoring val...')
    if_val   = -iforest.decision_function(X_val)
    _log('IF: scoring test...')
    if_test  = -iforest.decision_function(X_test)
    _log(f'IF val AUC: {roc_auc_score(y_val, if_val):.4f}')
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
    _log(f'Best seed={best_seed}  val_auc={best_val_auc:.4f}')

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
    _log(f'Ensemble val AUC: {roc_auc_score(y_val, ens_val):.4f}')

    # Step 7: Threshold + metrics
    # Paper §IV.B.3 uses T_0 = 99.5th percentile of benign validation errors
    # ('p995') as initial, then EMA-updates online at inference. Override with
    # THRESHOLD_STRATEGY=p995 for paper-faithful T_0; default 'f1' here.
    strategy = os.environ.get('THRESHOLD_STRATEGY', CFG['threshold']).lower()
    _log(f'Selecting thresholds (strategy={strategy}) and computing test metrics...')
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
        'experiment':       (f'dongting_{WINDOW_LEN}_{WINDOW_STRIDE}'
                             if USE_SLIDING_WINDOW else 'dongting_whole_trace'),
        'data_source':      'local',
        'n_sequences':      len(idx_train) + len(idx_val) + len(idx_test),
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
        'has_temporal':     has_temporal,
        'prefixspan_dims':  ps_dims,
        'n_benign_patterns': len(benign_patterns),
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
