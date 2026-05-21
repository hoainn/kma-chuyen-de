"""
Build 174-dim feature vectors from `collect/recordings/*.npy` syscall
recordings by sliding a 200-syscall window over each recording (the
same window size the live detector uses).

Reuses the FE config (top_ids / disc_ids / top_bigrams / ver_cols) from
`outputs/fe_report.json` and the StandardScaler from
`outputs/scaler_fe.pkl` so that v1 and v2 models share the same feature
space.

Outputs:
  outputs/X_container_train_scaled.npy
  outputs/X_container_val_scaled.npy
  outputs/X_container_test_scaled.npy
  y_container_{train,val,test}.npy
  outputs/container_features_meta.json

Usage:
  python train_v2/build_features.py --window 200 --step 50
"""
import argparse
import collections
import glob
import json
import os

import joblib
import numpy as np
from scipy.stats import entropy as scipy_entropy


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'outputs')
RECORDINGS = os.path.join(ROOT, 'collect', 'recordings')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--window', type=int, default=200)
    p.add_argument('--step', type=int, default=50)
    p.add_argument('--kernel-ver', default='6.12')
    p.add_argument('--max-windows-per-pod', type=int, default=2000,
                   help='Cap to keep balanced contribution per pod')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_fe_config():
    with open(os.path.join(OUT, 'fe_report.json')) as f:
        cfg = json.load(f)
    return {
        'TOP_IDS': list(cfg['top_ids']),
        'DISC_IDS': list(cfg['disc_ids']),
        'TOP_BIGRAMS': [tuple(b) for b in cfg['top_bigrams']],
        'VER_COLS': cfg['ver_cols'],
    }


def build_feature(seq, fe, kernel_ver):
    """Mirror the live detector's build_features() — 60+40+8+40+26 = 174 dims."""
    arr = np.asarray(seq, dtype=np.int32)
    total = len(seq)
    if total == 0:
        return np.zeros(174, dtype=np.float32)

    counts = collections.Counter(seq)
    freq = np.array([counts.get(s, 0) / (total + 1e-9) for s in fe['TOP_IDS']],
                    dtype=np.float32)

    s_set = set(seq.tolist() if isinstance(seq, np.ndarray) else seq)
    disc = np.array([1.0 if d in s_set else 0.0 for d in fe['DISC_IDS']],
                    dtype=np.float32)

    unique = len(s_set)
    freq_pr = np.bincount(arr[arr >= 0], minlength=548).astype(np.float32)
    freq_pr /= (freq_pr.sum() + 1e-9)
    ent = float(scipy_entropy(freq_pr + 1e-9))
    stats = np.array([
        np.log1p(total),
        np.log1p(unique),
        unique / (total + 1e-9),
        ent,
        np.log1p(total),
        np.log1p(total * 20),
        float(np.mean(arr)),
        float(np.std(arr)),
    ], dtype=np.float32)

    if total >= 2:
        bg_cnt = collections.Counter(zip(seq[:-1].tolist(), seq[1:].tolist())
                                     if isinstance(seq, np.ndarray)
                                     else zip(seq[:-1], seq[1:]))
        bg_tot = sum(bg_cnt.values()) + 1e-9
        bigrams = np.array([bg_cnt.get(bg, 0) / bg_tot for bg in fe['TOP_BIGRAMS']],
                           dtype=np.float32)
    else:
        bigrams = np.zeros(len(fe['TOP_BIGRAMS']), dtype=np.float32)

    ver_short = '.'.join(kernel_ver.split('.')[:2])
    ver_vec = np.array([1.0 if v == ver_short else 0.0 for v in fe['VER_COLS']],
                       dtype=np.float32)

    return np.concatenate([freq, disc, stats, bigrams, ver_vec])


def sliding_windows(seq, win, step):
    seq = np.asarray(seq, dtype=np.int32)
    if len(seq) < win:
        return []
    out = []
    for i in range(0, len(seq) - win + 1, step):
        out.append(seq[i:i + win])
    return out


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    fe = load_fe_config()

    rec_files = sorted(glob.glob(os.path.join(RECORDINGS, '*.npy')))
    if not rec_files:
        raise SystemExit(f'No recordings under {RECORDINGS}')

    X_rows = []
    y_rows = []
    meta = {'sources': [], 'per_pod_window_counts': {}}

    for npy in rec_files:
        json_side = npy.replace('.npy', '.json')
        if not os.path.exists(json_side):
            continue
        with open(json_side) as f:
            md = json.load(f)
        label = md.get('label', 'unknown')
        if label not in ('benign', 'attack'):
            print(f'  skip {os.path.basename(npy)} (label={label})')
            continue

        seq = np.load(npy)
        wins = sliding_windows(seq, args.window, args.step)
        if not wins:
            print(f'  skip {os.path.basename(npy)} (only {len(seq)} syscalls)')
            continue

        if len(wins) > args.max_windows_per_pod:
            idx = rng.choice(len(wins), args.max_windows_per_pod, replace=False)
            wins = [wins[i] for i in sorted(idx)]

        y_val = 1 if label == 'attack' else 0
        for w in wins:
            X_rows.append(build_feature(w, fe, args.kernel_ver))
            y_rows.append(y_val)

        meta['sources'].append({
            'file': os.path.basename(npy),
            'pod': md.get('pod_key'),
            'label': label,
            'total_syscalls': int(seq.size),
            'windows': len(wins),
        })
        meta['per_pod_window_counts'][md.get('pod_key', 'unknown')] = len(wins)
        print(f'  ok {os.path.basename(npy):60s} {len(wins):5d} windows ({label})')

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int8)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f'\nTotal windows: {len(X)}  attack rate: {y.mean():.3f}')

    # Apply v1 scaler (no refit — keep feature space identical to v1).
    scaler = joblib.load(os.path.join(OUT, 'scaler_fe.pkl'))
    X_scaled = scaler.transform(X).astype(np.float32)

    # Stratified 70/15/15 split
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    def split(arr, ratios):
        a, b = int(len(arr) * ratios[0]), int(len(arr) * (ratios[0] + ratios[1]))
        return arr[:a], arr[a:b], arr[b:]
    pos_tr, pos_va, pos_te = split(pos, [0.7, 0.15])
    neg_tr, neg_va, neg_te = split(neg, [0.7, 0.15])
    tr = np.concatenate([pos_tr, neg_tr])
    va = np.concatenate([pos_va, neg_va])
    te = np.concatenate([pos_te, neg_te])
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)

    for name, ix in [('train', tr), ('val', va), ('test', te)]:
        np.save(os.path.join(OUT, f'X_container_{name}_scaled.npy'), X_scaled[ix])
        np.save(os.path.join(OUT, f'y_container_{name}.npy'), y[ix])
        print(f'  {name}: {len(ix)} samples, attack_rate={y[ix].mean():.3f}')

    meta['split_counts'] = {'train': int(len(tr)), 'val': int(len(va)),
                            'test': int(len(te))}
    meta['feature_dim'] = int(X.shape[1])
    meta['window'] = args.window
    meta['step'] = args.step
    with open(os.path.join(OUT, 'container_features_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\nWrote outputs/X_container_*_scaled.npy + y_container_*.npy')


if __name__ == '__main__':
    main()
