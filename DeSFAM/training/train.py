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


# ── Step 2: Load sequences from MongoDB ───────────────────────────────────

def load_sequences(name_to_id: dict[str, int]):
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
    db = client[MONGO_DB]

    print('  Indexing normal sequences...')
    normal_idx: dict[str, str] = {}
    for doc in db.kernel_syscall_normal_strace.find(
            {}, {'kns_normal_file_name': 1, 'kns_normal_mlseq_list': 1}):
        normal_idx[doc['kns_normal_file_name']] = doc['kns_normal_mlseq_list']

    print('  Indexing attack sequences...')
    attack_idx: dict[str, str] = {}
    for doc in db.kernel_syscallhook_bugpoc_trace_sum.find(
            {}, {'kshs_poclog_name': 1, 'kshs_bugpoc_syscall_list': 1}):
        attack_idx[doc['kshs_poclog_name']] = doc['kshs_bugpoc_syscall_list']

    print('  Reading baseline index...')
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


def build_features(seqs, ver_list, top_ids, disc_ids, top_ngrams, ver_cols, ngram_n=2) -> np.ndarray:
    n_freq  = len(top_ids)
    n_disc  = len(disc_ids)
    n_ng    = len(top_ngrams)
    n_ver   = len(ver_cols)
    n_feat  = n_freq + n_disc + 8 + n_ng + n_ver
    top_idx  = {sc: i for i, sc in enumerate(top_ids)}
    disc_idx = {sc: i for i, sc in enumerate(disc_ids)}
    ng_idx   = {g: i  for i, g  in enumerate(top_ngrams)}
    ver_idx  = {v: i  for i, v  in enumerate(ver_cols)}
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
    if strategy == 'p99':
        return float(np.percentile(scores[y_val == 0], 99))
    fpr_a, tpr_a, thresholds = roc_curve(y_val, scores)
    if strategy == 'fpr5':
        valid = np.where(fpr_a <= 0.05)[0]
        idx = valid[np.argmax(tpr_a[valid])] if len(valid) else 0
        return float(thresholds[idx])
    best_f1, best_t = -1.0, 0.5
    for t in thresholds:
        f = f1_score(y_val, (scores >= t).astype(int), zero_division=0)
        if f > best_f1: best_f1, best_t = f, float(t)
    return best_t


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

    # Step 3a
    normal_train_seqs = [seqs_train[i] for i in range(len(seqs_train)) if y_train[i] == 0]
    normal_train_ver  = [ver_train[i]  for i in range(len(seqs_train)) if y_train[i] == 0]
    top_ids, disc_ids, top_ngrams, ver_cols = fit_vocab(
        normal_train_seqs, normal_train_ver,
        CFG['top_k_freq'], CFG['top_k_disc'], CFG['top_k_ngrams'], CFG['ngram_n'],
    )
    n_ng     = len(top_ngrams)
    n_ver    = len(ver_cols)
    feat_dim = CFG['top_k_freq'] + CFG['top_k_disc'] + 8 + n_ng + n_ver
    print(f'freq={len(top_ids)}  disc={len(disc_ids)}  bigrams={n_ng}  ver={n_ver}  total_dim={feat_dim}')

    # Step 3b
    print('Building feature matrices...')
    X_train_full = build_features(seqs_train, ver_train, top_ids, disc_ids, top_ngrams, ver_cols, CFG['ngram_n'])
    X_val        = build_features(seqs_val,   ver_val,   top_ids, disc_ids, top_ngrams, ver_cols, CFG['ngram_n']).astype(np.float32)
    X_test       = build_features(seqs_test,  ver_test,  top_ids, disc_ids, top_ngrams, ver_cols, CFG['ngram_n']).astype(np.float32)

    # Step 3c
    X_train_norm = X_train_full[y_train == 0].astype(np.float32)
    feature_scaler = RobustScaler(quantile_range=(1.0, 99.0))
    feature_scaler.fit(X_train_norm)
    X_train = feature_scaler.transform(X_train_norm)
    X_val   = feature_scaler.transform(X_val).astype(np.float32)
    X_test  = feature_scaler.transform(X_test).astype(np.float32)

    joblib.dump(feature_scaler, out / 'feature_scaler.joblib')
    input_dim = feat_dim
    fe_report = {
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
    (out / 'fe_report.json').write_text(json.dumps(fe_report, indent=2))
    print('feature_scaler.joblib + fe_report.json saved')

    # Step 4: Isolation Forest
    print(f'Training IsolationForest: {CFG["n_estimators"]} trees ...')
    iforest = IsolationForest(
        n_estimators=CFG['n_estimators'],
        contamination=CFG['contamination'],
        random_state=GLOBAL_SEED,
        n_jobs=-1,
    )
    iforest.fit(X_train)
    if_train = -iforest.decision_function(X_train)
    if_val   = -iforest.decision_function(X_val)
    if_test  = -iforest.decision_function(X_test)
    print(f'IF val AUC: {roc_auc_score(y_val, if_val):.4f}  (paper: 0.646)')
    joblib.dump(iforest, out / 'iforest.joblib')

    # Step 5: VAE
    print(f'Training VAE with seeds {CFG["seeds"]} ...')
    seed_reports = []
    best_val_auc, best_enc_w, best_dec_w = -1.0, None, None
    best_seed = CFG['seeds'][0]

    for seed in CFG['seeds']:
        set_seeds(seed)
        encoder, decoder, vae = build_vae(
            input_dim, CFG['latent_dim'], CFG['hidden_dim'], CFG['dropout'], CFG['l2'])
        vae.compile(optimizer=keras.optimizers.Adam(CFG['lr']), loss='mse')
        hist = vae.fit(
            X_train, X_train,
            epochs=CFG['epochs'], batch_size=CFG['batch_size'],
            validation_split=0.05,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0,
        )
        val_auc    = float(roc_auc_score(y_val, recon_error(X_val, encoder, decoder)))
        best_loss  = float(min(hist.history['val_loss']))
        epochs_run = len(hist.history['loss'])
        print(f'  seed={seed}  val_auc={val_auc:.4f}  best_val_loss={best_loss:.4f}  epochs={epochs_run}')
        seed_reports.append({'seed': seed, 'val_auc': val_auc,
                             'epochs': epochs_run, 'best_val_loss': best_loss})
        if val_auc > best_val_auc:
            best_val_auc = val_auc; best_seed = seed
            best_enc_w = encoder.get_weights(); best_dec_w = decoder.get_weights()

    set_seeds(best_seed)
    encoder, decoder, _ = build_vae(
        input_dim, CFG['latent_dim'], CFG['hidden_dim'], CFG['dropout'], CFG['l2'])
    encoder.set_weights(best_enc_w); decoder.set_weights(best_dec_w)
    print(f'Best seed={best_seed}  val_auc={best_val_auc:.4f}  (paper: 0.747)')

    vae_train = recon_error(X_train, encoder, decoder)
    vae_val   = recon_error(X_val,   encoder, decoder)
    vae_test  = recon_error(X_test,  encoder, decoder)

    # Step 6: Ensemble
    ensemble = EnsembleScorer(alpha=CFG['alpha'])
    ensemble.fit(vae_train, if_train)
    ens_val  = ensemble.score(vae_val,  if_val)
    ens_test = ensemble.score(vae_test, if_test)
    print(f'Ensemble val AUC: {roc_auc_score(y_val, ens_val):.4f}  (paper: 0.656)')

    # Step 7: Threshold + metrics
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
        'experiment':       'dongting_build_and_train',
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
