"""
Fusion-weight α sensitivity for the paper-only run — reproduces the paper's
Sensitivity-Analysis figure on OUR data using the saved artifacts.

ens(α) = α · RobustScale(vae_recon) + (1-α) · RobustScale(if_score)   [paper Eq. 2]

Loads the trained IF + VAE + ensemble scalers from /model, recomputes validation
scores, sweeps α and reports AUC / AP / F1. Run in the training container:
    docker compose -f docker-compose.pipeline.yml run --rm train python sweep_alpha.py
"""
import json
import os

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import train  # reuse the exact feature pipeline

MODEL = os.environ.get('OUTPUT_DIR', '/model')
# /model is reserved for trained-model artifacts only — write the sweep figure
# elsewhere (defaults to the training dir / cwd, which is /workspace in the container).
SWEEP_OUT = os.environ.get('SWEEP_OUT_DIR', os.getcwd())
VAL_CAP = int(os.environ.get('SWEEP_VAL_CAP', '300000'))
ALPHAS = [0.0, 0.25, 0.5, 0.7, 0.85, 1.0]

def main():
    fe = json.load(open(os.path.join(MODEL, 'fe_report.json')))
    cat_cols     = fe['cat_cols']
    ps_dims      = int(fe.get('prefixspan_dims', 0))
    input_dim    = int(fe['total_dims'])
    top_ids      = [int(x) for x in fe.get('top_ids', [])]
    disc_ids     = [int(x) for x in fe.get('disc_ids', [])]
    top_bigrams  = [tuple(g) for g in fe.get('top_bigrams', [])]
    enable_stats = bool(fe.get('enable_stats', False))
    mode         = fe.get('feature_mode', 'paper_only')

    name_to_id = train.load_syscall_table(train.SYSCALL_TBL)
    id_to_cat  = train.build_id_to_category(name_to_id)
    bp = json.load(open(os.path.join(MODEL, 'benign_patterns.json')))
    pbf = train.build_patterns_by_first([tuple(p) for p in bp['patterns']])

    seqs, labels, splits, ver, ts = train.load_sequences(name_to_id)
    idx = [i for i, s in enumerate(splits) if s == 'val']
    sva = [seqs[i] for i in idx]
    yv0 = np.array([labels[i] for i in idx], dtype=np.int32)
    print(f'mode={mode}  val traces={len(sva):,}', flush=True)

    X, y, _ = train.build_features_windowed(
        sva, yv0, train.WINDOW_LEN, train.WINDOW_STRIDE, cat_cols, id_to_cat,
        pbf, ps_dims, None, 0,
        top_ids=top_ids, disc_ids=disc_ids, top_bigrams=top_bigrams,
        enable_stats=enable_stats,
        max_windows=VAL_CAP, rng_seed=123)
    print(f'val windows={X.shape[0]:,}  dim={X.shape[1]}', flush=True)

    scaler = joblib.load(os.path.join(MODEL, 'feature_scaler.joblib'))
    X = scaler.transform(X).astype(np.float32)

    iforest = joblib.load(os.path.join(MODEL, 'iforest.joblib'))
    if_val = -iforest.decision_function(X)

    enc, dec, _ = train.build_vae(input_dim, train.CFG['latent_dim'],
                                  train.CFG['hidden_dim'], train.CFG['dropout'], train.CFG['l2'])
    import tensorflow as tf
    enc(tf.zeros((1, input_dim))); dec(tf.zeros((1, train.CFG['latent_dim'])))
    enc.load_weights(os.path.join(MODEL, 'vae_encoder.weights.h5'))
    dec.load_weights(os.path.join(MODEL, 'vae_decoder.weights.h5'))
    vae_val = train.recon_error(X, enc, dec)

    vsc = joblib.load(os.path.join(MODEL, 'ensemble_vae_scaler.joblib'))
    isc = joblib.load(os.path.join(MODEL, 'ensemble_if_scaler.joblib'))
    v = vsc.transform(vae_val.reshape(-1, 1)).ravel()
    f = isc.transform(if_val.reshape(-1, 1)).ravel()

    print(f'\n{"alpha":>6} {"AUC":>7} {"AP":>7} {"F1*":>7}   (alpha = VAE weight)')
    print('-' * 40)
    rows = []
    for a in ALPHAS:
        s = a * v + (1 - a) * f
        auc = roc_auc_score(y, s)
        ap = average_precision_score(y, s)
        thr = train.select_threshold(y, s, 'f1')
        f1 = f1_score(y, (s >= thr).astype(int), zero_division=0)
        rows.append((a, auc, ap, f1))
        tag = '  <- paper peak' if a == 0.7 else ('  <- IF only' if a == 0 else ('  <- VAE only' if a == 1 else ''))
        print(f'{a:>6.2f} {auc:>7.4f} {ap:>7.4f} {f1:>7.4f}{tag}', flush=True)

    best = max(rows, key=lambda r: r[1])
    print(f'\nOur best AUC at alpha={best[0]:.2f} (AUC={best[1]:.4f}). Paper peak: alpha=0.70.')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        a_ = [r[0] for r in rows]
        plt.figure(figsize=(7.5, 4.2))
        plt.plot(a_, [r[3] for r in rows], 'o-', color='#E8B23A', label='F1-Score')
        plt.plot(a_, [r[1] for r in rows], 'o-', color='#D2691E', label='AUC')
        plt.axvline(0.7, ls='--', c='grey', lw=1, alpha=0.6)
        plt.title(f'Fusion Weight α Sensitivity — {mode} run (DongTing)')
        plt.xlabel('Fusion Weight α (VAE contribution)'); plt.ylabel('Score')
        plt.legend(); plt.grid(ls=':', alpha=0.5); plt.tight_layout()
        out = os.path.join(SWEEP_OUT, f'alpha_sweep_{mode}.png')
        plt.savefig(out, dpi=130)
        print(f'figure saved: {out}')
    except Exception as e:
        print(f'(plot skipped: {e})')


if __name__ == '__main__':
    main()
