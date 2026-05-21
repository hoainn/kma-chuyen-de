"""
Retrain the VAE on container-derived 174-dim windows.

Identical architecture + RobustScaler quantile-band normalisation as
`train/train_fe.py` (so the live detector loads v2 weights via the same
code path). Only the input data differs.

Output: outputs/vae_encoder_v2_container.weights.h5,
        outputs/vae_decoder_v2_container.weights.h5,
        outputs/model_params_v2_container.json,
        outputs/train_vae_container_report.json
"""
import json
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'outputs')


def load_data():
    X_tr = np.load(os.path.join(OUT, 'X_container_train_scaled.npy')).astype(np.float32)
    X_va = np.load(os.path.join(OUT, 'X_container_val_scaled.npy')).astype(np.float32)
    X_te = np.load(os.path.join(OUT, 'X_container_test_scaled.npy')).astype(np.float32)
    y_tr = np.load(os.path.join(OUT, 'y_container_train.npy'))
    y_va = np.load(os.path.join(OUT, 'y_container_val.npy'))
    y_te = np.load(os.path.join(OUT, 'y_container_test.npy'))
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def build_encoder(input_dim, latent_dim):
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='selu')(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='selu')(x)
    x = layers.Dropout(0.2)(x)
    mu = layers.Dense(latent_dim, name='mu')(x)
    lv = layers.Dense(latent_dim, name='lv')(x)
    return keras.Model(inp, [mu, lv], name='encoder')


def build_decoder(latent_dim, output_dim):
    inp = keras.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation='selu')(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='selu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(output_dim)(x)
    return keras.Model(inp, out, name='decoder')


def reparam(mu, lv):
    return mu + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(mu))


def vae_step(enc, dec, opt, xb, training):
    with tf.GradientTape() as tape:
        mu, lv = enc(xb, training=training)
        z = reparam(mu, lv)
        xh = dec(z, training=training)
        recon = tf.reduce_mean(tf.reduce_sum(tf.square(xb - xh), axis=1))
        kl = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + lv - tf.square(mu) - tf.exp(lv), axis=1))
        loss = recon + kl
    if training:
        grads = tape.gradient(loss, enc.trainable_variables + dec.trainable_variables)
        opt.apply_gradients(zip(grads, enc.trainable_variables + dec.trainable_variables))
    return loss


def recon_error(enc, dec, x, n=5):
    errs = []
    for _ in range(n):
        mu, lv = enc(x, training=False)
        z = reparam(mu, lv)
        xh = dec(z, training=False)
        errs.append(tf.reduce_sum(tf.square(x - xh), axis=1).numpy())
    return np.mean(errs, axis=0)


class RobustScaler:
    def __init__(self, ref):
        self.lo = float(np.percentile(ref, 1))
        self.hi = float(np.percentile(ref, 99))

    def __call__(self, x):
        return np.clip((x - self.lo) / (self.hi - self.lo + 1e-9), 0.0, 1.0)


def main():
    LATENT, EPOCHS, BATCH, PATIENCE = 24, 50, 256, 8
    N_SEEDS = 3
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_data()
    X_tr_normal = X_tr[y_tr == 0]
    print(f'Train: {X_tr.shape} attack_rate={y_tr.mean():.3f}')
    print(f'  Normal-only training subset: {X_tr_normal.shape}')
    INPUT_DIM = X_tr.shape[1]

    best_val_auc = -1
    best_enc = best_dec = None
    seed_results = []

    from sklearn.metrics import roc_auc_score

    for seed in range(N_SEEDS):
        tf.random.set_seed(seed); np.random.seed(seed)
        enc = build_encoder(INPUT_DIM, LATENT)
        dec = build_decoder(LATENT, INPUT_DIM)
        opt = keras.optimizers.Adam(1e-3)
        n_vv = max(64, int(len(X_tr_normal) * 0.1))
        Xvv = tf.constant(X_tr_normal[:n_vv])
        Xtr = tf.constant(X_tr_normal[n_vv:])
        best_val_loss = np.inf
        best_ew = best_dw = None
        patience_cnt = 0

        t0 = time.time()
        for epoch in range(EPOCHS):
            idx = np.random.permutation(len(Xtr))
            Xs = tf.gather(Xtr, idx)
            for i in range(0, len(Xs), BATCH):
                vae_step(enc, dec, opt, Xs[i:i + BATCH], training=True)
            vl = float(vae_step(enc, dec, opt, Xvv, training=False))
            if vl < best_val_loss:
                best_val_loss = vl
                best_ew = enc.get_weights()
                best_dw = dec.get_weights()
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    break

        enc.set_weights(best_ew); dec.set_weights(best_dw)
        scores_v = recon_error(enc, dec, X_va)
        val_auc = roc_auc_score(y_va, scores_v) if y_va.sum() > 0 else 0.5
        elapsed = time.time() - t0
        print(f'  seed {seed}: {epoch+1} epochs, {elapsed:.1f}s val_AUC={val_auc:.4f}')
        seed_results.append({'seed': seed, 'val_auc': float(val_auc),
                             'best_val_loss': best_val_loss, 'epochs': epoch + 1})
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_enc, best_dec = enc, dec

    print(f'\nBest seed val_AUC={best_val_auc:.4f}')

    # RobustScaler fitted on TRAINING scores of best model
    scores_train = recon_error(best_enc, best_dec, X_tr)
    norm = RobustScaler(scores_train)

    scores_va = recon_error(best_enc, best_dec, X_va)
    scores_te = recon_error(best_enc, best_dec, X_te)

    # Pick threshold from val ROC (Youden-J)
    from sklearn.metrics import roc_curve, classification_report, confusion_matrix
    fpr, tpr, th = roc_curve(y_va, scores_va)
    threshold = float(th[np.argmax(tpr - fpr)])

    preds_te = (scores_te >= threshold).astype(int)
    rep = classification_report(y_te, preds_te, target_names=['Normal', 'Attack'],
                                output_dict=True)
    cm = confusion_matrix(y_te, preds_te).tolist()
    test_auc = roc_auc_score(y_te, scores_te) if y_te.sum() > 0 else float('nan')
    print(f'\nTest AUC={test_auc:.4f}  F1={rep["Attack"]["f1-score"]:.4f}  '
          f'P={rep["Attack"]["precision"]:.4f}  R={rep["Attack"]["recall"]:.4f}')

    # Save artifacts
    best_enc.save_weights(os.path.join(OUT, 'vae_encoder_v2_container.weights.h5'))
    best_dec.save_weights(os.path.join(OUT, 'vae_decoder_v2_container.weights.h5'))
    with open(os.path.join(OUT, 'model_params_v2_container.json'), 'w') as f:
        json.dump({
            'variant': 'vae_container',
            'latent_dim': LATENT,
            'input_dim': INPUT_DIM,
            'vae_lo': norm.lo,
            'vae_hi': norm.hi,
            'vae_threshold': threshold,
            'best_val_auc': float(best_val_auc),
        }, f, indent=2)
    with open(os.path.join(OUT, 'train_vae_container_report.json'), 'w') as f:
        json.dump({
            'seeds': seed_results,
            'test': {'auc': float(test_auc),
                     'f1': rep['Attack']['f1-score'],
                     'precision': rep['Attack']['precision'],
                     'recall': rep['Attack']['recall'],
                     'cm': cm,
                     'threshold': threshold},
        }, f, indent=2)
    print('\nSaved vae_encoder_v2_container.weights.h5 + model_params_v2_container.json')


if __name__ == '__main__':
    main()
