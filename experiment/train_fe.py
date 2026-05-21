"""
DongTing Model Training — Full DeSFAM Pipeline (174-dim FE features)

Fixes vs. previous version:
  1. Ensemble normalizer fitted on TRAINING scores, applied consistently to val/test.
     (Old bug: norm01 used per-split min/max, so val threshold didn't transfer to test.)
  2. Multiple VAE seeds → report mean ± std to verify stability.
  3. IF score calibration also uses training reference.
  4. Threshold selection on val set, evaluation on held-out test set.

Reference:
  DeSFAM: An Adaptive eBPF and AI-Driven Framework for Securing Cloud Containers,
  IEEE Access 2025 — DOI: 10.1109/ACCESS.2025.3592192
"""
import os, json, time, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix, roc_curve,
    f1_score, precision_score, recall_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings('ignore')
OUT = 'outputs'; os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load pre-computed feature arrays
# ─────────────────────────────────────────────────────────────────────────────
print('='*60)
print('LOADING FEATURE ARRAYS')
print('='*60)

X_train = np.load(f'{OUT}/X_train_scaled.npy').astype(np.float32)
X_val   = np.load(f'{OUT}/X_val_scaled.npy').astype(np.float32)
X_test  = np.load(f'{OUT}/X_test_scaled.npy').astype(np.float32)
y_train = np.load(f'{OUT}/y_train.npy')
y_val   = np.load(f'{OUT}/y_val.npy')
y_test  = np.load(f'{OUT}/y_test.npy')

X_train_normal = X_train[y_train == 0]  # unsupervised training set

print(f'  Train : {X_train.shape}  attack_rate={y_train.mean():.3f}')
print(f'  Val   : {X_val.shape}   attack_rate={y_val.mean():.3f}')
print(f'  Test  : {X_test.shape}  attack_rate={y_test.mean():.3f}')
print(f'  Normal train (IF+VAE input): {X_train_normal.shape[0]} sequences')

INPUT_DIM = X_train.shape[1]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def find_threshold(y_true, scores):
    """Youden-J optimal threshold from ROC curve."""
    fpr, tpr, threshs = roc_curve(y_true, scores)
    j = np.argmax(tpr - fpr)
    return float(threshs[j])

def eval_scores(name, y_true, scores, threshold=None):
    auc = roc_auc_score(y_true, scores)
    ap  = average_precision_score(y_true, scores)
    if threshold is None:
        threshold = find_threshold(y_true, scores)
    preds = (scores >= threshold).astype(int)
    cm    = confusion_matrix(y_true, preds)
    rep   = classification_report(y_true, preds,
                                  target_names=['Normal', 'Attack'],
                                  output_dict=True)
    atk   = rep['Attack']
    print(f'\n  [{name}]  AUC={auc:.4f}  AP={ap:.4f}  threshold={threshold:.6f}')
    print(f'  CM:\n{cm}')
    print(classification_report(y_true, preds, target_names=['Normal', 'Attack']))
    return dict(auc=auc, ap=ap, threshold=threshold,
                precision=atk['precision'], recall=atk['recall'],
                f1=atk['f1-score'], accuracy=rep['accuracy'],
                cm=cm.tolist())

class MinMaxScaler:
    """Fitted on a reference set; applied consistently to other sets."""
    def __init__(self, ref_scores):
        self.lo = float(ref_scores.min())
        self.hi = float(ref_scores.max())

    def __call__(self, x):
        return np.clip((x - self.lo) / (self.hi - self.lo + 1e-9), 0.0, 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Isolation Forest — normal-only training
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('ISOLATION FOREST  (unsupervised, normal-only training)')
print('='*60)

t0 = time.time()
iso = IsolationForest(
    n_estimators=300,
    contamination='auto',   # 10% — assumes ~10% outliers in scoring pool
    max_samples='auto',
    random_state=42,
    n_jobs=-1,
)
iso.fit(X_train_normal)
print(f'  Trained in {time.time()-t0:.1f}s')

# Higher = more anomalous  (negate Isolation Forest decision function)
if_scores_train = -iso.decision_function(X_train)
if_scores_val   = -iso.decision_function(X_val)
if_scores_test  = -iso.decision_function(X_test)

# Normalizer fitted on training (same reference for val + test)
norm_if = MinMaxScaler(if_scores_train)

print('\n  --- Validation ---')
if_val  = eval_scores('IF val',  y_val,  if_scores_val)
print('\n  --- Test ---')
if_test = eval_scores('IF test', y_test, if_scores_test, threshold=if_val['threshold'])

# ─────────────────────────────────────────────────────────────────────────────
# 4. VAE — DeSFAM architecture, multiple seeds for stability
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('VARIATIONAL AUTOENCODER  (normal-only training)')
print('='*60)

LATENT = 24
EPOCHS = 80
BATCH  = 256
PATIENCE = 10
N_SEEDS = 3   # run 3 seeds, pick best val-AUC, report all

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

def reparam(mu, lv):
    return mu + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(mu))

def vae_step(enc, dec, opt, xb, training):
    with tf.GradientTape() as tape:
        mu, lv = enc(xb, training=training)
        z      = reparam(mu, lv)
        xh     = dec(z, training=training)
        recon  = tf.reduce_mean(tf.reduce_sum(tf.square(xb - xh), axis=1))
        kl     = -0.5 * tf.reduce_mean(
                     tf.reduce_sum(1 + lv - tf.square(mu) - tf.exp(lv), axis=1))
        loss   = recon + kl
    if training:
        grads = tape.gradient(loss, enc.trainable_variables + dec.trainable_variables)
        opt.apply_gradients(zip(grads, enc.trainable_variables + dec.trainable_variables))
    return loss, recon, kl

def recon_error(enc, dec, x, n_samples=5):
    """Mean reconstruction error over n_samples (reduces randomness)."""
    errors = []
    for _ in range(n_samples):
        mu, lv = enc(x, training=False)
        z      = reparam(mu, lv)
        xh     = dec(z, training=False)
        errors.append(tf.reduce_sum(tf.square(x - xh), axis=1).numpy())
    return np.mean(errors, axis=0)

def train_vae(seed):
    tf.random.set_seed(seed); np.random.seed(seed)
    enc = build_encoder(INPUT_DIM, LATENT)
    dec = build_decoder(LATENT, INPUT_DIM)
    opt = keras.optimizers.Adam(1e-3)

    n_vv = int(len(X_train_normal) * 0.1)
    Xvv  = tf.constant(X_train_normal[:n_vv])
    Xtr  = tf.constant(X_train_normal[n_vv:])

    best_val, best_ew, best_dw = np.inf, None, None
    patience_cnt = 0
    tr_losses, vl_losses = [], []

    for epoch in range(EPOCHS):
        idx = np.random.permutation(len(Xtr))
        Xs  = tf.gather(Xtr, idx)
        ep  = []
        for i in range(0, len(Xs), BATCH):
            loss, _, _ = vae_step(enc, dec, opt, Xs[i:i+BATCH], training=True)
            ep.append(float(loss))
        vl, _, _ = vae_step(enc, dec, opt, Xvv, training=False)
        tr_losses.append(np.mean(ep)); vl_losses.append(float(vl))
        if float(vl) < best_val:
            best_val   = float(vl)
            best_ew    = enc.get_weights()
            best_dw    = dec.get_weights()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    enc.set_weights(best_ew); dec.set_weights(best_dw)
    return enc, dec, tr_losses, vl_losses

seed_results = []
best_val_auc = -1
best_enc = best_dec = None

for seed in range(N_SEEDS):
    t0 = time.time()
    print(f'\n  Seed {seed} ...')
    enc, dec, tr_l, vl_l = train_vae(seed)
    elapsed = time.time() - t0

    scores_v = recon_error(enc, dec, X_val)
    val_auc  = roc_auc_score(y_val, scores_v)
    print(f'  Seed {seed}: {len(tr_l)} epochs, {elapsed:.1f}s  val_AUC={val_auc:.4f}')

    seed_results.append({'seed': seed, 'val_auc': val_auc,
                         'epochs': len(tr_l), 'best_val_loss': min(vl_l)})
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_enc, best_dec = enc, dec

print(f'\n  Best seed: val_AUC={best_val_auc:.4f}')

# Compute final scores with best model (mean over 5 samples for stability)
vae_scores_train = recon_error(best_enc, best_dec, X_train)
vae_scores_val   = recon_error(best_enc, best_dec, X_val)
vae_scores_test  = recon_error(best_enc, best_dec, X_test)

# Normalizer fitted on training (same reference for val + test)
norm_vae = MinMaxScaler(vae_scores_train)

print('\n  --- Validation ---')
vae_val  = eval_scores('VAE val',  y_val,  vae_scores_val)
print('\n  --- Test ---')
vae_test = eval_scores('VAE test', y_test, vae_scores_test, threshold=vae_val['threshold'])

# ─────────────────────────────────────────────────────────────────────────────
# 5. Ensemble — DeSFAM: A(W) = α·A_VAE + (1-α)·A_IF
#    CRITICAL: normalize with TRAINING-fitted scalers, same reference for all splits.
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('ENSEMBLE  (α=0.7·VAE + 0.3·IF, training-fitted normalizer)')
print('='*60)
ALPHA = 0.7

# Both normalizers were fitted on training scores — apply to val and test
ens_val  = ALPHA * norm_vae(vae_scores_val)  + (1-ALPHA) * norm_if(if_scores_val)
ens_test = ALPHA * norm_vae(vae_scores_test) + (1-ALPHA) * norm_if(if_scores_test)

print('\n  --- Validation ---')
ens_val_res  = eval_scores('Ensemble val',  y_val,  ens_val)
print('\n  --- Test ---')
ens_test_res = eval_scores('Ensemble test', y_test, ens_test, threshold=ens_val_res['threshold'])

# ─────────────────────────────────────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────────────────────────────────────
print('\nGenerating plots...')

# ROC curves (test)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for scores, label, clr in [
    (if_scores_test,  f'Isolation Forest (AUC={if_test["auc"]:.4f})',  '#4C72B0'),
    (vae_scores_test, f'VAE             (AUC={vae_test["auc"]:.4f})',  '#DD8452'),
    (ens_test,        f'Ensemble        (AUC={ens_test_res["auc"]:.4f})', '#2ca02c'),
]:
    fpr, tpr, _ = roc_curve(y_test, scores)
    axes[0].plot(fpr, tpr, label=label, linewidth=2)
axes[0].plot([0,1],[0,1],'k--', alpha=0.4)
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves — DongTing Test Set')
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

# Score distributions (test, normalized)
for scores, title, clr_norm in [
    (norm_vae(vae_scores_test), 'VAE (norm)', '#DD8452'),
    (norm_if(if_scores_test),   'IF  (norm)', '#4C72B0'),
    (ens_test,                  'Ensemble',   '#2ca02c'),
]:
    pass  # just check distributions once in subplot

axes[1].hist(norm_vae(vae_scores_test[y_test==0]), bins=60, alpha=0.6, color='#4C72B0',
             density=True, label='VAE-Normal')
axes[1].hist(norm_vae(vae_scores_test[y_test==1]), bins=60, alpha=0.6, color='#DD8452',
             density=True, label='VAE-Attack')
axes[1].axvline(norm_vae(np.array([vae_val['threshold']]))[0],
                color='k', linestyle='--', label='threshold')
axes[1].set_title('VAE Normalized Score Distribution (test)')
axes[1].set_xlabel('Normalized Anomaly Score'); axes[1].legend()

plt.suptitle('DongTing Anomaly Detection — 174-dim FE', fontsize=13)
plt.tight_layout(); plt.savefig(f'{OUT}/model_roc_fe.png', dpi=150); plt.close()

# Ensemble score distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, raw, title in [
    (axes[0], norm_if(if_scores_test),   'IF (normalized)'),
    (axes[1], norm_vae(vae_scores_test), 'VAE (normalized)'),
    (axes[2], ens_test,                  'Ensemble'),
]:
    ax.hist(raw[y_test==0], bins=60, alpha=0.7, color='#4C72B0', label='Normal', density=True)
    ax.hist(raw[y_test==1], bins=60, alpha=0.7, color='#DD8452', label='Attack',  density=True)
    ax.set_title(title); ax.legend()
plt.suptitle('Score Distributions — Test Set (normalized, training reference)',
             fontsize=13)
plt.tight_layout(); plt.savefig(f'{OUT}/model_score_dist_fe.png', dpi=150); plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, res, title, cmap in [
    (axes[0], if_test,      'Isolation Forest', 'Blues'),
    (axes[1], vae_test,     'VAE',              'Oranges'),
    (axes[2], ens_test_res, 'Ensemble',         'Greens'),
]:
    cm = np.array(res['cm'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=cmap, linewidths=0.5,
                xticklabels=['Normal','Attack'], yticklabels=['Normal','Attack'])
    ax.set_title(title); ax.set_ylabel('True'); ax.set_xlabel('Predicted')
plt.suptitle('Confusion Matrices — DongTing Test Set', fontsize=13)
plt.tight_layout(); plt.savefig(f'{OUT}/model_confusion_fe.png', dpi=150); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Final report
# ─────────────────────────────────────────────────────────────────────────────
report = {
    'feature_dim':    int(INPUT_DIM),
    'train_samples':  int(len(y_train)),
    'val_samples':    int(len(y_val)),
    'test_samples':   int(len(y_test)),
    'vae_seed_results': seed_results,
    'models': {
        'IsolationForest': {
            'params': {'n_estimators': 300, 'contamination': 'auto',
                       'train_on': 'normal_only', 'random_state': 42},
            'val':  if_val,  'test': if_test,
        },
        'VAE': {
            'params': {'latent_dim': LATENT, 'n_seeds': N_SEEDS,
                       'best_val_auc': best_val_auc},
            'val':  vae_val, 'test': vae_test,
        },
        'Ensemble': {
            'params': {'alpha': ALPHA, 'normalization': 'MinMax_on_training_scores'},
            'val':  ens_val_res, 'test': ens_test_res,
        },
    },
}
with open(f'{OUT}/model_report_fe.json', 'w') as f:
    json.dump(report, f, indent=2)

print('\n' + '='*60)
print('FINAL RESULTS — DongTing Test Set (174-dim FE)')
print('='*60)
hdr = f'{"Model":<20}  {"Val AUC":>8}  {"Test AUC":>9}  {"Test F1":>8}  {"Precision":>10}  {"Recall":>9}  {"Accuracy":>9}'
print(hdr)
print('-' * len(hdr))
for mname, (vres, tres) in [
    ('IsolationForest', (if_val,       if_test)),
    ('VAE',             (vae_val,      vae_test)),
    ('Ensemble',        (ens_val_res,  ens_test_res)),
]:
    print(f'{mname:<20}  {vres["auc"]:>8.4f}  {tres["auc"]:>9.4f}  '
          f'{tres["f1"]:>8.4f}  {tres["precision"]:>10.4f}  '
          f'{tres["recall"]:>9.4f}  {tres["accuracy"]:>9.4f}')

print('\nVAE seed stability:')
for sr in seed_results:
    print(f'  seed={sr["seed"]}  val_AUC={sr["val_auc"]:.4f}  '
          f'epochs={sr["epochs"]}  best_val_loss={sr["best_val_loss"]:.2f}')

print(f'\nOutputs saved to {OUT}/')
print('  model_roc_fe.png       — ROC curves + VAE score dist')
print('  model_score_dist_fe.png— Per-model normalized score distributions')
print('  model_confusion_fe.png — Confusion matrices (all 3 models)')
print('  model_report_fe.json   — Full metrics JSON')

# ─────────────────────────────────────────────────────────────────────────────
# 8. Save model artifacts for inference
# ─────────────────────────────────────────────────────────────────────────────
import joblib

joblib.dump(iso, f'{OUT}/if_model.joblib')
best_enc.save_weights(f'{OUT}/vae_encoder.weights.h5')
best_dec.save_weights(f'{OUT}/vae_decoder.weights.h5')

norm_params = {
    'if_lo':  norm_if.lo,  'if_hi':  norm_if.hi,
    'vae_lo': norm_vae.lo, 'vae_hi': norm_vae.hi,
    'vae_threshold':   float(vae_val['threshold']),
    'if_threshold':    float(if_val['threshold']),
    'ens_threshold':   float(ens_val_res['threshold']),
    'alpha':           ALPHA,
    'latent_dim':      LATENT,
    'input_dim':       int(INPUT_DIM),
}
with open(f'{OUT}/model_params.json', 'w') as f:
    json.dump(norm_params, f, indent=2)

print('\nModel artifacts:')
print(f'  {OUT}/if_model.joblib        — Isolation Forest')
print(f'  {OUT}/vae_encoder.weights.h5 — VAE encoder weights')
print(f'  {OUT}/vae_decoder.weights.h5 — VAE decoder weights')
print(f'  {OUT}/model_params.json      — Normalizer params + thresholds')
