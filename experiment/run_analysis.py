"""
SyscallAD analysis — DongTing dataset
Isolation Forest vs VAE: same methodology as syscallad_dongting.ipynb
Outputs metrics to outputs/summary.json
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from pymongo import MongoClient
from tqdm import tqdm

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import MeanSquaredError

np.random.seed(42)
tf.random.set_seed(42)

MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://mongo:27017/')
MONGO_DB  = os.environ.get('MONGO_DB', 'syzbot_DB')
OUT       = 'outputs'
os.makedirs(OUT, exist_ok=True)

print(f'TF {tf.__version__} | MongoDB {MONGO_URI}')

# ── 1. Connect & load ─────────────────────────────────────────────────────────
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

print('Loading baseline metadata...')
docs = list(db['kernel_convert_baseline'].find({}, {'_id': 0}))
df = pd.DataFrame(docs)

def to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0

df['syscall_counts'] = df['kcb_syscall_counts'].apply(to_int)
df['syscall_sizes']  = df['kcb_syscall_sizes'].apply(to_int)
df['label'] = (df['kcb_seq_lables'] == 'Attach').astype(int)
df['split'] = df['kcb_seq_class'].map({
    'DTDS-train': 'train', 'DTDS-validation': 'val', 'DTDS-test': 'test'
})
ver_dummies = pd.get_dummies(df['kcb_master_line_ver'], prefix='ver')
df = pd.concat([df, ver_dummies], axis=1)

print(f'Baseline loaded: {len(df)} sequences | '
      f'Normal={int((df.label==0).sum())} Attack={int((df.label==1).sum())}')

# ── 2. Feature engineering ────────────────────────────────────────────────────
HAS_SEQ = (db['kernel_syscall_normal_strace'].estimated_document_count() > 0 and
           db['kernel_syscallhook_bugpoc_trace_sum'].estimated_document_count() > 0)
print(f'Full sequence data: {HAS_SEQ}')

def parse_mlcode(s):
    if not s or not isinstance(s, str) or s.startswith(('sy_', 'ml_')):
        return []
    parts = s.split('|')
    result = []
    for p in parts:
        p = p.strip()
        if p.lstrip('-').isdigit():
            result.append(int(p))
    return result

SAMPLE_LIMIT = 5000
top_ids = []
if HAS_SEQ:
    print('Scanning top syscall IDs (sample)...')
    all_ids = []
    for doc in db['kernel_syscall_normal_strace'].find(
            {}, {'kns_normal_mlseq_list': 1}, limit=SAMPLE_LIMIT):
        all_ids.extend(parse_mlcode(doc.get('kns_normal_mlseq_list', '')))
    for doc in db['kernel_syscallhook_bugpoc_trace_sum'].find(
            {}, {'kshs_bugpoc_syscall_mlcode': 1}, limit=SAMPLE_LIMIT):
        all_ids.extend(parse_mlcode(doc.get('kshs_bugpoc_syscall_mlcode', '')))
    top_ids = pd.Series(all_ids).value_counts().head(50).index.tolist()
    print(f'Top-50 syscall IDs selected.')

def freq_vec(mlcode_str):
    ids = parse_mlcode(mlcode_str)
    if not ids:
        return np.zeros(len(top_ids), dtype=np.float32)
    total = len(ids)
    return np.array([ids.count(s) / total for s in top_ids], dtype=np.float32)

def build_features():
    ver_cols = [c for c in df.columns if c.startswith('ver_')]
    if HAS_SEQ:
        print('Building sequence lookup tables...')
        n_col = db['kernel_syscall_normal_strace']
        a_col = db['kernel_syscallhook_bugpoc_trace_sum']
        nlookup = {
            d['kns_normal_file_name']: d.get('kns_normal_mlseq_list', '')
            for d in tqdm(n_col.find({}, {'kns_normal_file_name':1,'kns_normal_mlseq_list':1}),
                          total=n_col.estimated_document_count(), desc='normal')
        }
        alookup = {
            d['kshs_poclog_name']: d.get('kshs_bugpoc_syscall_mlcode', '')
            for d in tqdm(a_col.find({}, {'kshs_poclog_name':1,'kshs_bugpoc_syscall_mlcode':1}),
                          total=a_col.estimated_document_count(), desc='attack')
        }

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='features'):
        meta = np.array([
            np.log1p(row['syscall_counts']),
            np.log1p(row['syscall_sizes']),
        ], dtype=np.float32)
        ver = row[ver_cols].values.astype(np.float32)
        if HAS_SEQ:
            bname = row['kcb_bug_name']
            if row['label'] == 0:
                mlcode = nlookup.get(bname.rsplit('.', 1)[0], '')
            else:
                mlcode = alookup.get(bname, '')
            rows.append(np.concatenate([freq_vec(mlcode), meta, ver]))
        else:
            rows.append(np.concatenate([meta, ver]))
    return np.array(rows, dtype=np.float32)

print('Building feature matrix...')
X_all = build_features()
y_all = df['label'].values.astype(int)
split_all = df['split'].values

print(f'Feature matrix: {X_all.shape}')

# ── 3. Split ──────────────────────────────────────────────────────────────────
X_train_full = X_all[split_all == 'train'];  y_train_full = y_all[split_all == 'train']
X_val        = X_all[split_all == 'val'];    y_val        = y_all[split_all == 'val']
X_test       = X_all[split_all == 'test'];   y_test       = y_all[split_all == 'test']

scaler = StandardScaler()
scaler.fit(X_train_full[y_train_full == 0])
X_train = scaler.transform(X_train_full)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
X_train_vae = X_train[y_train_full == 0]

print(f'Train={len(X_train)} Val={len(X_val)} Test={len(X_test)} '
      f'(test normal={int((y_test==0).sum())} attack={int((y_test==1).sum())})')

# ── 4. Isolation Forest ───────────────────────────────────────────────────────
print('\nTraining Isolation Forest...')
iso = IsolationForest(
    n_estimators=200,
    contamination=min(0.5, float(y_train_full.mean())),
    max_samples='auto',
    random_state=42,
    n_jobs=-1,
)
iso.fit(X_train)
iso_scores = -iso.decision_function(X_test)
iso_preds  = (iso.predict(X_test) == -1).astype(int)
print('IF done.')
print(classification_report(y_test, iso_preds, target_names=['Normal','Attack']))

# ── 5. VAE ────────────────────────────────────────────────────────────────────
print('\nBuilding VAE...')
INPUT_DIM = X_train.shape[1]
HIDDEN    = 32
LATENT    = 16

enc_in   = keras.Input(shape=(INPUT_DIM,))
h        = layers.Dense(HIDDEN, activation='selu',
                        kernel_regularizer=keras.regularizers.l2(1e-4))(enc_in)
h        = layers.Dropout(0.2)(h)
z_mean   = layers.Dense(LATENT)(h)
z_lv     = layers.Dense(LATENT)(h)

def sample(args):
    m, lv = args
    return m + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(m))

z       = layers.Lambda(sample)([z_mean, z_lv])
encoder = Model(enc_in, [z_mean, z_lv, z], name='encoder')

dec_in  = keras.Input(shape=(LATENT,))
h2      = layers.Dense(HIDDEN, activation='selu',
                       kernel_regularizer=keras.regularizers.l2(1e-4))(dec_in)
h2      = layers.Dropout(0.2)(h2)
dec_out = layers.Dense(INPUT_DIM)(h2)
decoder = Model(dec_in, dec_out, name='decoder')

class VAE(Model):
    def __init__(self, enc, dec, **kw):
        super().__init__(**kw)
        self.enc, self.dec = enc, dec
        self.mse_t   = keras.metrics.Mean('mse_loss')
        self.kl_t    = keras.metrics.Mean('kl_loss')
        self.total_t = keras.metrics.Mean('total_loss')

    @property
    def metrics(self):
        return [self.total_t, self.mse_t, self.kl_t]

    def call(self, x, training=False):
        zm, zlv, z_s = self.enc(x, training=training)
        return self.dec(z_s, training=training)

    def train_step(self, data):
        if isinstance(data, tuple): data = data[0]
        with tf.GradientTape() as tape:
            zm, zlv, z_s = self.enc(data, training=True)
            recon = self.dec(z_s, training=True)
            mse = tf.reduce_mean(tf.reduce_sum(
                MeanSquaredError(reduction='none')(data, recon), axis=-1))
            kl = -0.5 * tf.reduce_mean(1 + zlv - tf.square(zm) - tf.exp(zlv))
            loss = mse + kl
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.mse_t.update_state(mse); self.kl_t.update_state(kl); self.total_t.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(1e-3))

val_sz = max(1, int(len(X_train_vae) * 0.1))
X_vval, X_vtrain = X_train_vae[:val_sz], X_train_vae[val_sz:]

print('Training VAE...')
hist = vae.fit(
    X_vtrain, X_vtrain,
    epochs=80,
    batch_size=256,
    validation_data=(X_vval, X_vval),
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0),
    ],
    verbose=0,
)
print(f'VAE stopped at epoch {len(hist.history["total_loss"])}')

def recon_error(X):
    zm, zlv, z_s = encoder.predict(X, verbose=0)
    recon = decoder.predict(z_s, verbose=0)
    return np.mean((X - recon) ** 2, axis=1)

train_errs  = recon_error(X_train_vae)
threshold   = np.percentile(train_errs, 99.5)
vae_scores  = recon_error(X_test)
vae_preds   = (vae_scores > threshold).astype(int)
print(f'VAE threshold: {threshold:.6f}')
print(classification_report(y_test, vae_preds, target_names=['Normal','Attack']))

# ── 6. Ensemble (DeSFAM α=0.7) ───────────────────────────────────────────────
mm = MinMaxScaler()
iso_norm = mm.fit_transform(iso_scores.reshape(-1,1)).ravel()
mm2 = MinMaxScaler()
vae_norm = mm2.fit_transform(vae_scores.reshape(-1,1)).ravel()

ALPHA = 0.7
ens   = ALPHA * vae_norm + (1 - ALPHA) * iso_norm
ens_thr   = np.percentile(ens[y_test == 0], 99.5)
ens_preds = (ens > ens_thr).astype(int)

# ── 7. Metrics summary ────────────────────────────────────────────────────────
def metrics(y_true, scores, preds):
    return {
        'accuracy':  round(float(accuracy_score(y_true, preds)),  4),
        'precision': round(float(precision_score(y_true, preds, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_true, preds, zero_division=0)),    4),
        'f1':        round(float(f1_score(y_true, preds, zero_division=0)),        4),
        'roc_auc':   round(float(roc_auc_score(y_true, scores)),                  4),
        'avg_prec':  round(float(average_precision_score(y_true, scores)),         4),
    }

summary = {
    'dataset': 'DongTing',
    'test_normal': int((y_test==0).sum()),
    'test_attack': int((y_test==1).sum()),
    'feature_dim': int(X_all.shape[1]),
    'models': {
        'isolation_forest':    metrics(y_test, iso_scores, iso_preds),
        'vae':                 metrics(y_test, vae_scores, vae_preds),
        f'ensemble_a{ALPHA}':  metrics(y_test, ens,        ens_preds),
    }
}

with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# ── 8. Plots ──────────────────────────────────────────────────────────────────
colors = ['#4C72B0', '#DD8452', '#55A868']
model_names = ['Isolation Forest', 'VAE', f'Ensemble α={ALPHA}']
all_scores  = [iso_scores, vae_scores, ens]
all_preds   = [iso_preds,  vae_preds,  ens_preds]

# ROC + PR comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, sc, color in zip(model_names, all_scores, colors):
    fpr, tpr, _ = roc_curve(y_test, sc)
    auc = roc_auc_score(y_test, sc)
    axes[0].plot(fpr, tpr, color=color, label=f'{name} (AUC={auc:.3f})')
    p, r, _ = precision_recall_curve(y_test, sc)
    ap = average_precision_score(y_test, sc)
    axes[1].plot(r, p, color=color, label=f'{name} (AP={ap:.3f})')
axes[0].plot([0,1],[0,1],'k--',alpha=0.4)
axes[0].set_title('ROC Curves'); axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].legend(fontsize=9)
axes[1].set_title('Precision-Recall Curves')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].legend(fontsize=9)
plt.suptitle('DongTing — IF vs VAE vs Ensemble', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/roc_pr.png', dpi=150)
plt.close()

# Bar chart
metric_names = ['roc_auc','avg_prec','f1','precision','recall','accuracy']
x = np.arange(len(metric_names))
w = 0.25
fig, ax = plt.subplots(figsize=(14, 5))
for i, (name, sc, preds, color) in enumerate(zip(model_names, all_scores, all_preds, colors)):
    m = metrics(y_test, sc, preds)
    vals = [m[k] for k in metric_names]
    bars = ax.bar(x + i*w, vals, w, label=name, color=color, alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=7)
ax.set_xticks(x + w); ax.set_xticklabels([k.replace('_',' ').upper() for k in metric_names])
ax.set_ylim(0, 1.15); ax.legend(); ax.grid(axis='y', alpha=0.3)
ax.set_title('Metric Comparison — DongTing Dataset', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/metrics_bar.png', dpi=150)
plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, name, preds, cmap in zip(axes, model_names, all_preds, ['Blues','Oranges','Greens']):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                xticklabels=['Normal','Attack'], yticklabels=['Normal','Attack'],
                cmap=cmap, linewidths=0.5)
    ax.set_title(name); ax.set_ylabel('True'); ax.set_xlabel('Predicted')
plt.suptitle('Confusion Matrices', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/confusion_matrices.png', dpi=150)
plt.close()

# ── 9. Print final table ──────────────────────────────────────────────────────
print('\n' + '='*65)
print('FINAL METRICS — DongTing Syscall Anomaly Detection')
print('='*65)
df_res = pd.DataFrame(summary['models']).T
df_res.index = ['Isolation Forest', 'VAE', f'Ensemble α={ALPHA}']
print(df_res.to_string())
print('='*65)
print(f'\nOutputs saved to {OUT}/')
