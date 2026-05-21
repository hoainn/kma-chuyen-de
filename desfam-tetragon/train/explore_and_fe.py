"""
DongTing EDA + Feature Engineering
Explores raw syscall sequences, then builds a richer feature set.
Outputs:
  outputs/eda_*.png     — plots
  outputs/fe_report.json — feature statistics
  outputs/X_train.npy / X_val.npy / X_test.npy / y_*.npy  — ready-to-train arrays
"""
import os, json, warnings, collections
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy as scipy_entropy
warnings.filterwarnings('ignore')

from pymongo import MongoClient
from tqdm import tqdm

MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://mongo:27017/')
MONGO_DB  = os.environ.get('MONGO_DB', 'syzbot_DB')
OUT = 'outputs'; os.makedirs(OUT, exist_ok=True)

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

# Load syscall name → ID lookup (needed for attack sequences)
syscall_tbl = {}      # id  → name
syscall_name_to_id = {}  # name → id
with open('/workspace/data/dongting/syscall_64.tbl') as f:
    for line in f:
        parts = line.strip().split()
        if parts and parts[0].isdigit():
            sid, sname = int(parts[0]), parts[2]
            syscall_tbl[sid] = sname
            syscall_name_to_id[sname] = sid

def id_to_name(sid):
    return syscall_tbl.get(sid, str(sid))

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Load baseline + raw sequences
# ─────────────────────────────────────────────────────────────────────────────
print('='*60)
print('PART 1: Loading data')
print('='*60)

def to_int(v):
    try: return int(v)
    except: return 0

def parse_ml(s):
    """Parse pipe-separated numeric syscall IDs (normal sequences)."""
    if not s or not isinstance(s, str) or s.startswith(('sy_', 'ml_')):
        return []
    return [int(x) for x in s.split('|') if x.strip().lstrip('-').isdigit()]

def parse_names(s):
    """Parse pipe-separated syscall names (attack sequences) → IDs."""
    if not s or not isinstance(s, str):
        return []
    return [syscall_name_to_id[n.strip()] for n in s.split('|')
            if n.strip() in syscall_name_to_id]

# Load baseline
docs = list(db['kernel_convert_baseline'].find({}, {'_id': 0}))
df = pd.DataFrame(docs)
df['syscall_counts'] = df['kcb_syscall_counts'].apply(to_int)
df['syscall_sizes']  = df['kcb_syscall_sizes'].apply(to_int)
df['label']  = (df['kcb_seq_lables'] == 'Attach').astype(int)
df['split']  = df['kcb_seq_class'].map({'DTDS-train':'train','DTDS-validation':'val','DTDS-test':'test'})
df['label_name'] = df['label'].map({0:'Normal',1:'Attack'})

print(f"Sequences: {len(df)}")
print(df.groupby(['label_name','split']).size().unstack(fill_value=0))

# Load all sequences into memory (both collections)
print('\nLoading normal sequences...')
normal_seqs = {}   # fname → list[int]
for doc in tqdm(db['kernel_syscall_normal_strace'].find(
        {}, {'kns_normal_file_name':1,'kns_normal_mlseq_list':1}),
        total=6850):
    fname = doc['kns_normal_file_name']
    normal_seqs[fname] = parse_ml(doc.get('kns_normal_mlseq_list',''))

print('Loading attack sequences...')
attack_seqs = {}   # pocname → list[int]
for doc in tqdm(db['kernel_syscallhook_bugpoc_trace_sum'].find(
        {}, {'kshs_poclog_name':1,'kshs_bugpoc_syscall_list':1}),
        total=12116):
    name = doc['kshs_poclog_name']
    # mlcode is a filename reference for most docs; use name-based list + id lookup
    attack_seqs[name] = parse_names(doc.get('kshs_bugpoc_syscall_list', ''))

def get_seq(row):
    if row['label'] == 0:
        # baseline.kcb_bug_name == normal_strace.kns_normal_file_name (e.g. "bug23.log")
        return normal_seqs.get(row['kcb_bug_name'], [])
    else:
        # baseline.kcb_bug_name == attack_strace.kshs_poclog_name
        return attack_seqs.get(row['kcb_bug_name'], [])

print('Joining sequences to metadata...')
df['seq'] = [get_seq(r) for _, r in tqdm(df.iterrows(), total=len(df))]
df['seq_len']    = df['seq'].apply(len)
df['n_unique']   = df['seq'].apply(lambda s: len(set(s)))
df['empty_seq']  = df['seq_len'] == 0

print(f"\nSequences with content: {(~df.empty_seq).sum()} / {len(df)}")
print(f"Empty sequences: {df.empty_seq.sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Exploratory Data Analysis
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('PART 2: EDA')
print('='*60)

colors = {'Normal':'#4C72B0', 'Attack':'#DD8452'}

# -- 2a. Sequence length distribution -----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, lbl, cap in [(axes[0],'Normal',5000),(axes[1],'Attack',200000)]:
    sub = df[df.label_name==lbl]['seq_len'].clip(upper=cap)
    ax.hist(sub, bins=60, color=colors[lbl], alpha=0.85, edgecolor='white')
    ax.axvline(sub.median(), color='black', linestyle='--', label=f'median={sub.median():.0f}')
    ax.set_title(f'{lbl} — Sequence Length (capped {cap:,})')
    ax.set_xlabel('Syscall Count'); ax.set_ylabel('Sequences')
    ax.legend()
plt.suptitle('Sequence Length Distribution by Class', fontsize=13)
plt.tight_layout(); plt.savefig(f'{OUT}/eda_seq_len.png', dpi=150); plt.close()

stats = df.groupby('label_name')['seq_len'].describe().round(1)
print('\nSequence length stats:')
print(stats)

# -- 2b. Unique syscall count per sequence ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, lbl in [(axes[0],'Normal'),(axes[1],'Attack')]:
    sub = df[df.label_name==lbl]['n_unique']
    ax.hist(sub, bins=50, color=colors[lbl], alpha=0.85, edgecolor='white')
    ax.axvline(sub.median(), color='black', linestyle='--', label=f'median={sub.median():.0f}')
    ax.set_title(f'{lbl} — Unique Syscall IDs per Sequence')
    ax.set_xlabel('# Unique Syscall IDs'); ax.legend()
plt.suptitle('Unique Syscall Diversity by Class', fontsize=13)
plt.tight_layout(); plt.savefig(f'{OUT}/eda_unique_syscalls.png', dpi=150); plt.close()

# -- 2c. Most frequent syscall IDs per class ----------------------------------
def top_syscalls(seqs, n=30):
    cnt = collections.Counter()
    for s in seqs:
        cnt.update(set(s))   # per-sequence presence (not raw count)
    return cnt.most_common(n)

normal_top = top_syscalls(df[df.label==0]['seq'])
attack_top = top_syscalls(df[df.label==1]['seq'])

print('\nTop-20 syscall IDs (by sequence presence):')
print('  Normal:', [(id_, c) for id_,c in normal_top[:20]])
print('  Attack:', [(id_, c) for id_,c in attack_top[:20]])

# Differential: syscalls that appear far more in attacks than normal
total_normal = (df.label==0).sum()
total_attack = (df.label==1).sum()
normal_freq  = {k: v/total_normal for k,v in dict(top_syscalls(df[df.label==0]['seq'], n=400)).items()}
attack_freq  = {k: v/total_attack  for k,v in dict(top_syscalls(df[df.label==1]['seq'], n=400)).items()}
all_ids = set(normal_freq) | set(attack_freq)
diff = {k: attack_freq.get(k,0) - normal_freq.get(k,0) for k in all_ids}
top_attack_excl = sorted(diff.items(), key=lambda x: x[1], reverse=True)[:20]
top_normal_excl = sorted(diff.items(), key=lambda x: x[1])[:20]

print('\nSyscalls disproportionately in ATTACK (attack_freq - normal_freq):')
for sid, d in top_attack_excl:
    print(f'  {id_to_name(sid):20s} (id={sid:4d})  diff={d:+.3f}  '
          f'atk={attack_freq.get(sid,0):.3f} nor={normal_freq.get(sid,0):.3f}')

print('\nSyscalls disproportionately in NORMAL:')
for sid, d in top_normal_excl:
    print(f'  {id_to_name(sid):20s} (id={sid:4d})  diff={d:+.3f}  '
          f'atk={attack_freq.get(sid,0):.3f} nor={normal_freq.get(sid,0):.3f}')

# Differential bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, data, title, clr in [
    (axes[0], top_attack_excl, 'Top Attack-Exclusive Syscalls', '#DD8452'),
    (axes[1], top_normal_excl, 'Top Normal-Exclusive Syscalls', '#4C72B0'),
]:
    names = [id_to_name(k) for k,_ in data]
    vals  = [abs(v) for _,v in data]
    ax.barh(names[::-1], vals[::-1], color=clr, alpha=0.85)
    ax.set_title(title); ax.set_xlabel('|attack_rate − normal_rate|')
plt.suptitle('Discriminative Syscalls — DongTing', fontsize=13)
plt.tight_layout(); plt.savefig(f'{OUT}/eda_discriminative_syscalls.png', dpi=150); plt.close()

# -- 2d. Syscall bigrams (transition pairs) -----------------------------------
print('\nComputing syscall bigrams...')
BIGRAM_SAMPLE = 2000
def top_bigrams(seqs, n=20, limit=BIGRAM_SAMPLE):
    cnt = collections.Counter()
    for s in seqs[:limit]:
        for a, b in zip(s, s[1:]):
            cnt[(a, b)] += 1
    return cnt.most_common(n)

norm_bg = top_bigrams(df[df.label==0]['seq'].tolist())
atk_bg  = top_bigrams(df[df.label==1]['seq'].tolist())
print('Top-10 Normal bigrams:',  [(f'{id_to_name(a)}→{id_to_name(b)}', c) for (a,b),c in norm_bg[:10]])
print('Top-10 Attack bigrams:',  [(f'{id_to_name(a)}→{id_to_name(b)}', c) for (a,b),c in atk_bg[:10]])

# -- 2e. Kernel version heatmap -----------------------------------------------
ver_label = df.groupby(['kcb_master_line_ver','label_name']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(12, 5))
ver_label.plot(kind='bar', ax=ax, color=[colors['Normal'],colors['Attack']], alpha=0.85, edgecolor='white')
ax.set_title('Sequences per Kernel Version', fontsize=13)
ax.set_xlabel('Kernel Version'); ax.set_ylabel('Count'); ax.legend(title='Label')
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(f'{OUT}/eda_kernel_versions.png', dpi=150); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('PART 3: Feature Engineering')
print('='*60)

# Define 5 feature groups

## A. Syscall frequency vector — top-60 by overall presence
print('A) Syscall frequency vector...')
all_top = pd.Series(
    {k: normal_freq.get(k,0) + attack_freq.get(k,0) for k in all_ids}
).sort_values(ascending=False)
TOP_IDS = all_top.head(60).index.tolist()

def feat_freq(seq):
    if not seq: return np.zeros(len(TOP_IDS), dtype=np.float32)
    total = len(seq)
    return np.array([seq.count(s)/total for s in TOP_IDS], dtype=np.float32)

## B. Discriminative syscall flags — binary presence of top attack-exclusive syscalls
print('B) Discriminative flags...')
DISC_IDS = [k for k,_ in top_attack_excl[:20]] + [k for k,_ in top_normal_excl[:20]]

def feat_disc(seq):
    s = set(seq)
    return np.array([1.0 if d in s else 0.0 for d in DISC_IDS], dtype=np.float32)

## C. Statistical features
print('C) Statistical features...')
def feat_stats(seq, counts, sizes):
    if not seq:
        return np.zeros(8, dtype=np.float32)
    arr  = np.array(seq)
    freq = np.bincount(arr[arr >= 0], minlength=548).astype(np.float32)
    freq /= (freq.sum() + 1e-9)
    ent  = float(scipy_entropy(freq + 1e-9))
    return np.array([
        np.log1p(len(seq)),           # log sequence length
        np.log1p(len(set(seq))),      # log unique syscall count
        len(set(seq)) / (len(seq) + 1e-9),  # diversity ratio
        ent,                          # Shannon entropy of syscall distribution
        np.log1p(counts),             # log total syscall count (metadata)
        np.log1p(sizes),              # log total byte size (metadata)
        float(np.mean(arr)),          # mean syscall ID
        float(np.std(arr)),           # std syscall ID
    ], dtype=np.float32)

## D. Bigram transition features — top-40 bigrams (attack + normal combined)
print('D) Bigram transition features...')
all_bigrams_norm = dict(top_bigrams(df[df.label==0]['seq'].tolist(), n=200))
all_bigrams_atk  = dict(top_bigrams(df[df.label==1]['seq'].tolist(), n=200))
all_bg_keys = set(all_bigrams_norm) | set(all_bigrams_atk)
bg_diff = {k: all_bigrams_atk.get(k,0)/max(1,total_attack)
              - all_bigrams_norm.get(k,0)/max(1,total_normal)
           for k in all_bg_keys}
TOP_BIGRAMS = [k for k,_ in sorted(bg_diff.items(), key=lambda x: abs(x[1]), reverse=True)[:40]]

def feat_bigrams(seq):
    if len(seq) < 2: return np.zeros(len(TOP_BIGRAMS), dtype=np.float32)
    cnt = collections.Counter(zip(seq, seq[1:]))
    total = sum(cnt.values()) + 1e-9
    return np.array([cnt.get(bg,0)/total for bg in TOP_BIGRAMS], dtype=np.float32)

## E. Kernel version one-hot
print('E) Kernel version one-hot...')
ver_cols = sorted(df['kcb_master_line_ver'].unique())
def feat_ver(ver):
    return np.array([1.0 if v == ver else 0.0 for v in ver_cols], dtype=np.float32)

# -- Build full feature matrix ------------------------------------------------
print('\nBuilding full feature matrix...')
feature_groups = {
    'freq_60':    len(TOP_IDS),
    'disc_40':    len(DISC_IDS),
    'stats_8':    8,
    'bigram_40':  len(TOP_BIGRAMS),
    'ver_onehot': len(ver_cols),
}
total_dims = sum(feature_groups.values())
print(f'Feature groups: {feature_groups}')
print(f'Total dimensions: {total_dims}')

rows = []
for _, row in tqdm(df.iterrows(), total=len(df), desc='building features'):
    seq = row['seq']
    f = np.concatenate([
        feat_freq(seq),
        feat_disc(seq),
        feat_stats(seq, row['syscall_counts'], row['syscall_sizes']),
        feat_bigrams(seq),
        feat_ver(row['kcb_master_line_ver']),
    ])
    rows.append(f)

X = np.array(rows, dtype=np.float32)
y = df['label'].values.astype(int)
splits = df['split'].values

print(f'Feature matrix: {X.shape}  |  NaN: {np.isnan(X).sum()}  |  Inf: {np.isinf(X).sum()}')
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# -- Split --------------------------------------------------------------------
X_train = X[splits=='train'];  y_train = y[splits=='train']
X_val   = X[splits=='val'];    y_val   = y[splits=='val']
X_test  = X[splits=='test'];   y_test  = y[splits=='test']

print(f'Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}')

# -- Feature correlation / importance heatmap ---------------------------------
print('\nComputing feature importance via mutual information...')
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

mi = mutual_info_classif(X_train_sc, y_train, discrete_features=False, random_state=42)
mi_series = pd.Series(mi, name='MI').sort_values(ascending=False)

# Top-30 most informative features
top30_idx  = mi_series.head(30).index.tolist()
feat_names = (
    [f'freq_{id_to_name(t)}' for t in TOP_IDS] +
    [f'disc_{id_to_name(t)}'  for t in DISC_IDS] +
    ['stat_log_len','stat_log_unique','stat_diversity','stat_entropy',
     'stat_log_count','stat_log_size','stat_mean_id','stat_std_id'] +
    [f'bg_{id_to_name(a)}->{id_to_name(b)}' for a,b in TOP_BIGRAMS] +
    [f'ver_{v}' for v in ver_cols]
)
top30_names = [feat_names[i] for i in top30_idx]
top30_mi    = mi_series.iloc[:30].values

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(top30_names[::-1], top30_mi[::-1], color='#4C72B0', alpha=0.85)
ax.set_title('Top-30 Features by Mutual Information (vs label)', fontsize=13)
ax.set_xlabel('Mutual Information Score')
plt.tight_layout(); plt.savefig(f'{OUT}/fe_feature_importance.png', dpi=150); plt.close()

print('\nTop-20 most informative features:')
for name, score in zip(top30_names[:20], top30_mi[:20]):
    print(f'  {score:.4f}  {name}')

# -- Class separation per group -----------------------------------------------
group_slices = {}
start = 0
for gname, gdim in feature_groups.items():
    group_slices[gname] = (start, start + gdim)
    start += gdim

print('\nMean MI per feature group:')
for gname, (s, e) in group_slices.items():
    group_mi = mi[s:e].mean()
    print(f'  {gname:15s}  mean_MI={group_mi:.4f}  max_MI={mi[s:e].max():.4f}')

# -- Save arrays + report -----------------------------------------------------
np.save(f'{OUT}/X_train.npy', X_train)
np.save(f'{OUT}/X_val.npy',   X_val)
np.save(f'{OUT}/X_test.npy',  X_test)
np.save(f'{OUT}/y_train.npy', y_train)
np.save(f'{OUT}/y_val.npy',   y_val)
np.save(f'{OUT}/y_test.npy',  y_test)
np.save(f'{OUT}/X_train_scaled.npy', X_train_sc)
np.save(f'{OUT}/X_val_scaled.npy',   scaler.transform(X_val))
np.save(f'{OUT}/X_test_scaled.npy',  scaler.transform(X_test))

import joblib
joblib.dump(scaler, f'{OUT}/scaler_fe.pkl')
json.dump({
    'feature_groups': feature_groups,
    'total_dims': total_dims,
    'top_ids': TOP_IDS,
    'disc_ids': DISC_IDS,
    'top_bigrams': [list(bg) for bg in TOP_BIGRAMS],
    'ver_cols': ver_cols,
    'top30_features': list(zip(top30_names, [round(float(v),4) for v in top30_mi])),
    'group_mi': {g: {'mean': round(float(mi[s:e].mean()),4), 'max': round(float(mi[s:e].max()),4)}
                 for g, (s,e) in group_slices.items()},
    'seq_len_stats': df.groupby('label_name')['seq_len'].describe().round(1).to_dict(),
}, open(f'{OUT}/fe_report.json','w'), indent=2)

print('\n' + '='*60)
print('DONE — outputs saved to', OUT)
print(f'Feature matrix shape: {X.shape}')
print('='*60)
