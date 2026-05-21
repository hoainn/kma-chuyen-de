"""
Comparative evaluation of all model variants on multiple test splits.

Variants evaluated:
  1. vae_dongting   (v1, the reproduction baseline)
  2. vae_container  (v2, retrained on container data)
  3. iforest        (v1 Isolation Forest reused)
  4. lstm           (v2 supervised)
  5. cnn1d          (v2 supervised)

Test splits evaluated (when available):
  A. DongTing test       (outputs/X_test_scaled.npy + y_test.npy)
  B. Container test      (outputs/X_container_test_scaled.npy + y_container_test.npy)
  C. CVE traces          (collect/recordings/cve_*.npy)

Outputs: outputs/evaluation_v2.json + outputs/v2_roc_comparison.png
"""
import collections
import glob
import json
import os

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'outputs')
RECORDINGS = os.path.join(ROOT, 'collect', 'recordings')


def load_vae_dongting():
    """Returns (encoder, decoder, RobustScaler params) for v1 VAE."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    with open(os.path.join(OUT, 'model_params.json')) as f:
        mp = json.load(f)
    INPUT_DIM = mp['input_dim']; LATENT = mp['latent_dim']

    def enc():
        inp = keras.Input(shape=(INPUT_DIM,))
        x = layers.Dense(128, activation='selu')(inp)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='selu')(x)
        x = layers.Dropout(0.2)(x)
        mu = layers.Dense(LATENT, name='mu')(x)
        lv = layers.Dense(LATENT, name='lv')(x)
        return keras.Model(inp, [mu, lv])

    def dec():
        inp = keras.Input(shape=(LATENT,))
        x = layers.Dense(64, activation='selu')(inp)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='selu')(x)
        x = layers.Dropout(0.2)(x)
        out = layers.Dense(INPUT_DIM)(x)
        return keras.Model(inp, out)

    encoder, decoder = enc(), dec()
    encoder.load_weights(os.path.join(OUT, 'vae_encoder.weights.h5'))
    decoder.load_weights(os.path.join(OUT, 'vae_decoder.weights.h5'))
    return encoder, decoder, mp['vae_lo'], mp['vae_hi'], mp.get('vae_threshold')


def load_vae_container():
    """Same arch, container weights."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    p = os.path.join(OUT, 'model_params_v2_container.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        mp = json.load(f)
    INPUT_DIM = mp['input_dim']; LATENT = mp['latent_dim']

    def enc():
        inp = keras.Input(shape=(INPUT_DIM,))
        x = layers.Dense(128, activation='selu')(inp)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='selu')(x)
        x = layers.Dropout(0.2)(x)
        mu = layers.Dense(LATENT, name='mu')(x)
        lv = layers.Dense(LATENT, name='lv')(x)
        return keras.Model(inp, [mu, lv])

    def dec():
        inp = keras.Input(shape=(LATENT,))
        x = layers.Dense(64, activation='selu')(inp)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='selu')(x)
        x = layers.Dropout(0.2)(x)
        out = layers.Dense(INPUT_DIM)(x)
        return keras.Model(inp, out)

    encoder, decoder = enc(), dec()
    encoder.load_weights(os.path.join(OUT, 'vae_encoder_v2_container.weights.h5'))
    decoder.load_weights(os.path.join(OUT, 'vae_decoder_v2_container.weights.h5'))
    return encoder, decoder, mp['vae_lo'], mp['vae_hi'], mp.get('vae_threshold')


def vae_score(encoder, decoder, X, lo, hi, n=5):
    import tensorflow as tf
    x = tf.constant(X, dtype=tf.float32)
    errs = []
    for _ in range(n):
        mu, lv = encoder(x, training=False)
        lv = tf.clip_by_value(lv, -8, 8)
        z = mu + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(mu))
        xh = decoder(z, training=False)
        errs.append(tf.reduce_sum(tf.square(x - xh), axis=1).numpy())
    raw = np.mean(errs, axis=0)
    raw = np.nan_to_num(raw, nan=1e6, posinf=1e6, neginf=0.0)
    norm = np.clip((raw - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return raw, norm


def metrics_for(scores, y):
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    from sklearn.metrics import classification_report, confusion_matrix
    auc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)
    fpr, tpr, th = roc_curve(y, scores)
    thr = float(th[np.argmax(tpr - fpr)])
    preds = (scores >= thr).astype(int)
    rep = classification_report(y, preds, target_names=['Normal', 'Attack'],
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y, preds).tolist()
    return {'auc': float(auc), 'ap': float(ap),
            'f1': rep['Attack']['f1-score'],
            'precision': rep['Attack']['precision'],
            'recall': rep['Attack']['recall'],
            'threshold': thr, 'cm': cm}


def measure_cve_latency(scores_per_window, threshold, label='cve'):
    """Number of windows from start until first window with score >= threshold."""
    for i, s in enumerate(scores_per_window):
        if s >= threshold:
            return i
    return None  # never detected


def main():
    import tensorflow as tf
    results = {'models': {}, 'cve_latency': {}}

    # ── Load all models ──
    print('Loading models...')
    vae_dt = load_vae_dongting()
    vae_co = load_vae_container()
    iso = joblib.load(os.path.join(OUT, 'if_model.joblib'))

    with open(os.path.join(OUT, 'model_params.json')) as f:
        mp_v1 = json.load(f)
    if_lo, if_hi = mp_v1['if_lo'], mp_v1['if_hi']

    # Try supervised models
    lstm_model = cnn1d_model = None
    sup_p = os.path.join(OUT, 'train_supervised_report.json')
    if os.path.exists(sup_p):
        from tensorflow import keras
        from tensorflow.keras import layers
        def build_lstm(d):
            inp = keras.Input(shape=(d,))
            x = layers.Reshape((d, 1))(inp); x = layers.LSTM(64)(x)
            x = layers.Dense(32, activation='relu')(x); x = layers.Dropout(0.3)(x)
            return keras.Model(inp, layers.Dense(1, activation='sigmoid')(x))
        def build_cnn(d):
            inp = keras.Input(shape=(d,))
            x = layers.Reshape((d, 1))(inp)
            x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
            x = layers.MaxPool1D(2)(x)
            x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
            x = layers.GlobalMaxPool1D()(x)
            x = layers.Dense(32, activation='relu')(x); x = layers.Dropout(0.3)(x)
            return keras.Model(inp, layers.Dense(1, activation='sigmoid')(x))
        with open(sup_p) as f:
            sup_report = json.load(f)
        d = sup_report['input_dim']
        lstm_model = build_lstm(d)
        lstm_model.load_weights(os.path.join(OUT, 'lstm_v2.weights.h5'))
        cnn1d_model = build_cnn(d)
        cnn1d_model.load_weights(os.path.join(OUT, 'cnn1d_v2.weights.h5'))

    # ── Test sets ──
    test_sets = {}
    for tag, fx, fy in [
        ('dongting', 'X_test_scaled.npy', 'y_test.npy'),
        ('container', 'X_container_test_scaled.npy', 'y_container_test.npy'),
    ]:
        try:
            Xt = np.load(os.path.join(OUT, fx)).astype(np.float32)
            yt = np.load(os.path.join(OUT, fy))
            test_sets[tag] = (Xt, yt)
            print(f'  {tag} test: {Xt.shape}, attack_rate={yt.mean():.3f}')
        except FileNotFoundError:
            print(f'  {tag} test: missing — skipping')

    # ── Per-model × per-test-set ──
    def evaluate_one(name, scorer, supports_norm=True):
        results['models'][name] = {}
        for tag, (Xt, yt) in test_sets.items():
            try:
                raw, norm = scorer(Xt)
                m = metrics_for(norm, yt)
                results['models'][name][tag] = m
                print(f'    {name:<20} {tag:10s}  AUC={m["auc"]:.4f}  F1={m["f1"]:.4f}')
            except Exception as e:
                print(f'    {name} {tag}: ERROR {e}')
                results['models'][name][tag] = {'error': str(e)}

    # Define scorers
    def score_vae_dt(X):
        enc, dec, lo, hi, thr = vae_dt
        return vae_score(enc, dec, X, lo, hi)

    def score_vae_co(X):
        if vae_co is None:
            raise RuntimeError('vae_container weights not available')
        enc, dec, lo, hi, thr = vae_co
        return vae_score(enc, dec, X, lo, hi)

    def score_iforest(X):
        raw = -iso.decision_function(X)
        norm = np.clip((raw - if_lo) / (if_hi - if_lo + 1e-9), 0.0, 1.0)
        return raw, norm

    def score_lstm(X):
        p = lstm_model.predict(X, batch_size=256, verbose=0).ravel()
        return p, p

    def score_cnn1d(X):
        p = cnn1d_model.predict(X, batch_size=256, verbose=0).ravel()
        return p, p

    print('\nPer-model evaluation:')
    evaluate_one('vae_dongting', score_vae_dt)
    if vae_co is not None:
        evaluate_one('vae_container', score_vae_co)
    evaluate_one('iforest', score_iforest)
    if lstm_model is not None:
        evaluate_one('lstm', score_lstm)
    if cnn1d_model is not None:
        evaluate_one('cnn1d', score_cnn1d)

    # ── CVE traces ──
    print('\nCVE traces:')
    from collections import Counter
    from scipy.stats import entropy as scipy_entropy

    # Reuse feature builder from build_features
    import sys
    sys.path.insert(0, HERE)
    from build_features import build_feature, load_fe_config, sliding_windows
    fe = load_fe_config()
    scaler = joblib.load(os.path.join(OUT, 'scaler_fe.pkl'))

    cve_files = sorted(glob.glob(os.path.join(RECORDINGS, '*cve_*.npy')) +
                       glob.glob(os.path.join(RECORDINGS, '*attack_*.npy')))
    for cve_f in cve_files:
        seq = np.load(cve_f)
        wins = sliding_windows(seq, 200, 50)
        if not wins:
            print(f'  {os.path.basename(cve_f)}: only {len(seq)} syscalls, skip')
            continue
        feats = np.array([build_feature(w, fe, '6.12') for w in wins], dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        feats = scaler.transform(feats).astype(np.float32)
        tag = os.path.splitext(os.path.basename(cve_f))[0]
        results['cve_latency'][tag] = {'n_windows': len(wins)}
        for mname, scorer in [
            ('vae_dongting', score_vae_dt),
            ('vae_container', score_vae_co) if vae_co else (None, None),
            ('iforest', score_iforest),
            ('lstm', score_lstm) if lstm_model else (None, None),
            ('cnn1d', score_cnn1d) if cnn1d_model else (None, None),
        ]:
            if mname is None:
                continue
            try:
                raw, norm = scorer(feats)
                model_metrics = results['models'].get(mname, {})
                if 'dongting' in model_metrics and 'threshold' in model_metrics['dongting']:
                    thr = model_metrics['dongting']['threshold']
                else:
                    thr = 0.5
                lat = measure_cve_latency(norm, thr)
                max_score = float(np.max(norm))
                results['cve_latency'][tag][mname] = {
                    'detection_window': lat,
                    'max_score': max_score,
                    'threshold': thr,
                    'detected': lat is not None,
                }
                print(f'  {tag:30s} {mname:<15} detect@{lat}  max={max_score:.3f}')
            except Exception as e:
                print(f'  {tag} {mname}: ERROR {e}')

    # ── Save + plot ──
    with open(os.path.join(OUT, 'evaluation_v2.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved outputs/evaluation_v2.json')

    # ROC comparison plot (DongTing test if available)
    if 'dongting' in test_sets:
        Xt, yt = test_sets['dongting']
        from sklearn.metrics import roc_curve
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, scorer in [
            ('vae_dongting', score_vae_dt),
            ('vae_container', score_vae_co) if vae_co else (None, None),
            ('iforest', score_iforest),
            ('lstm', score_lstm) if lstm_model else (None, None),
            ('cnn1d', score_cnn1d) if cnn1d_model else (None, None),
        ]:
            if name is None:
                continue
            try:
                _, norm = scorer(Xt)
                fpr, tpr, _ = roc_curve(yt, norm)
                auc = results['models'][name]['dongting']['auc']
                ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
            except Exception:
                pass
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('v2 ROC Comparison — DongTing Test Split')
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, 'v2_roc_comparison.png'), dpi=150)
        plt.close()
        print('Saved outputs/v2_roc_comparison.png')


if __name__ == '__main__':
    main()
