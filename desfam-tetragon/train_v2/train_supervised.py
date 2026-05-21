"""
Train two supervised baselines on the 174-dim windowed features:
  - LSTM(64)  -> Dense(32) -> Dense(1, sigmoid)
  - Conv1D(32,k=5)+Conv1D(64,k=3) -> Dense(32) -> Dense(1, sigmoid)

Training data = DongTing labelled windows (already in outputs/) +
container labelled windows (from build_features.py). The supervised
models can see attack examples directly, unlike the VAE.

Outputs:
  outputs/lstm_v2.weights.h5
  outputs/cnn1d_v2.weights.h5
  outputs/train_supervised_report.json
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


def load_combined():
    """Stack DongTing windows + container windows into one labelled dataset."""
    splits = {}
    for name in ('train', 'val', 'test'):
        X_dt = np.load(os.path.join(OUT, f'X_{name}_scaled.npy')).astype(np.float32)
        y_dt = np.load(os.path.join(OUT, f'y_{name}.npy')).astype(np.int8)
        try:
            X_co = np.load(os.path.join(OUT, f'X_container_{name}_scaled.npy')).astype(np.float32)
            y_co = np.load(os.path.join(OUT, f'y_container_{name}.npy')).astype(np.int8)
        except FileNotFoundError:
            X_co = np.empty((0, X_dt.shape[1]), dtype=np.float32)
            y_co = np.empty(0, dtype=np.int8)
        X = np.concatenate([X_dt, X_co], axis=0)
        y = np.concatenate([y_dt, y_co], axis=0)
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(X))
        splits[name] = (X[idx], y[idx])
        print(f'  {name}: {len(X)} samples ({len(X_dt)} DT + {len(X_co)} container) '
              f'attack_rate={y.mean():.3f}')
    return splits


def build_lstm(input_dim):
    inp = keras.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inp)
    x = layers.LSTM(64)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inp, out, name='lstm_v2')


def build_cnn1d(input_dim):
    inp = keras.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inp)
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inp, out, name='cnn1d_v2')


def fit_and_test(model, splits, name, save_path, epochs=50, batch=256):
    X_tr, y_tr = splits['train']
    X_va, y_va = splits['val']
    X_te, y_te = splits['test']
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.AUC(name='auc')])
    cb = [keras.callbacks.EarlyStopping(monitor='val_auc', mode='max',
                                        patience=6, restore_best_weights=True)]
    t0 = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                     epochs=epochs, batch_size=batch, callbacks=cb, verbose=2)
    elapsed = time.time() - t0
    proba = model.predict(X_te, batch_size=batch, verbose=0).ravel()
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 classification_report, confusion_matrix, roc_curve)
    auc = roc_auc_score(y_te, proba)
    ap = average_precision_score(y_te, proba)
    fpr, tpr, th = roc_curve(y_te, proba)
    thr = float(th[np.argmax(tpr - fpr)])
    preds = (proba >= thr).astype(int)
    rep = classification_report(y_te, preds,
                                target_names=['Normal', 'Attack'],
                                output_dict=True)
    cm = confusion_matrix(y_te, preds).tolist()
    print(f'\n[{name}] AUC={auc:.4f} F1={rep["Attack"]["f1-score"]:.4f} '
          f'P={rep["Attack"]["precision"]:.4f} R={rep["Attack"]["recall"]:.4f} '
          f'thr={thr:.4f} t={elapsed:.1f}s')
    model.save_weights(save_path)
    return {
        'model': name, 'auc': float(auc), 'ap': float(ap),
        'f1': rep['Attack']['f1-score'],
        'precision': rep['Attack']['precision'],
        'recall': rep['Attack']['recall'],
        'cm': cm, 'threshold': thr,
        'epochs_run': len(hist.history['loss']),
        'train_seconds': elapsed,
    }


def main():
    tf.random.set_seed(42); np.random.seed(42)
    splits = load_combined()
    INPUT_DIM = splits['train'][0].shape[1]
    print(f'\nInput dim: {INPUT_DIM}\n')

    results = []
    print('=== LSTM ===')
    results.append(fit_and_test(build_lstm(INPUT_DIM), splits, 'lstm_v2',
                                os.path.join(OUT, 'lstm_v2.weights.h5')))
    print('\n=== CNN-1D ===')
    results.append(fit_and_test(build_cnn1d(INPUT_DIM), splits, 'cnn1d_v2',
                                os.path.join(OUT, 'cnn1d_v2.weights.h5')))

    with open(os.path.join(OUT, 'train_supervised_report.json'), 'w') as f:
        json.dump({'models': results, 'input_dim': INPUT_DIM}, f, indent=2)
    print('\nSaved lstm_v2.weights.h5 + cnn1d_v2.weights.h5 + train_supervised_report.json')


if __name__ == '__main__':
    main()
