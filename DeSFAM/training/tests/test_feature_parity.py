"""
Train ↔ inference feature-vector parity (the single most important guard).

Asserts that train.py::build_features (per-window canonical form), train.py's
vectorised build_features_windowed, and inference's Featurizer.transform all
produce byte-identical paper-only vectors for the same windows.

Runs without a real TensorFlow install: train.py imports tf at module load, but
the feature functions never touch it, so we stub tf/keras in sys.modules when the
real packages are missing. Run:  python tests/test_feature_parity.py
"""
import importlib.util
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.dirname(HERE)
INFER_SRC = os.path.normpath(os.path.join(TRAIN_DIR, "..", "inference", "src"))


def _stub_tensorflow():
    """Provide dummy tensorflow/keras modules so `import train` succeeds where TF
    is not installed. The feature code under test does not use them."""
    for name in ("tensorflow", "tensorflow.keras", "tensorflow.keras.layers"):
        if name in sys.modules:
            return
    tf = types.ModuleType("tensorflow")
    tf.__version__ = "stub"
    keras = types.ModuleType("tensorflow.keras")
    layers = types.ModuleType("tensorflow.keras.layers")
    keras.layers = layers
    tf.keras = keras
    sys.modules.setdefault("tensorflow", tf)
    sys.modules.setdefault("tensorflow.keras", keras)
    sys.modules.setdefault("tensorflow.keras.layers", layers)


def _import_train():
    try:
        import tensorflow  # noqa: F401
    except Exception:
        _stub_tensorflow()
    sys.path.insert(0, TRAIN_DIR)
    import train  # noqa: E402
    return train


def _import_featurizer():
    # Import inference featurizer in isolation (numpy-only module).
    spec = importlib.util.spec_from_file_location(
        "infer_featurizer", os.path.join(INFER_SRC, "featurizer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    train = _import_train()
    feat = _import_featurizer()

    rng = np.random.default_rng(0)
    cat_cols = list(train.SYSCALL_CATEGORIES)
    # A small id→category map spanning several categories + an unmapped id (→other).
    id_to_cat = {0: "process", 1: "file", 2: "file", 3: "memory",
                 4: "network", 5: "signal", 9: "time", 12: "ipc",
                 33: "security", 7: "io_event"}
    # Benign patterns (subsequences) + first-id index, exactly as the loaders build them.
    patterns = [(0, 1, 2), (1, 2, 3), (4, 5), (0, 3, 9, 12)]
    pbf_train = train.build_patterns_by_first(patterns)

    from collections import defaultdict
    pbf_infer = defaultdict(list)
    for p in patterns:
        pbf_infer[int(p[0])].append(tuple(int(x) for x in p))
    pbf_infer = {k: sorted(v) for k, v in pbf_infer.items()}

    L = 15
    ps_dims = train.PREFIXSPAN_DIMS
    n_cat = len(cat_cols)
    input_dim = n_cat + ps_dims  # temporal off (DongTing)

    # 200 random windows over a small id vocabulary that exercises matches + misses.
    windows = [[int(x) for x in rng.integers(0, 40, size=L)] for _ in range(200)]
    # force some windows to contain exact patterns
    windows[0][:3] = [0, 1, 2]
    windows[1][:4] = [0, 3, 9, 12]
    windows[2][0:2] = [4, 5]

    X_train = train.build_features(windows, cat_cols, id_to_cat,
                                   pbf_train, ps_dims, ts_list=None, temporal_dims=0)

    fz = feat.Featurizer(cat_cols=cat_cols, id_to_cat=id_to_cat,
                         patterns_by_first=pbf_infer, ps_dims=ps_dims,
                         input_dim=input_dim, has_temporal=False, temporal_dims=0,
                         feature_scaler=None)
    X_infer = np.vstack([fz.transform(w)[0] for w in windows])

    assert X_train.shape == X_infer.shape == (200, input_dim), \
        f"shape mismatch: {X_train.shape} vs {X_infer.shape} vs (200,{input_dim})"
    max_abs = float(np.max(np.abs(X_train - X_infer)))
    assert max_abs == 0.0, f"train vs inference differ (max abs diff={max_abs})"
    print(f"[OK] minimal-vector parity: train == inference, dim={input_dim}, max_abs_diff=0")

    # ── Full engineered-vector parity (the configuration the detector actually
    # ships with: freq+disc+stats+bigrams+cat+prefixspan) ────────────────────
    top_ids     = [0, 1, 2, 3, 4, 5, 9, 12]
    disc_ids    = [33, 7]
    top_bigrams = [(0, 1), (1, 2), (4, 5), (9, 9)]
    input_dim_p = len(top_ids) + len(disc_ids) + 8 + len(top_bigrams) + n_cat + ps_dims

    X_tr2 = train.build_features(
        windows, cat_cols, id_to_cat, pbf_train, ps_dims,
        ts_list=None, temporal_dims=0,
        top_ids=top_ids, disc_ids=disc_ids, top_bigrams=top_bigrams, enable_stats=True)

    fz2 = feat.Featurizer(cat_cols=cat_cols, id_to_cat=id_to_cat,
                          patterns_by_first=pbf_infer, ps_dims=ps_dims,
                          input_dim=input_dim_p, has_temporal=False, temporal_dims=0,
                          top_ids=top_ids, disc_ids=disc_ids, top_bigrams=top_bigrams,
                          enable_stats=True, feature_scaler=None)
    X_inf2 = np.vstack([fz2.transform(w)[0] for w in windows])
    assert X_tr2.shape == X_inf2.shape == (200, input_dim_p), \
        f"full-vector shape: {X_tr2.shape} vs {X_inf2.shape}"
    max_p = float(np.max(np.abs(X_tr2 - X_inf2)))
    assert max_p < 1e-6, f"full-vector train vs inference differ (max abs diff={max_p})"
    print(f"[OK] full-vector parity: train == inference, dim={input_dim_p}, max_abs_diff={max_p:.2e}")

    # Windowed (vectorised) path parity: one trace, windows must equal per-window form.
    trace = []
    for w in windows[:20]:
        trace.extend(w)
    Xw, yw, dropped = train.build_features_windowed(
        [trace], np.array([0], dtype=np.int32), L, 3, cat_cols, id_to_cat,
        pbf_train, ps_dims, ts_list=None, temporal_dims=0)
    # rebuild the same windows by hand and compare
    n_win = train.count_windows(len(trace), L, 3)
    manual = [trace[i * 3: i * 3 + L] for i in range(n_win)]
    Xm = train.build_features(manual, cat_cols, id_to_cat, pbf_train, ps_dims,
                              ts_list=None, temporal_dims=0)
    assert Xw.shape == Xm.shape, f"windowed shape {Xw.shape} vs manual {Xm.shape}"
    max_abs_w = float(np.max(np.abs(Xw - Xm)))
    assert max_abs_w == 0.0, f"windowed vs per-window differ (max abs diff={max_abs_w})"
    print(f"[OK] windowed parity (minimal vector): build_features_windowed == build_features, "
          f"windows={n_win}, max_abs_diff=0")

    # Full engineered-vector windowed parity (vectorised stats/freq/disc/bigram
    # vs the per-window form).
    Xw2, _, _ = train.build_features_windowed(
        [trace], np.array([0], dtype=np.int32), L, 3, cat_cols, id_to_cat,
        pbf_train, ps_dims, ts_list=None, temporal_dims=0,
        top_ids=top_ids, disc_ids=disc_ids, top_bigrams=top_bigrams, enable_stats=True)
    Xm2 = train.build_features(
        manual, cat_cols, id_to_cat, pbf_train, ps_dims,
        ts_list=None, temporal_dims=0,
        top_ids=top_ids, disc_ids=disc_ids, top_bigrams=top_bigrams, enable_stats=True)
    assert Xw2.shape == Xm2.shape, f"full-vector windowed shape {Xw2.shape} vs {Xm2.shape}"
    max_w2 = float(np.max(np.abs(Xw2 - Xm2)))
    # Allow tiny float drift for the entropy/std vectorised math.
    assert max_w2 < 1e-5, f"full-vector windowed vs per-window differ (max abs diff={max_w2})"
    print(f"[OK] windowed parity (full vector): vectorised == per-window, "
          f"windows={n_win}, dim={Xw2.shape[1]}, max_abs_diff={max_w2:.2e}")

    print("ALL PARITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
