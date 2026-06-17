"""
Feature engineering for syscall windows — mirrors train.py::build_features
bit-for-bit. Supports two feature modes:

  paper_only — vector exactly as named in paper §IV.B.3:
      [ cat_K | temporal_T? | prefixspan_P ]
  paper_plus — prepends engineered support groups the paper underspecifies:
      [ freq_F | disc_D | stats_8 | bigrams_B | cat_K | temporal_T? | prefixspan_P ]

`transform()` is the canonical per-window form; guarded by `tests/test_feature_parity.py`.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy as scipy_entropy


def match_benign_patterns(window: list[int],
                          patterns_by_first: dict[int, list]) -> tuple[float, int]:
    """PrefixSpan match — identical copy of train.py::match_benign_patterns.

    A benign pattern matches if it is an order-preserving subsequence of the window.
    Returns (matched_flag, longest_match_len). Order-independent.
    """
    matched = 0.0
    longest = 0
    n = len(window)
    for i in range(n):
        pats = patterns_by_first.get(window[i])
        if not pats:
            continue
        for pat in pats:
            L = len(pat)
            if matched and L <= longest:
                continue
            pi, wi = 0, i
            while wi < n and pi < L:
                if window[wi] == pat[pi]:
                    pi += 1
                wi += 1
            if pi == L:
                matched = 1.0
                if L > longest:
                    longest = L
    return matched, longest


class Featurizer:
    def __init__(
        self,
        cat_cols: list[str],
        id_to_cat: dict[int, str],
        patterns_by_first: dict[int, list],
        ps_dims: int,
        input_dim: int,
        has_temporal: bool = False,
        temporal_dims: int = 0,
        top_ids: list[int] | tuple = (),
        disc_ids: list[int] | tuple = (),
        top_bigrams: list[tuple] | tuple = (),
        enable_stats: bool = False,
        feature_scaler=None,
        log_path: str | None = None,
        sensor_alphabet: set[int] | None = None,
    ):
        # Live windows are dropped to the training syscall alphabet before any
        # feature math, mirroring train.py's loader-side filter. No-op in normal
        # operation (the Tetragon policy already emits only these); guards cat_K
        # against a policy that traces more syscalls than training did.
        self._alphabet   = set(int(x) for x in sensor_alphabet) if sensor_alphabet else None
        self._cat_cols   = cat_cols or []
        self._id_to_cat  = id_to_cat or {}
        self._patterns_by_first = patterns_by_first or {}
        self._ps_dims    = ps_dims
        self._input_dim  = input_dim
        self._has_temporal  = bool(has_temporal)
        self._temporal_dims = temporal_dims
        self._top_ids     = [int(x) for x in (top_ids or [])]
        self._disc_ids    = [int(x) for x in (disc_ids or [])]
        self._top_bigrams = [tuple(g) for g in (top_bigrams or [])]
        self._enable_stats = bool(enable_stats)
        self._feature_scaler = feature_scaler
        self._log_path   = log_path

        self._cat_idx  = {c: i for i, c in enumerate(self._cat_cols)}
        self._top_idx  = {sc: i for i, sc in enumerate(self._top_ids)}
        self._disc_idx = {sc: i for i, sc in enumerate(self._disc_ids)}
        self._ng_idx   = {tuple(g): i for i, g in enumerate(self._top_bigrams)}

        self._n_cat   = len(self._cat_cols)
        self._n_freq  = len(self._top_ids)
        self._n_disc  = len(self._disc_ids)
        self._n_ng    = len(self._top_bigrams)
        self._n_stats = 8 if self._enable_stats else 0

        # Layout offsets (must match train.py::build_features exactly).
        self._freq_off  = 0
        self._disc_off  = self._n_freq
        self._stats_off = self._disc_off + self._n_disc
        self._bg_off    = self._stats_off + self._n_stats
        self._cat_off   = self._bg_off + self._n_ng
        self._temp_off  = self._cat_off + self._n_cat
        self._ps_off    = self._temp_off + self._temporal_dims

    def transform(self, seq_ids: list[int], kernel_ver: str | None = None,
                  timestamps: list[float] | None = None) -> np.ndarray:
        """Build the paper-only / paper-plus feature vector for one window.

        `kernel_ver` is accepted for call-site compatibility but unused (the
        version one-hot was dropped). `timestamps` activates the temporal group
        only when the model was trained with it (has_temporal)."""
        X = np.zeros(self._input_dim, dtype=np.float32)
        if not seq_ids:
            feat = X.reshape(1, -1)
            if self._feature_scaler is not None:
                feat = self._feature_scaler.transform(feat).astype(np.float32)
            return feat

        seq = [int(s) for s in seq_ids]
        if self._alphabet is not None:
            seq = [s for s in seq if s in self._alphabet]
            if not seq:
                feat = X.reshape(1, -1)
                if self._feature_scaler is not None:
                    feat = self._feature_scaler.transform(feat).astype(np.float32)
                return feat
        total = len(seq)

        # freq_F
        if self._n_freq:
            for sc in seq:
                j = self._top_idx.get(sc)
                if j is not None:
                    X[self._freq_off + j] += 1
            X[self._freq_off: self._freq_off + self._n_freq] /= max(total, 1)

        # disc_D (binary presence)
        if self._n_disc:
            for sc in set(seq):
                j = self._disc_idx.get(sc)
                if j is not None:
                    X[self._disc_off + j] = 1.0

        # stats_8 (derived from top-K raw counts)
        if self._n_stats:
            raw = X[self._freq_off: self._freq_off + self._n_freq] * total if self._n_freq \
                  else np.zeros(0, np.float32)
            total_raw = float(raw.sum()) if self._n_freq else 0.0
            if total_raw > 0:
                p = raw / total_raw
                X[self._stats_off + 0] = float(scipy_entropy(p[p > 0], base=2))
            else:
                X[self._stats_off + 0] = 0.0
            X[self._stats_off + 1] = float(len(set(seq)))
            X[self._stats_off + 2] = float(np.log1p(total))
            X[self._stats_off + 3] = (float(raw.max()) / max(total, 1)) if self._n_freq else 0.0
            nz = raw[raw > 0] if self._n_freq else np.zeros(0, np.float32)
            X[self._stats_off + 4] = float(np.percentile(nz, 75)) if nz.size else 0.0
            X[self._stats_off + 5] = float(np.std(raw)) if self._n_freq else 0.0
            X[self._stats_off + 6] = (float((raw > 0).sum()) / max(self._n_freq, 1)) if self._n_freq else 0.0
            X[self._stats_off + 7] = float(total) / 1000.0

        # bigrams_B
        if self._n_ng and total >= 2:
            n_win_ng = max(total - 1, 1)
            for k in range(total - 1):
                g = (seq[k], seq[k + 1])
                j = self._ng_idx.get(g)
                if j is not None:
                    X[self._bg_off + j] += 1
            X[self._bg_off: self._bg_off + self._n_ng] /= n_win_ng

        # cat_K
        if self._n_cat > 0 and self._id_to_cat:
            for sc in seq:
                ci = self._cat_idx.get(self._id_to_cat.get(sc, 'other'))
                if ci is not None:
                    X[self._cat_off + ci] += 1
            X[self._cat_off: self._cat_off + self._n_cat] /= max(total, 1)

        # temporal_T (gated by has_temporal to match trained dims)
        if self._has_temporal and self._temporal_dims and timestamps is not None \
                and len(timestamps) >= 2:
            dt = np.diff(np.asarray(timestamps, dtype=np.float64))
            X[self._temp_off + 0] = float(dt.mean())
            X[self._temp_off + 1] = float(dt.std())
            X[self._temp_off + 2] = float(dt.max())

        # prefixspan_P
        if self._ps_dims and self._patterns_by_first:
            matched, longest = match_benign_patterns(seq, self._patterns_by_first)
            X[self._ps_off + 0] = matched
            X[self._ps_off + 1] = longest / max(total, 1)

        feat = X[:self._input_dim].astype(np.float32).reshape(1, -1)
        if self._feature_scaler is not None:
            feat = self._feature_scaler.transform(feat).astype(np.float32)
        if self._log_path is not None:
            self._log_feature(feat)
        return feat

    def _log_feature(self, feat: np.ndarray) -> None:
        """Append feature row to Parquet file (requires pyarrow)."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            import os

            table = pa.table({f"f{i}": [float(v)] for i, v in enumerate(feat[0])})
            if os.path.exists(self._log_path):
                existing = pq.read_table(self._log_path)
                table = pa.concat_tables([existing, table])
            pq.write_table(table, self._log_path)
        except Exception:
            pass  # logging is best-effort
