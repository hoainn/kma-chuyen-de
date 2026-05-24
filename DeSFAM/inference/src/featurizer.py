"""
Feature engineering for syscall windows with configurable n-gram order.

Default layout (ngram_n=2, paper baseline): [freq_60 | disc_40 | stats_8 | bigram_40 | ver_26]
With ngram_n=3: [freq_60 | disc_40 | stats_8 | trigram_40 | ver_26]
Mirrors feature_engineering.py::transform() exactly (corrected stats_8).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy as scipy_entropy


class Featurizer:
    def __init__(
        self,
        top_ids: list[int],
        disc_ids: list[int],
        top_ngrams: list[tuple],
        ver_cols: list[str],
        input_dim: int,
        ngram_n: int = 2,
        feature_scaler=None,
        log_path: str | None = None,
    ):
        self._top_ids    = top_ids
        self._disc_ids   = disc_ids
        self._top_ngrams = top_ngrams
        self._ver_cols   = ver_cols
        self._input_dim  = input_dim
        self._ngram_n    = ngram_n
        self._feature_scaler = feature_scaler
        self._log_path   = log_path

        # Pre-build lookup dicts for speed
        self._top_idx  = {sc: i for i, sc in enumerate(top_ids)}
        self._disc_idx = {sc: i for i, sc in enumerate(disc_ids)}
        self._ng_idx   = {g: i for i, g in enumerate(top_ngrams)}

        self._n_freq = len(top_ids)
        self._n_disc = len(disc_ids)
        self._n_ng   = len(top_ngrams)

    def transform(self, seq_ids: list[int], kernel_ver: str) -> np.ndarray:
        """Build 174-dim feature vector from a syscall sequence."""
        if not seq_ids:
            return np.zeros(self._input_dim, dtype=np.float32)

        total   = len(seq_ids)
        n_freq  = self._n_freq
        n_disc  = self._n_disc

        X = np.zeros(self._input_dim, dtype=np.float32)

        # freq_60: normalised frequencies of top-N syscalls
        for sc in seq_ids:
            if sc in self._top_idx:
                X[self._top_idx[sc]] += 1
        X[:n_freq] /= max(total, 1)

        # disc_40: binary presence flags
        present = set(seq_ids)
        for sc in present:
            if sc in self._disc_idx:
                X[n_freq + self._disc_idx[sc]] = 1.0

        # stats_8: matches feature_engineering.py lines 128-136 exactly
        off = n_freq + n_disc
        raw_counts = X[:n_freq] * total              # top-N raw counts
        # [0] Shannon entropy of top-N freq distribution (base-2)
        p = raw_counts / (raw_counts.sum() + 1e-9)
        X[off + 0] = float(scipy_entropy(p[p > 0], base=2)) if p.any() else 0.0
        # [1] unique syscall count (raw integer)
        X[off + 1] = float(len(present))
        # [2] log(1 + total)
        X[off + 2] = float(np.log1p(total))
        # [3] dominant syscall fraction
        X[off + 3] = float(raw_counts.max()) / max(total, 1)
        # [4] 75th percentile of nonzero top-N counts
        nonzero = raw_counts[raw_counts > 0]
        X[off + 4] = float(np.percentile(nonzero, 75)) if len(nonzero) > 0 else 0.0
        # [5] std of top-N raw counts
        X[off + 5] = float(np.std(raw_counts))
        # [6] fraction of vocab syscalls present in this window
        X[off + 6] = float((raw_counts > 0).sum()) / max(n_freq, 1)
        # [7] sequence length / 1000
        X[off + 7] = float(total) / 1000.0

        # n-gram transition counts (normalised), order = self._ngram_n
        ng_off = off + 8
        n_ng = self._n_ng
        ngram_n = self._ngram_n
        if n_ng > 0 and total >= ngram_n:
            n_windows = max(total - ngram_n + 1, 1)
            for j in range(total - ngram_n + 1):
                gram = tuple(seq_ids[j : j + ngram_n])
                if gram in self._ng_idx:
                    X[ng_off + self._ng_idx[gram]] += 1
            X[ng_off : ng_off + n_ng] /= n_windows

        # ver_onehot: kernel version one-hot
        ver_short = ".".join(kernel_ver.split(".")[:2])
        ver_off = ng_off + n_ng
        for i, v in enumerate(self._ver_cols):
            if v == ver_short:
                X[ver_off + i] = 1.0
                break

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
