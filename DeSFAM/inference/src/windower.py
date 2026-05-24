"""
Per-PID trace accumulator.

Matches training data format: one sequence = one process execution trace.
A sequence is emitted when the process calls exit_group, or when it exceeds
max_len (safety valve for long-running processes), or when flush_idle() evicts
processes that have been silent longer than max_idle_secs.
"""

from __future__ import annotations

import time


_EXIT_GROUP = 231  # sys_exit_group syscall number


class Windower:
    def __init__(
        self,
        min_len: int = 8,
        max_len: int = 5000,
        max_idle_secs: float = 30.0,
    ):
        self._min_len = min_len
        self._max_len = max_len
        self._max_idle = max_idle_secs

        # (pod_key, pid) → accumulated syscall list
        self._seqs: dict[tuple[str, int], list[int]] = {}
        # (pod_key, pid) → syscall argument dicts, parallel to _seqs
        self._args: dict[tuple[str, int], list[dict]] = {}
        # (pod_key, pid) → last-seen wall time
        self._last_seen: dict[tuple[str, int], float] = {}

    def push(
        self, pod_key: str, pid: int, nr: int, ts: float, args: dict | None = None
    ) -> tuple[str, list[int], list[dict]] | None:
        """
        Append syscall nr (and optional args) to the (pod_key, pid) trace.

        Returns (pod_key, syscall_seq, args_seq) when the sequence is ready to score.
        Returns None otherwise.
        """
        key = (pod_key, pid)
        if key not in self._seqs:
            self._seqs[key] = []
            self._args[key] = []
        self._seqs[key].append(nr)
        self._args[key].append(args or {})
        self._last_seen[key] = ts

        # Process exited — emit and clean up
        if nr == _EXIT_GROUP:
            seq  = self._seqs.pop(key)
            aseq = self._args.pop(key, [])
            self._last_seen.pop(key, None)
            if len(seq) >= self._min_len:
                return pod_key, seq, aseq
            return None

        # Safety valve: long-running process — emit chunk and reset
        if len(self._seqs[key]) >= self._max_len:
            seq  = self._seqs[key][:]
            aseq = self._args[key][:]
            self._seqs[key] = []
            self._args[key] = []
            return pod_key, seq, aseq

        return None

    def flush_idle(self, now: float | None = None) -> list[tuple[str, list[int]]]:
        """
        Emit traces for processes that haven't produced a syscall in max_idle_secs.
        Call this periodically from the main loop (e.g. every 5 seconds).
        """
        if now is None:
            now = time.time()
        results = []
        dead = [k for k, t in self._last_seen.items() if now - t > self._max_idle]
        for key in dead:
            seq  = self._seqs.pop(key, [])
            aseq = self._args.pop(key, [])
            self._last_seen.pop(key, None)
            if len(seq) >= self._min_len:
                results.append((key[0], seq, aseq))
        return results

    @property
    def active_pids(self) -> int:
        return len(self._seqs)
