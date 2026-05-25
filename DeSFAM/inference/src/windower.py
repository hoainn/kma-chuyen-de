"""
Per-PID sliding-window emitter.

Matches paper §IV.B.1 SyscallAD spec: each window has fixed length L (default 15)
and a new window is emitted every stride S syscalls (default 3). The (pod_key, pid)
buffer is a deque of size L — once full, every S new pushes emit the current window
and slide forward.

flush_idle() does NOT emit windows — it only evicts dead PIDs from memory. Partial
windows that never reach length L are dropped (the model was trained on length-L
windows only; scoring an under-filled buffer would be out-of-distribution).
"""

from __future__ import annotations

import time
from collections import deque


_EXIT_GROUP = 231  # sys_exit_group syscall number


class Windower:
    def __init__(
        self,
        window_len: int = 15,
        window_stride: int = 3,
        max_idle_secs: float = 30.0,
    ):
        if window_len < 2 or window_stride < 1 or window_stride > window_len:
            raise ValueError(
                f"invalid window config: len={window_len} stride={window_stride}"
            )
        self._L = window_len
        self._S = window_stride
        self._max_idle = max_idle_secs

        # (pod_key, pid) → rolling buffer (size up to L)
        self._buf:  dict[tuple[str, int], deque[int]]  = {}
        self._args: dict[tuple[str, int], deque[dict]] = {}
        # syscalls pushed since last emit (used to enforce stride)
        self._since_emit: dict[tuple[str, int], int]   = {}
        # (pod_key, pid) → last-seen wall time, for idle eviction
        self._last_seen: dict[tuple[str, int], float]  = {}

    def push(
        self, pod_key: str, pid: int, nr: int, ts: float, args: dict | None = None
    ) -> tuple[str, list[int], list[dict]] | None:
        """
        Append syscall nr to the (pod_key, pid) rolling buffer.

        Emits (pod_key, window_seq, window_args) when the buffer is full AND
        S new syscalls have arrived since the previous emit. Otherwise returns None.
        """
        key = (pod_key, pid)
        buf  = self._buf.setdefault(key,  deque(maxlen=self._L))
        ab   = self._args.setdefault(key, deque(maxlen=self._L))
        buf.append(nr)
        ab.append(args or {})
        self._last_seen[key] = ts
        self._since_emit[key] = self._since_emit.get(key, 0) + 1

        # Process exited — drop its buffer (don't score partial windows).
        if nr == _EXIT_GROUP:
            self._buf.pop(key, None)
            self._args.pop(key, None)
            self._since_emit.pop(key, None)
            self._last_seen.pop(key, None)
            return None

        # Emit when the buffer is full and we've advanced ≥ stride syscalls.
        if len(buf) == self._L and self._since_emit[key] >= self._S:
            self._since_emit[key] = 0
            return pod_key, list(buf), list(ab)

        return None

    def flush_idle(self, now: float | None = None) -> list[tuple[str, list[int], list[dict]]]:
        """
        Evict PIDs that haven't produced a syscall in max_idle_secs.

        Returns an empty list — partial buffers are never scored. The signature
        is kept tuple-of-three for backward compatibility with detect.py.
        """
        if now is None:
            now = time.time()
        dead = [k for k, t in self._last_seen.items() if now - t > self._max_idle]
        for key in dead:
            self._buf.pop(key, None)
            self._args.pop(key, None)
            self._since_emit.pop(key, None)
            self._last_seen.pop(key, None)
        return []

    @property
    def active_pids(self) -> int:
        return len(self._buf)
