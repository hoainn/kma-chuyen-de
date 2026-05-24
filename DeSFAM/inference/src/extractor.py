"""
Tetragon gRPC extractor — connects to the event stream and yields (pod_key, pid, syscall_nr, timestamp).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Iterator

import grpc

# tetragon stubs use bare `from tetragon import ...` — add tetragon_grpc/ to path
_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tetragon_grpc"))
sys.path.insert(0, _REPO_ROOT)
from tetragon import events_pb2, sensors_pb2_grpc
from syscalls_x86_64 import name_to_nr, NR_SYSCALL


def _pid(proc) -> int:
    """pid is UInt32Value in Tetragon proto — extract the wrapped integer."""
    raw = getattr(proc, "pid", None)
    if raw is None:
        return 0
    return int(raw.value) if hasattr(raw, "value") else int(raw)


_ARG_SYSCALLS = {"openat", "open", "execve", "stat", "lstat"}


def _extract_args(fn_name: str, kp_args) -> dict[str, str]:
    """Return the key string argument for syscalls where it matters for human triage."""
    stripped = fn_name
    for prefix in ("__x64_sys_", "__ia32_sys_", "__se_sys_", "sys_"):
        if fn_name.startswith(prefix):
            stripped = fn_name[len(prefix):]
            break
    if stripped not in _ARG_SYSCALLS:
        return {}
    for arg in kp_args:
        try:
            if arg.HasField("string_arg"):
                val = arg.string_arg.rstrip("\x00").strip()
                if val:
                    key = "exec" if stripped == "execve" else "path"
                    return {key: val}
        except Exception:
            pass
    return {}


def _pod_key(proc) -> str | None:
    """Return 'namespace/pod-name', or None for non-Kubernetes (host) processes."""
    pod = getattr(proc, "pod", None)
    if pod and getattr(pod, "namespace", ""):
        return f"{pod.namespace}/{pod.name or _pid(proc)}"
    return None


def _parse_event(response) -> tuple[str | None, int | None, int | None, dict]:
    """Return (pod_key, pid, syscall_nr, args) or (None, None, None, {}) if not usable."""
    if response.HasField("process_kprobe"):
        kp = response.process_kprobe
        nr = name_to_nr(kp.function_name or "")
        if nr is None:
            return None, None, None, {}
        args = _extract_args(kp.function_name, kp.args)
        return _pod_key(kp.process), _pid(kp.process), nr, args

    if response.HasField("process_tracepoint"):
        tp = response.process_tracepoint
        if tp.event != "sys_enter":
            return None, None, None, {}
        tp_args = list(tp.args)
        if not tp_args:
            return None, None, None, {}
        arg0 = tp_args[0]
        nr = None
        if arg0.HasField("long_arg"):
            nr = int(arg0.long_arg)
        elif arg0.HasField("int_arg"):
            nr = int(arg0.int_arg)
        if nr is None or nr < 0 or nr not in NR_SYSCALL:
            return None, None, None, {}
        return _pod_key(tp.process), _pid(tp.process), nr, {}

    if response.HasField("process_exec"):
        proc = response.process_exec.process
        binary = getattr(proc, "binary", "") or ""
        args = {"exec": binary} if binary else {}
        return _pod_key(proc), _pid(proc), 59, args  # execve

    if response.HasField("process_exit"):
        proc = response.process_exit.process
        return _pod_key(proc), _pid(proc), 231, {}  # exit_group

    return None, None, None, {}


class Extractor:
    def __init__(self, addr: str, namespace: str, log: logging.Logger | None = None):
        self._addr = addr
        self._namespace = namespace
        self._log = log or logging.getLogger(__name__)

    def stream(self) -> Iterator[tuple[str, int, int, float]]:
        """Yields (pod_key, pid, syscall_nr, wall_time) for every valid syscall event."""
        self._log.info(f"Connecting to Tetragon at {self._addr} ...")
        channel = grpc.insecure_channel(
            self._addr,
            options=[
                ("grpc.keepalive_time_ms", 120_000),
                ("grpc.keepalive_timeout_ms", 20_000),
                ("grpc.max_receive_message_length", 16 * 1024 * 1024),
            ],
        )
        stub = sensors_pb2_grpc.FineGuidanceSensorsStub(channel)
        req = events_pb2.GetEventsRequest()
        # Namespace scoping is handled at eBPF level by TracingPolicyNamespaced.
        # The gRPC allow_list.namespace filter in Tetragon v1.7 silently drops
        # matching events, so we rely on the policy instead.
        self._log.info(f"Streaming (ns_filter={self._namespace or '<all>'}) ...")

        ns_prefix = self._namespace + "/" if self._namespace else None
        for response in stub.GetEvents(req):
            pod_key, pid, nr, args = _parse_event(response)
            if pod_key is None:
                continue
            if ns_prefix and not pod_key.startswith(ns_prefix):
                continue
            yield pod_key, pid, nr, time.time(), args
