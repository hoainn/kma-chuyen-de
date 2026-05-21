"""
Tetragon Event Recorder — captures syscall sequences from a live cluster
via Tetragon's FineGuidanceSensors.GetEvents gRPC API and writes per-pod
.npy files for downstream feature engineering / model training.

Usage (from host with `kubectl port-forward svc/tetragon-grpc 54321:54321
-n kube-system` running):

    python record_tetragon.py \
        --tetragon-addr localhost:54321 \
        --namespace baseline \
        --label benign \
        --out collect/recordings \
        --duration 1800            # 30 minutes

Reuses parse_event_pb() and the syscall table loader from
inference/detect_tetra_grpc.py — same parsing path the live detector uses.
"""
import argparse
import collections
import json
import logging
import os
import sys
import time

import grpc
import numpy as np

# Generated Tetragon stubs (events_pb2 holds GetEventsRequest, sensors_pb2_grpc
# holds the FineGuidanceSensorsStub).
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'inference', 'tetragon_grpc'))
from tetragon import events_pb2, sensors_pb2_grpc  # noqa: E402


# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Tetragon syscall recorder')
parser.add_argument('--tetragon-addr', default='localhost:54321',
                    help='gRPC endpoint (host:port). Port-forward the '
                         'tetragon-grpc Service to localhost first.')
parser.add_argument('--namespace', required=True,
                    help='K8s namespace to record from')
parser.add_argument('--label', choices=['benign', 'attack', 'unknown'],
                    default='unknown',
                    help='Default label applied to every recorded pod')
parser.add_argument('--out', default=os.path.join(HERE, 'recordings'),
                    help='Directory for .npy + .json sidecar files')
parser.add_argument('--syscall-tbl', default=os.path.join(
    HERE, '..', '..', 'experiment', 'data', 'dongting', 'syscall_64.tbl'))
parser.add_argument('--duration', type=int, default=1800,
                    help='Recording duration in seconds (default 1800 = 30 min)')
parser.add_argument('--flush-every', type=int, default=50_000,
                    help='Flush a pod to disk after this many syscalls')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('record')

os.makedirs(args.out, exist_ok=True)


# ── Syscall table ─────────────────────────────────────────────────────────────
syscall_name_to_id = {}
syscall_id_to_name = {}
with open(args.syscall_tbl) as f:
    for line in f:
        parts = line.strip().split()
        if parts and parts[0].isdigit():
            sid, sname = int(parts[0]), parts[2]
            syscall_name_to_id[sname] = sid
            syscall_id_to_name[sid] = sname
log.info(f'{len(syscall_name_to_id)} syscalls in table')


# ── Tetragon event parser (mirrors detect_tetra_grpc.parse_event_pb) ──────────
SYSCALL_FN_PREFIXES = ('__x64_sys_', '__arm64_sys_', 'sys_', '__se_sys_')


def _strip_syscall_prefix(fn_name):
    for p in SYSCALL_FN_PREFIXES:
        if fn_name.startswith(p):
            return fn_name[len(p):]
    return fn_name


def _pod_key(process_pb, default_ns):
    pod = getattr(process_pb, 'pod', None)
    if pod and pod.namespace:
        ns = pod.namespace
        name = pod.name or str(getattr(process_pb.pid, 'value', process_pb.pid))
    else:
        ns = default_ns or 'host'
        name = str(getattr(process_pb.pid, 'value', process_pb.pid))
    return f'{ns}/{name}'


def parse_event_pb(response):
    if response.HasField('process_kprobe'):
        kp = response.process_kprobe
        fn = _strip_syscall_prefix(kp.function_name or '')
        sid = syscall_name_to_id.get(fn)
        if sid is None:
            return None, None
        return _pod_key(kp.process, ''), sid

    if response.HasField('process_tracepoint'):
        tp = response.process_tracepoint
        if tp.event != 'sys_enter' and tp.subsys != 'raw_syscalls':
            return None, None
        if not tp.args:
            return None, None
        arg0 = tp.args[0]
        sid = None
        if arg0.HasField('long_arg'):
            sid = int(arg0.long_arg)
        elif arg0.HasField('int_arg'):
            sid = int(arg0.int_arg)
        if sid is None or sid < 0 or sid not in syscall_id_to_name:
            return None, None
        return _pod_key(tp.process, ''), sid

    if response.HasField('process_exec'):
        pe = response.process_exec
        return _pod_key(pe.process, ''), syscall_name_to_id.get('execve', 59)

    if response.HasField('process_exit'):
        pe = response.process_exit
        return _pod_key(pe.process, ''), syscall_name_to_id.get('exit_group', 231)

    return None, None


# ── Recording state ───────────────────────────────────────────────────────────
buffers = collections.defaultdict(list)         # pod → [syscall_id, ...]
counts = collections.defaultdict(int)           # pod → total syscalls seen
first_seen = {}                                 # pod → unix-ts of first event
session_id = time.strftime('%Y%m%dT%H%M%S')


def flush_pod(pod_key, force=False):
    """Persist pod's buffer to disk if it has enough syscalls or force=True."""
    buf = buffers[pod_key]
    if not buf:
        return
    if not force and len(buf) < args.flush_every:
        return
    safe_pod = pod_key.replace('/', '_').replace(':', '_')
    npy_path = os.path.join(args.out, f'{safe_pod}_{session_id}.npy')
    arr = np.array(buf, dtype=np.int32)
    # If file already exists from an earlier flush, concatenate
    if os.path.exists(npy_path):
        old = np.load(npy_path)
        arr = np.concatenate([old, arr])
    np.save(npy_path, arr)
    # Sidecar JSON
    json_path = os.path.join(args.out, f'{safe_pod}_{session_id}.json')
    meta = {
        'pod_key': pod_key,
        'namespace': args.namespace,
        'label': args.label,
        'session_id': session_id,
        'first_seen_unix': first_seen.get(pod_key),
        'total_syscalls': int(arr.size),
        'kernel_ver_hint': '6.12',
        'recorded_at': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
    }
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    buffers[pod_key] = []
    log.info(f'  flushed {pod_key}: total={arr.size} syscalls -> {npy_path}')


# ── gRPC stream ───────────────────────────────────────────────────────────────
def main():
    channel = grpc.insecure_channel(
        args.tetragon_addr,
        options=[
            ('grpc.keepalive_time_ms', 120_000),
            ('grpc.keepalive_timeout_ms', 20_000),
            ('grpc.keepalive_permit_without_calls', 0),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.max_receive_message_length', 16 * 1024 * 1024),
        ],
    )
    stub = sensors_pb2_grpc.FineGuidanceSensorsStub(channel)
    req = events_pb2.GetEventsRequest()
    flt = req.allow_list.add()
    flt.namespace.append(args.namespace)

    log.info(f'Recording namespace={args.namespace!r} label={args.label!r} '
             f'duration={args.duration}s out={args.out}')
    log.info(f'Connecting to {args.tetragon_addr}...')

    deadline = time.time() + args.duration
    last_status = time.time()

    try:
        for response in stub.GetEvents(req, timeout=args.duration + 60):
            now = time.time()
            if now >= deadline:
                break

            pod_key, sid = parse_event_pb(response)
            if pod_key is None:
                continue

            if pod_key not in first_seen:
                first_seen[pod_key] = now
                log.info(f'  new pod: {pod_key}')

            buffers[pod_key].append(sid)
            counts[pod_key] += 1
            flush_pod(pod_key)

            if now - last_status > 30:
                summary = ', '.join(f'{k}={counts[k]}' for k in sorted(counts))
                log.info(f'  [t+{int(now-(deadline-args.duration))}s] {summary}')
                last_status = now
    except grpc.RpcError as e:
        log.error(f'gRPC error: {e.code()} — {e.details()}')

    log.info('Final flush...')
    for pk in list(buffers.keys()):
        flush_pod(pk, force=True)

    total = sum(counts.values())
    log.info(f'Done. {total} syscalls across {len(counts)} pods in '
             f'{time.strftime("%H:%M:%S")}')


if __name__ == '__main__':
    main()
