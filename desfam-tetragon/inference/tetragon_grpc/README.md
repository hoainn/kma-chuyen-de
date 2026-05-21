# Tetragon gRPC Stubs

Python stubs generated from Cilium Tetragon's `.proto` API definitions.

## Regenerate

1. Check your cluster's Tetragon version:
   ```
   kubectl exec -n kube-system $(kubectl get pod -n kube-system -l app.kubernetes.io/name=tetragon -o jsonpath='{.items[0].metadata.name}') -c tetragon -- tetragon version
   ```

2. Download the matching `.proto` files into `proto/tetragon/`:
   ```
   VER=v1.5.0   # ← replace with your version (e.g. v1.5.0, v1.4.1, v1.3.0)
   mkdir -p proto/tetragon
   for f in tetragon.proto events.proto sensors.proto stack.proto capabilities.proto; do
     curl -L https://raw.githubusercontent.com/cilium/tetragon/$VER/api/v1/tetragon/$f \
       -o proto/tetragon/$f
   done
   ```

3. Generate the Python stubs:
   ```
   pip install grpcio-tools
   make stubs
   ```

   This produces:
   - `tetragon/tetragon_pb2.py`, `tetragon/tetragon_pb2_grpc.py` — service stubs
   - `tetragon/events_pb2.py` — event message types (`ProcessExec`, `ProcessKprobe`, `ProcessTracepoint`, etc.)
   - `tetragon/sensors_pb2.py` — sensor management

## Usage

```python
import grpc
from tetragon import tetragon_pb2, tetragon_pb2_grpc

channel = grpc.insecure_channel("tetragon.kube-system.svc.cluster.local:54321")
stub = tetragon_pb2_grpc.FineGuidanceSensorsStub(channel)
req = tetragon_pb2.GetEventsRequest()    # add filters here

for response in stub.GetEvents(req):
    if response.HasField("process_kprobe"):
        kp = response.process_kprobe
        print(kp.function_name, kp.process.pod.namespace)
```

See `../detect_tetra_grpc.py` for the full integration.
