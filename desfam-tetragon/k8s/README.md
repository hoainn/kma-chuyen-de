# DeSFAM Live K8s Deployment

End-to-end flow:

1. **Train the model** — `cd ../train && bash run.sh up` then run training in container.
   Outputs land in `../outputs/`.

2. **Build & push detector image**:
   ```
   cd ../inference
   IMG=<your-registry>/desfam-detector ./build.sh v1.0 push
   ```
   `build.sh` bakes the trained artifacts into the image, so each model
   revision = a new image tag.

3. **Edit `desfam-detector.yaml`** — replace:
   - `<REGISTRY>/desfam-detector:latest` with your pushed image.
   - `TARGET_NAMESPACE` with the K8s namespace you want to monitor.
   - `--kernel-ver=5.15` with your node kernel `major.minor` (run
     `kubectl get nodes -o wide` to check).

4. **Apply** to your cluster:
   ```
   kubectl apply -f tracing-policy-syscalls.yaml
   kubectl apply -f desfam-detector.yaml
   ```

5. **Verify**:
   ```
   kubectl logs -n desfam -f deploy/desfam-detector
   ```
   Expected first lines:
   ```
   [INFO] Loading syscall table... 449 syscalls in table
   [INFO] Loading model params... alpha=0.7 threshold=0.0769 latent=24 input=174
   [INFO] Connecting to Tetragon gRPC at tetragon.kube-system.svc.cluster.local:54321...
   [INFO] Streaming events (namespace_filter=TARGET_NAMESPACE)...
   ```

6. **Smoke test**:
   - **Benign** — `kubectl run busybox --image=busybox -n TARGET_NAMESPACE -- sh -c "ls / && sleep 60"` → expect `normal` verdicts.
   - **Malicious** — `kubectl run privpod --image=alpine -n TARGET_NAMESPACE --privileged --restart=Never -- sh -c "unshare -U -r id && cat /etc/shadow"` → expect `ATTACK` verdicts on at least one window.

## Tetragon gRPC service discovery

The Tetragon Helm chart exposes the gRPC API as a Service. Confirm the DNS
name with:

```
kubectl get svc -n kube-system | grep tetragon
```

Typical name: `tetragon` or `tetragon-grpc`. If different, update
`--tetragon-addr` in `desfam-detector.yaml`. The default port is 54321.

## Tuning

- **Window/step** — paper uses window=15 stride=3 on highly-curated DongTing
  sequences. In production with mixed syscalls, larger windows (200/50)
  yield more stable features. Tune via the `--window` / `--step` args.
- **Threshold** — overridden via `--threshold 0.10` if you want to trade
  recall for precision. Default comes from `model_params.json`.
- **Filtering** — add more allow_list entries in `detect_tetra_grpc.py` if
  you want to monitor multiple namespaces or filter by pod label.

## Removing

```
kubectl delete -f desfam-detector.yaml
kubectl delete -f tracing-policy-syscalls.yaml
```
