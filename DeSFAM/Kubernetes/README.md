# DeSFAM — Kubernetes

Kubernetes manifests, workloads, and Tetragon setup for the DeSFAM live detector.

```
Kubernetes/
├── tetragon/
│   ├── install.sh            # Helm install + TracingPolicy apply
│   ├── tracing-policy.yaml   # TracingPolicy — syscalls captured by eBPF
│   └── nodeport.yaml         # NodePort Service exposing gRPC on :30321
├── normal/
│   └── app.yaml              # Benign workload (normal-app pod)
├── attack/
│   ├── 01-shell-spawn.yaml   # TA0002 Execution
│   ├── 02-credential-access.yaml  # TA0006 Credential Access
│   ├── 03-discovery.yaml     # TA0007 Discovery
│   ├── 04-cryptominer.yaml   # TA0040 Impact
│   ├── 05-exfiltration.yaml  # TA0009/TA0010 Collection + Exfiltration
│   ├── 06-rootkit.yaml       # TA0005 Defense Evasion
│   └── 07-lateral-movement.yaml  # TA0008 Lateral Movement
├── deploy.sh                 # kubectl apply all workloads
└── teardown.sh               # kubectl delete all workloads
```

---

## 1. Install Tetragon

**Prerequisites:** Helm 3, kubectl, kernel >= 5.10

```bash
bash tetragon/install.sh
```

This:
1. Adds the Cilium Helm repo and installs the `tetragon` DaemonSet into `kube-system`
2. Waits for the DaemonSet to be ready
3. Applies `tracing-policy.yaml` — the eBPF hooks that capture syscalls DeSFAM needs
4. Prints the next step (expose gRPC)

**Expose gRPC as NodePort (port 30321):**

```bash
kubectl apply -f tetragon/nodeport.yaml
```

Verify:

```bash
kubectl get tracingpolicy
kubectl get svc tetragon-grpc-nodeport -n kube-system
```

The detector's default `TETRAGON_ADDR` is `<node-ip>:30321`. Set it via env or `--tetragon-addr`.

---

## 2. Deploy Workloads

Pods run in the `demo` namespace. Create it first if needed:

```bash
kubectl create namespace demo --dry-run=client -o yaml | kubectl apply -f -
```

Deploy all workloads:

```bash
bash deploy.sh
```

Check status:

```bash
kubectl get pods -n demo -o wide
```

Tear down:

```bash
bash teardown.sh
```

---

## 3. Workload Reference

### Normal

| Pod | Description |
|-----|-------------|
| `normal-app` | Periodic file I/O, HTTP health checks, in-process sorting — benign baseline |

### Attack (MITRE ATT&CK for Containers)

| Pod | Tactic | Key syscalls |
|-----|--------|--------------|
| `attack-shell-spawn` | TA0002 Execution | `execve`, `clone`, `wait4` |
| `attack-credential-access` | TA0006 Credential Access | `openat(/etc/shadow)`, `read` |
| `attack-discovery` | TA0007 Discovery | `getdents64(/proc/*)`, `connect` |
| `attack-cryptominer` | TA0040 Impact | `clone` × N, tight loops, `connect(:3333)` |
| `attack-exfiltration` | TA0009/TA0010 Collection + Exfil | `read(files)` → `write(socket_fd)` |
| `attack-rootkit` | TA0005 Defense Evasion | `openat(/proc/kallsyms)`, `unshare`, `init_module` |
| `attack-lateral-movement` | TA0008 Lateral Movement | `socket`→`connect` × 100 hosts |

---

## 4. TracingPolicy

`tetragon/tracing-policy.yaml` hooks the following syscall groups via kprobes:

| Group | Syscalls |
|-------|----------|
| File access | `openat`, `read`, `write`, `close`, `stat`, `lstat`, `getdents64` |
| Process | `execve`, `clone`, `wait4`, `exit_group` |
| Memory | `mmap`, `mprotect` |
| Network | `socket`, `connect`, `bind`, `accept`, `sendto`, `recvfrom` |
| Kernel/privilege | `unshare`, `init_module`, `ptrace` |

These cover the 149-dim feature vector DeSFAM's featurizer expects (freq\_60 | disc\_40 | stats\_8 | bigrams\_40 | ver\_1).

---

## 5. Full Pipeline

```
K8s pods  →  Tetragon eBPF (tracing-policy.yaml)
          →  gRPC stream (:30321)
          →  DeSFAM detector (DeSFAM/inference/)
          →  Prometheus + Grafana
```

Start the detector (from `DeSFAM/inference/`):

```bash
docker compose up
```
