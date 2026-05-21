# DeSFAM Reproduction + Tetragon Live K8s Detector

End-to-end reproduction of the DeSFAM paper's **SyscallAD** anomaly
detector (VAE + Isolation Forest ensemble) on the DongTing dataset,
plus a gRPC-based detector that consumes live syscall events from
**Cilium Tetragon** running in a Kubernetes cluster.

## Layout

```
desfam-tetragon/
├── train/                   # Phase A: reproduce training
│   ├── docker-compose.yml   # Mongo + JupyterLab + DongTing import
│   ├── explore_and_fe.py    # 174-dim feature engineering
│   ├── train_fe.py          # IF + VAE + ensemble (with RobustScaler fix)
│   └── run.sh               # bash run.sh up/down/logs
├── inference/               # Phase B: live detector
│   ├── detect_tetra_grpc.py # gRPC client → IF+VAE scoring
│   ├── tetragon_grpc/       # Generated Python protobuf stubs
│   ├── Dockerfile.detector  # Image that bakes in trained artifacts
│   └── build.sh             # Build/push helper
├── k8s/                     # Phase B: cluster manifests
│   ├── tracing-policy-syscalls.yaml  # Tetragon TracingPolicy
│   ├── desfam-detector.yaml          # Deployment + SA
│   └── README.md            # Deploy walkthrough
├── outputs/                 # Trained artifacts (created by Phase A)
└── REPRODUCTION_REPORT.md   # Phase A results vs paper claims
```

## Quick start

```bash
# Phase A — train
cd train && bash run.sh up
docker exec dongting_jupyter python /workspace/explore_and_fe.py
docker exec dongting_jupyter python /workspace/train_fe.py

# Phase B — deploy
cd ../inference && IMG=<registry>/desfam-detector ./build.sh v1.0 push
# edit ../k8s/desfam-detector.yaml: set image + TARGET_NAMESPACE + kernel-ver
kubectl apply -f ../k8s/tracing-policy-syscalls.yaml
kubectl apply -f ../k8s/desfam-detector.yaml
kubectl logs -n desfam -f deploy/desfam-detector
```

See `REPRODUCTION_REPORT.md` for the result table and the key fix vs.
the prior `experiment/` pipeline; see `k8s/README.md` for the deploy
walkthrough and smoke tests.

## Source paper

> **DeSFAM**: An Adaptive eBPF and AI-Driven Framework for Securing
> Cloud Containers in Real Time. *IEEE Access* (2025), DOI:
> [10.1109/ACCESS.2025.3592192](https://doi.org/10.1109/ACCESS.2025.3592192).

LaTeX source lives under `../latex/sections_en/`. The reproduction
table is inserted in `05_evaluation.tex` under the *Anomaly Detection
Performance* subsection.
