# 02 — Thiết kế Testbed (Môi trường thí nghiệm)

> Applied Experimentation đòi hỏi **controlled testbed**: kiểm soát chặt nhân tố, quan sát DV. Ở đây
> *cơ chế phát hiện* giữ cố định; biến độc lập = **cường độ nhiễu noisy-neighbor** (RQ2), nhân tố phụ =
> **loại tấn công** (loud/stealthy). Chạy song song **2 nhánh ngưỡng (EMA động vs `T_0` cố định)**.

## Sơ đồ hệ thống

```
        ┌──────────────────────── Cụm Kubernetes cô lập (lab, không route ra Internet) ────────────────────────┐
        │  Node (kernel ≥ 5.10)                                                                                  │
        │                                                                                                        │
        │   ┌───────────────────────── Tetragon DaemonSet (eBPF, kube-system) ─────────────────────────┐        │
        │   │  TracingPolicy `desfam-syscall-trace` → hook 23 syscall an ninh (sensor alphabet)         │        │
        │   └───────────────────────────────────────┬───────────────────────────────────────────────────┘        │
        │                                            │ gRPC export (NodePort :30321)                              │
        │   ns: demo                                 ▼                                                            │
        │   ┌───────────┐  ┌───────────────┐   ┌──────────────────────────────────────────────┐                 │
        │   │ normal-app │  │ noisy-neighbor│   │  DeSFAM detector (inference/detect.py)        │                 │
        │   │ (benign)   │  │  tenants × N  │   │  loader→extractor→windower(15/3)→featurizer    │                 │
        │   ├───────────┤  └───────────────┘   │  →scorer(VAE+iForest ensemble)→alerter (T_0)   │                 │
        │   │ attack pod │ ───── syscalls ─────▶│  → điểm bất thường per-window + nhãn ns/pod    │                 │
        │   │ (kịch bản) │                      └───────────────────┬──────────────────────────┘                 │
        │   └───────────┘                                          │ Prometheus metrics / CSV                    │
        │                                                          ▼                                             │
        │                                           Prometheus + Grafana + Loki (quan sát + log syscall)         │
        └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Cấu hình (cố định trong suốt thí nghiệm)

| Thuộc tính | Giá trị |
|---|---|
| Cụm | kind/minikube hoặc cluster lab; **1 node** cho RQ2 (để các tenant thực sự chia sẻ kernel) |
| Kernel | pin một version (`uname -r` ghi vào báo cáo); Tetragon yêu cầu ≥ 5.10 |
| Cảm biến | Tetragon TracingPolicy `desfam-syscall-trace` — **đúng 23 syscall** sensor-alphabet |
| Mô hình | DeSFAM artifacts trong `model/` (pin commit); window 15 / stride 3 / 43-dim |
| Ngưỡng (RQ2) | Chạy **song song 2 nhánh**: (a) **EMA động** (mặc định DeSFAM, `EMA_TMIN` trong `.env.uat`) và (b) **`T_0` cố định** (`results.json`) — cùng luồng syscall, để tách né-tránh-do-EMA khỏi nhiễu-lấn-đặc-trưng |
| IV (RQ2) | **Cường độ nhiễu noisy-neighbor** {0, moderate, high} từ pod đồng-trú (`10-noisy-neighbor`, xem `07`) |
| Namespace | tấn công + lành tính + noisy-neighbor trong `demo`; detector trỏ `TARGET_NAMESPACE=demo` |
| Tài nguyên pod | requests/limits **pod tấn công** cố định mọi mức nhiễu; chỉ pod noisy-neighbor đổi tải |

## Triển khai

- **Cảm biến + cluster** (Tuần 1): `DeSFAM/Kubernetes/tetragon/install.sh` → `kubectl apply -f
  tetragon/tracing-policy.yaml` → `kubectl apply -f tetragon/nodeport.yaml`.
- **Mô hình** (Tuần 1): train (`DeSFAM/training`, docker-compose) hoặc dùng `model/` đã có; nạp vào
  detector (`DeSFAM/inference`, `docker compose --env-file .env.uat up -d --build`).
- **Workload** (Tuần 2): `normal/app.yaml` (benign) + `attack/*.yaml` + **privesc bổ sung** (`08` loud,
  `09` stealthy) + **`10-noisy-neighbor`** (xem `07_attack_suite.md`).
- **Nhiễu đối kháng (RQ2)** (Tuần 3): pod `10-noisy-neighbor` đồng-trú bơm tải benign ở **3 mức cường độ
  {0, moderate, high}**; detector ghi **`T_EMA(t)`** + chấm điểm cho **cả 2 nhánh ngưỡng** để đo né tránh.

## Tính lặp lại (reproducibility)

- Pin: image Tetragon (Helm chart version), model artifacts (`model/` + `results.json` ngưỡng), manifest
  (commit hash), `.env.uat` (ngưỡng live).
- Mọi điểm bất thường thô + `T_EMA(t)` + nhãn cửa sổ (ns/pod/scenario/attack_class/noise_level/threshold_arm/timestamp) lưu `data/scores_*.csv`.
- `test_feature_parity.py` đảm bảo `train.py` ↔ `featurizer.py` sinh vector **bit-identical** (train ↔ serve cùng không gian đặc trưng).
- Phụ lục tái lập: một script chạy tuần tự deploy → attack → xuất CSV → phân tích (≤15 phút).
