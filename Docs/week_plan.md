# Kế hoạch 4 tuần — Phát hiện leo thang đặc quyền (DeSFAM, Multi-tenant K8s)

| Tuần | Hoạt động | Đầu ra |
|---|---|---|
| **1** | Lược khảo phát hiện bất thường syscall + eBPF; **dựng cụm K8s + Tetragon** (TracingPolicy 23 syscall); tái lập DeSFAM (train hoặc nạp `model/`); chốt giả thuyết/biến/testbed/threat model/metric | `docs/*`, Tetragon chạy, detector live nạp model, pilot end-to-end (`docs/06`) |
| **2** | Triển khai kịch bản tấn công (`01..07`) + **privesc thuần** (`08-privesc-setuid` loud, `09-container-escape` stealthy) + **`10-noisy-neighbor`** (3 mức cường độ); cấu hình **log `T_EMA(t)`** + chạy **2 nhánh ngưỡng (EMA vs `T_0`)** + nhãn cửa sổ | manifest đầy đủ, pipeline `data/scores_*.csv` + `T_EMA` + nhãn ground-truth |
| **3** | Chạy **≥5–10 trial** mỗi ô **(loud/stealthy × cường độ nhiễu {0,mod,high})**; loại warm-up; thu điểm + `T_EMA` cho cả 2 nhánh ngưỡng | `data/scores_*.csv` đầy đủ theo nhân tố |
| **4** | RQ1: recall/FPR/latency/AUC per-scenario + CI. RQ2: xu hướng theo nhiễu (Jonckheere) + `T_EMA↑` + **evasion EMA vs `T_0` ghép cặp** (Wilcoxon); hoàn thiện báo cáo IEEE + **Phụ lục tái lập ≤15 phút** | ROC + box-plot + đường `T_EMA(t)`, bảng nhân tố, báo cáo 6–8 trang |

## TLTK chính
- Edgar & Manz (2017), *Research Methods for Cyber Security* — phương pháp luận (Applied Experimentation).
- DeSFAM (Zehra et al., IEEE Access 2025) — cơ chế cơ sở.
- eBPF-PATROL (2025) — peer SOTA privesc/container escape qua syscall.
- VAE (Kingma & Welling 2013), Isolation Forest (Liu et al. 2008) — nền tảng phương pháp.

## Công cụ
Kubernetes + Tetragon (eBPF) · DeSFAM (VAE+iForest, repo này) · DongTing dataset · Prometheus/Grafana/Loki
· Python (numpy/pandas/scikit-learn, bootstrap CI).

## Trạng thái hiện tại
- [x] Khung repo DeSFAM (training + inference + K8s manifest + Tetragon TracingPolicy)
- [x] Tái lập DeSFAM trên DongTing: ensemble test AUC 0.835 / recall ~0.26 / precision 0.999; val ~0.948
- [x] Detector live qua Tetragon: phân tách benign (≤0.19) vs attack (0.278–0.930), `T_0=0.25`
- [x] 7 kịch bản tấn công K8s (TA0002/05/06/07/08/09/40)
- [x] Tài liệu phương pháp luận PP-NCKH (`docs/00`–`09`, `week_plan`)
- [x] Lược khảo grounded NotebookLM (Q1+Q2+Q3) → `docs/10`; **chốt đóng góp đối kháng** (noisy-neighbor đẩy EMA → né tránh); xác nhận EMA Eq.3 (β=0.9, T_min, T0=p99.5) verbatim trong PDF
- [x] Verify số liệu metric DeSFAM/FedMon… đối chiếu PDF (`docs/10` PART A) — khớp (DeSFAM AUC0.94/AP0.87/FPR1.6%; FedMon non-IID…)
- [x] **Thực nghiệm cơ chế RQ2 trên DongTing thật** (`Experiment/`): detector kiểu DeSFAM **lọc sensor-alphabet → AUC 0.842** (≈ DeSFAM-test; +0.20 so với bản chưa lọc) + 3 chính sách ngưỡng (fixed/ema_uncond/ema_cond) → `docs/11_experiment_results.md`
  - [x] H1₂a: nhiễu↑ ⇒ recall stealthy↓ (Kruskal p=3e-17, Spearman ρ=−0.715)
  - [x] H1₂b: high-noise stealthy **evasion 33.2%** (fixed 1.0 vs ema_uncond 0.668, Wilcoxon p=1.6e-8); T bơm 32.6→~58
  - [x] Giảm thiểu: `ema_cond` (chỉ cập nhật cửa sổ không gắn cờ) → evasion 0% (cả 2 lớp); §C4: Eq.3 còn tự-bơm bởi loud (~50%)
- [ ] **W3–W4 (cần cụm):** dựng K8s+Tetragon; manifest `08/09/10`; chạy detector live 2 nhánh ngưỡng; đo `T_EMA(t)` + latency trên TTP thật → xác thực kết quả cơ chế
- [ ] Q5 (khi code live): ngưỡng EMA **toàn cục** hay **per-container**? (quyết định độ mạnh vector)
- [x] Nâng AUC detector bằng **lọc sensor-alphabet** (0.647 → 0.842); freq24+disc15+stats3
- [ ] (Tuỳ chọn) baseline Falco rule-based / so AUC full-syscall vs sensor-alphabet (đã có 0.647 vs 0.842)
- [ ] Báo cáo IEEE 6–8 trang (RQ1 §A, RQ2 §C `docs/11`; hình fig1/fig2) + Phụ lục tái lập (`Experiment/README.md`)
- [ ] Báo cáo IEEE 6–8 trang + Phụ lục tái lập ≤15 phút
