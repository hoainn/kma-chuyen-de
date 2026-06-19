# 11 — Kết quả thực nghiệm RQ2 (né tránh ngưỡng EMA, trên DongTing thật)

> Thực nghiệm: `../Experiment/` (chạy 2026-06-18). Dữ liệu: **DongTing thật** (`DongTing_Official/npz`,
> chuỗi syscall-ID đã mã hoá). Detector kiểu DeSFAM huấn luyện **normal-only** trên DongTing; thí nghiệm
> né tránh chạy trên **điểm bất thường thật** của detector. Tái lập: xem `../Experiment/README.md`.

> **⚠️ ĐÍNH CHÍNH (2 điểm) — đọc trước:**
> 1. **Khung diễn giải (quan trọng):** `ema_uncond` KHÔNG phải "DeSFAM như công bố". Đối chiếu trực tiếp
>    **Algorithm 2** của DeSFAM (nhánh `else`, dòng 13–17) cho thấy DeSFAM đặc tả **cập nhật CÓ điều kiện**
>    (chỉ trên cửa sổ không bị gắn cờ) — tức `ema_cond` **chính là Algorithm 2 của DeSFAM**, không phải
>    "biện pháp giảm thiểu do ta đề xuất". Chỉ phần lời quanh công thức (3) ("cập nhật sau mỗi cửa sổ") đọc
>    ra vô điều kiện. Đóng góp đúng = **vạch ra sự thiếu nhất quán Eq.(3) ↔ Algorithm 2** và chứng minh dạng
>    có điều kiện là *thiết yếu về an ninh*. (Xem `12_threats_to_validity.md` + báo cáo `Academic_Report/`.)
> 2. **Số liệu:** các bảng dưới là **lần chạy dense-AE cũ** (evasion 28.5%, AUC 0.848, ρ=−0.59). Báo cáo
>    hiện dùng **VAE thật (TensorFlow/Keras)**: AUC **0.832**, evasion **74.2%**, ρ=**−0.74**, `ema_cond`=0%.
>    Lấy `Academic_Report/` làm nguồn chuẩn.

## A0. Pipeline ba pha tách biệt (độ tin cậy — chống rò rỉ) — Hình `fig0_pipeline.png`
Mỗi pha làm gì (đầu vào → thao tác → đầu ra):

| Pha | Đầu vào | Thao tác chính | Đầu ra |
|---|---|---|---|
| **P0 Data** | DongTing (chuỗi syscall ID) | lọc sensor-alphabet (24) + windowing (10/2) | cửa sổ train/cal/test |
| **P1 Train** | cửa sổ normal-train (50.877) | featurize → fit RobustScaler + AE + iForest + comp-scalers (**normal-only**) | mô hình + scaler |
| **P2 Calibrate** | cửa sổ CAL (½ benign-val + ½ attack) | đặt `T0`=p99.5, `T_op`=p95, `T_min`=p50; mốc dải nhiễu (p50/p90) + tấn công (p99.9) — **chỉ trên CAL** | ngưỡng + mốc dải |
| **P3 Test** | cửa sổ TEST **rời rạc** | chấm điểm ensemble; đo AUC/AP; xuất điểm | điểm TEST; AUC 0.848 |
| **Thử nghiệm** | điểm TEST + ngưỡng/dải (CAL) | dựng dòng warm→noise→attack; 3 chính sách × nhiễu × loại; 40 trial/ô | recall/evasion/`T_EMA` + kiểm định |

Vì **CAL ∩ TEST = ∅**, không ngưỡng/dải nào được fit trên chính cửa sổ dùng để đánh giá → loại rò rỉ
hiệu chỉnh. (AUC dưới đây đo trên TEST.)

## A1. Trích chọn đặc trưng (Hình `fig_features.png`)
Cho cửa sổ `W=(s_1..s_L)`, `L≤10` (đã lọc alphabet, `|A|=24`); `c_v` = số lần xuất hiện syscall `v`:
- **freq** (24): `freq_v = c_v/L`.
- **disc** (15): `disc_d = 1[c_d>0]` cho 15 syscall đặc quyền hiếm (setuid/capset/ptrace/unshare/setns/mount/bpf…) —
  **hiện diện nhị phân, chống pha loãng** (một `ptrace` đơn lẻ vẫn lật bit).
- **stats** (3): entropy `H=-Σ freq_v·log2(freq_v)`, `unique=|{c_v>0}|/24`, `maxfreq=max(c_v)/L`.
- → `x ∈ ℝ⁴²` → `RobustScaler(p1,p99)` fit normal-train → ensemble `A(W)=0.7·z(‖AE(x̃)−x̃‖²)+0.3·z(−iForest(x̃))`.
Train ↔ serve dùng **cùng** không gian 42 chiều.

## A. Detector kiểu DeSFAM trên DongTing (cơ sở cho RQ1) — có lọc sensor-alphabet
- **Lọc sensor-alphabet** (24 syscall an ninh) **trước khi windowing** (như repo CLAUDE.md): bỏ
  read/write/futex… giữ execve/clone/setuid/setgid/capset/openat/…/ptrace/bpf… Cửa sổ **len 10/stride 2**
  trên luồng đã lọc (DeSFAM Algo 2 đa độ dài {5,10,15}); seq <10 alphabet-syscall → 1 cửa sổ ngắn.
- Đặc trưng **42-dim** = freq(24 alphabet) + **disc(15 binary presence)** của nhóm đặc quyền hiếm
  (ptrace/unshare/setns/mount/capset/bpf…; dilution-robust) + 3 thống kê. Ensemble `A = 0.7·z(AE)+0.3·z(iForest)`.
- **Chất lượng (benign-val vs attack, thật): AUC-ROC 0.848, AP 0.977, recall@T0(p99.5) 0.078, FPR 0.32%.**
  → **AUC 0.848 khớp DeSFAM-test repo (~0.835)**. So với biến thể KHÔNG lọc (AUC 0.647), lọc sensor-alphabet
  **+0.20 AUC** — xác nhận tín hiệu privesc nằm ở nhóm syscall đặc quyền, đúng thiết kế DeSFAM.
- **Vì sao lọc + 24 có đủ?** Lọc vì: (i) chống pha loãng (call đặc quyền hiếm bị chìm trong read/write/futex →
  0.647→0.848); (ii) đồng bộ train↔serve (chính sách Tetragon chỉ trace nhóm này); (iii) chi phí thấp; (iv) bền
  với né tránh bằng khối lượng. **24 đủ phủ** vì bao trùm "điểm nghẽn nhân" của privesc/escape (setuid/capset/prctl,
  unshare/setns/pivot_root/mount, ptrace, init/finit_module/bpf, execve/clone, openat…); bằng chứng: AUC 0.848 ≈
  DeSFAM-test. **Giới hạn:** tập curated, không liệt kê đủ — tấn công qua syscall ngoài tập (io_uring, data-only) vô hình.
- *Lưu ý:* VAE của DeSFAM thay bằng **dense auto-encoder** (CPU, không TF/torch); động học EMA độc lập scorer.

## B. Thiết kế RQ2 (đối kháng) — 3 chính sách ngưỡng trên CÙNG luồng điểm thật
- `fixed`: T cố định tại **T_op = p95 benign = 32.66** (FPR≈5%).
- `ema_uncond`: **triển khai theo nghĩa đen phần lời Eq.3** ("cập nhật sau mỗi cửa sổ"), cập nhật **mọi cửa sổ**: `T_{t+1}=max(T_min, βT+(1−β)A)`, β=0.9, T_min=p50.
- `ema_cond`: **đúng theo Algorithm 2 của DeSFAM** — **chỉ cập nhật trên cửa sổ KHÔNG bị gắn cờ (A≤T)** → T **đơn điệu không tăng**.
- Nhân tố: **cường độ nhiễu** {none,moderate,high} = cửa sổ benign thật theo dải quantile điểm;
  **lớp tấn công** {stealthy = attack thật trong (T_op, p99.9_benign=77.3], loud = attack thật > 77.3}. 40 trial/ô.

## C. Kết quả chính (điểm thật DongTing)

### C1. H1₂a — nhiễu↑ ⇒ recall stealthy của EMA-paper giảm (xu hướng đơn điệu) ✅
| Cường độ nhiễu | recall `ema_uncond` (stealthy) |
|---|---|
| none | 0.856 |
| moderate | 0.852 |
| high | 0.715 |
- Kruskal–Wallis **H=54.0, p=1.9e-12**; Spearman **ρ=−0.591, p=1.2e-12** → **bác bỏ H0₂**: nhiễu làm
  suy giảm recall có ý nghĩa, đơn điệu mạnh theo cường độ.

### C2. H1₂b — né tránh do bơm ngưỡng (high noise, stealthy) ✅
| Chính sách | recall | Evasion (bắt-bởi-fixed, bị-bỏ-sót) | T tại lúc tấn công |
|---|---|---|---|
| `fixed` | **1.00** | — | 32.66 |
| `ema_uncond` (nghĩa đen Eq.3) | **0.715** | **28.5%** | 42.5 → đỉnh ~52 trong đợt tấn công |
| `ema_cond` (Algorithm 2) | 1.00 | **0.0%** | ~0.03 (chạm sàn) |
- Wilcoxon ghép cặp `fixed > ema_uncond`: **p=1.7e-8**. Nhiễu (+ chính cửa sổ tấn công) **bơm ngưỡng EMA
  toàn cục vượt T_op** → che **28.5%** cửa sổ tấn công lén mà ngưỡng cố định bắt được.
- `fig1_threshold_trajectory.png`: `ema_uncond` vọt lên ~40 khi nhiễu bắt đầu, ~52 khi tấn công; `ema_cond`
  giảm về sàn, **không bao giờ bơm**; `fixed` phẳng tại 32.66.

### C3. Algorithm 2 (cập nhật có điều kiện) vốn đã vô hiệu hoá né tránh ✅
`ema_cond` (= Algorithm 2 của DeSFAM) giữ T **đơn điệu không tăng** (cửa sổ A>T bị bỏ qua, không kéo T lên)
⇒ **evasion 0%, recall 1.0** cho **cả hai** lớp tấn công. → **Phát hiện:** lỗ hổng chỉ xuất hiện khi triển
khai theo nghĩa đen Eq.3; Algorithm 2 đã miễn nhiễm. Đóng góp = làm rõ điều này, không phải đề xuất cơ chế mới.

### C4. Bất ổn của EMA Eq.3 — TỰ-bơm bởi chính tấn công (lớp loud) ⚠️
| loud (high noise) | recall |
|---|---|
| `fixed` | 1.00 |
| `ema_uncond` | **0.456** (evasion ~54%) |
| `ema_cond` | 1.00 |
- Khác với kỳ vọng "loud khó che", dưới `ema_uncond` lớp loud **cũng bị bỏ sót ~42% khi noise=none (lên ~54% khi nhiễu cao)**
  (recall none 0.575): điểm cực cao của **chính cửa sổ tấn công** kéo T lên (self-inflation) → các cửa sổ
  tấn công sau rơi dưới ngưỡng tự nâng. → Đây là bằng chứng **EMA Eq.3-như-viết bất ổn với mọi điểm cao**
  (tự thân tấn công lẫn nhiễu lân cận), KHÔNG phải "control loud kháng che". `ema_cond` vá cả lỗi này (recall 1.0).

## D. Diễn giải & đóng góp
1. **Phát hiện đối kháng (định lượng, dữ liệu thật):** ngưỡng EMA động *như công bố* (Eq.3, cập nhật mọi
   cửa sổ) **bị né tránh** bởi noisy-neighbor trong multi-tenant — 28.5% evasion lớp lén (p<1e-7), xu hướng
   đơn điệu theo nhiễu (ρ=−0.59). Thêm: Eq.3 còn **tự-bơm** bởi chính cửa sổ tấn công (lớp loud ~50%).
   DeSFAM/FedMon **chưa kiểm thử** kịch bản này (chỉ test workload chuẩn).
2. **Phát hiện về đặc tả:** cập nhật ngưỡng **chỉ trên cửa sổ không bị gắn cờ** chính là **Algorithm 2 của
   DeSFAM** (bất biến với bơm-ngưỡng, 0% evasion). Lỗ hổng chỉ phát sinh khi triển khai theo nghĩa đen công
   thức (3). Đóng góp = vạch ra sự thiếu nhất quán Eq.(3) ↔ Algorithm 2 và chứng minh cập nhật có điều kiện
   là thiết yếu về an ninh.
3. **Cảnh báo vận hành (SOC):** cơ chế ngưỡng tự-thích-ứng có thể tạo điểm mù — kẻ tấn công đồng-trú chủ
   động bơm nhiễu benign để hạ độ nhạy detector trên toàn node.

## E. Mối đe doạ tính hợp lệ (trung thực)
- **Construct/scorer:** dense-AE thay VAE (động học EMA độc lập scorer; điểm tuyệt đối khác VAE thật).
- **Internal:** thứ tự luồng (warm→noise→attack) trừu tượng hoá lập lịch co-tenant thật; giả định **một
  ngưỡng toàn cục** (đúng Eq.3) — nếu detector giữ ngưỡng **per-container** thì hiệu ứng khu trú (xác minh khi live — Q5).
- **External:** DongTing = syscall kernel-fuzzing, không phải TTP privesc K8s live; detector AUC **0.848**
  (đã lọc sensor-alphabet, ngang DeSFAM-test) → kết quả vững hơn bản chưa lọc; **khoảng cách fixed↔EMA giữ nguyên** ở cả hai mức AUC.
- **Phạm vi:** thực nghiệm cơ chế trên điểm thật; **xác thực live K8s + Tetragon là W3–W4** (cần cụm).

## F. Ánh xạ tài liệu
- RQ1 ⟶ §A (detector AUC 0.848 trên DongTing thật). RQ2/H1₂a ⟶ §C1; H1₂b ⟶ §C2; giảm thiểu ⟶ §C3; bất ổn tự-bơm ⟶ §C4.
- Hình (CẤM pie): `fig0_pipeline.png` (pipeline ba pha), `fig_features.png` (trích chọn đặc trưng), `fig1_threshold_trajectory.png` (cơ chế bơm ngưỡng), `fig2_recall_vs_noise.png` (box-plot recall×nhiễu×policy).
- Số liệu thô: `Experiment/output/{detector_report.json, trials.csv, summary.csv, stats.json}`.
