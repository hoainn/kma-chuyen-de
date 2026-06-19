# 04 — Metric & Mối đe dọa tính hợp lệ (Validity Threats)

## Định nghĩa metric (operational)

| Metric | Định nghĩa thao tác | Nguồn |
|---|---|---|
| **Recall / TPR** | `TP / (TP+FN)` trên **cửa sổ** thuộc khoảng thời gian tấn công, ở ngưỡng `T_0` | nhãn cửa sổ + score |
| **Precision** | `TP / (TP+FP)` | score + nhãn |
| **FPR** | `FP / (FP+TN)` trên cửa sổ workload **lành tính** (+ báo cáo **số alert/giờ** tuyệt đối) | score benign |
| **Detection latency** | `t(window alert đầu tiên) − t(syscall đặc trưng đầu tiên của hành vi)` (giây) | timestamp Tetragon |
| **AUC-ROC / AP** | trên toàn bộ điểm bất thường benign vs attack, **độc lập ngưỡng** | score |
| **`T_EMA(t)` (RQ2)** | giá trị ngưỡng động EMA theo thời gian; báo cáo `ΔT_EMA = T_EMA(high noise) − T_EMA(0)` | log detector |
| **Evasion Success Rate (RQ2)** | tỷ lệ cửa sổ tấn công **bị bỏ sót dưới EMA nhưng được bắt dưới `T_0` cố định** (cùng dữ liệu) — né tránh *thuần do ngưỡng nâng* | score × 2 nhánh ngưỡng |
| **Per-scenario × per-noise × per-arm** | mọi metric trên, **tách theo** loud/stealthy × cường độ nhiễu × nhánh ngưỡng (EMA/fixed) | bảng nhân tố |

> **Vì sao không chỉ Accuracy?** Cửa sổ lành tính áp đảo (base-rate) → accuracy gần 1 dù bỏ sót hết
> tấn công. Dùng **ROC/AP + recall@FPR cố định** và báo cáo **phân phối điểm** (CDF/box-plot) thay cho
> trung bình. Đây cũng là yêu cầu Phụ lục C (CẤM pie-chart, phải thể hiện phương sai).

## Đơn vị thực nghiệm & chống pseudoreplication

- **Đơn vị suy luận = một _trial_ (một lần deploy + chạy kịch bản)**, KHÔNG phải từng cửa sổ. Các cửa
  sổ trong cùng một trial **không độc lập** (cùng pod, cùng tiến trình) → nếu kiểm định coi mỗi cửa sổ
  là quan sát độc lập, p-value sẽ **bị lạm phát** (bài học từ review Topic 03).
- Vì vậy: chạy **≥ 5–10 trial / ô (kịch bản × cường độ nhiễu)**; tính **một metric tổng hợp/trial** (vd
  recall của trial, evasion-rate của trial); kiểm định/CI **ở mức trial** (bootstrap CI, Mann–Whitney /
  Jonckheere–Terpstra), không ở mức cửa sổ. Hai nhánh ngưỡng (EMA/fixed) chạy trên **cùng** luồng syscall
  của mỗi trial (paired) → so sánh ghép cặp, tăng power.

## Ba loại mối đe dọa tính hợp lệ

### Internal validity
- **Confounder #1 — nhiễu khởi động pod:** image pull + runtime init bắn `execve/clone/mmap/openat`
  dồn dập → cửa sổ đầu pod giống "bất thường" lành tính. → **loại trừ cửa sổ warm-up** (vd 30s đầu) hoặc
  gắn nhãn riêng; báo cáo FPR có và không có warm-up.
- **Confounder #2 — cộng tuyến nhân tố:** kịch bản rootkit tự dùng nhiều syscall đặc quyền → recall cao
  "dễ". → tách riêng kịch bản privesc *ít syscall/tinh vi* để đo năng lực thực, không gộp trung bình.
- **Rò rỉ dữ liệu:** scaler + ngưỡng `T_0` chỉ fit trên cửa sổ **lành tính** của train/validation
  (normal-only, đã đảm bảo trong `train.py`). Không bao giờ dùng cửa sổ tấn công để chọn ngưỡng.
- **Confounder #3 (RQ2) — né tránh do EMA vs nhiễu lấn đặc trưng:** recall giảm khi có nhiễu có thể do
  (a) EMA bị đẩy lên (lỗ hổng cơ chế — giả thuyết) hoặc (b) nhiễu làm méo chính điểm bất thường của cửa
  sổ tấn công. → **Tách bằng nhánh ngưỡng cố định `T_0`** chạy song song: né tránh quy cho EMA **chỉ khi**
  recall giảm dưới EMA *mà không* giảm dưới `T_0`. Nếu giảm cả hai → là (b), không phải lỗ hổng EMA.
  Báo cáo cả hai nhánh trung thực.

### Construct validity
- "Điểm bất thường cao" có thực sự = "leo thang đặc quyền"? Có thể chỉ là workload lành tính **hiếm**.
  → Phân tích định tính syscall trace của TP và FP; đối chiếu với dấu hiệu MITRE ở `03`. Kết hợp nhiều
  DV (recall + latency + AUC) thay vì một con số.
- **DongTing ≠ privesc:** mô hình học "bình thường" và "bất thường" từ syscall **kernel-fuzzing
  (syzbot)**. "Bất thường" nó học có thể không phải hình dạng của privesc thật → đo trên 7+ manifest K8s
  thật để kiểm chứng construct, không chỉ dựa test-set DongTing.

### External validity
- Workload kịch bản lặp lại = **cận trên** của recall; kẻ tấn công low-and-slow sẽ khó hơn. **Không**
  khái quát sang privesc lén lút hay 0-day kernel. Khai báo rõ ở Limitations (xem `03`).
- Một cluster / một kernel / một mô hình huấn luyện → external validity hẹp; nêu trong Abstract + Kết luận.

## Kiểm định thống kê

| Câu hỏi | Kiểm định | Điều kiện |
|---|---|---|
| RQ1: recall/precision/FPR/latency/AUC theo loại tấn công | ước lượng điểm + **bootstrap 95% CI** (ở mức trial) | ≥5 trial/điều kiện |
| RQ2a: nhiễu↑ → recall↓ / FPR↑ (xu hướng đơn điệu) | **Jonckheere–Terpstra** (xu hướng theo mức nhiễu) hoặc hồi quy; Mann–Whitney từng cặp mức | α = 0.05, mức trial |
| RQ2a: nhiễu↑ → `T_EMA`↑ | tương quan/hồi quy `T_EMA` theo cường độ nhiễu | mức trial |
| RQ2b: né tránh **do EMA** (không do nhiễu lấn) | **so ghép cặp EMA vs `T_0`** (Wilcoxon signed-rank trên evasion-rate/trial) | cùng luồng syscall |
| Độc lập ngưỡng | **ROC/AP** + recall@FPR={0.01, 0.05} | trên điểm bất thường |

## Đối chứng có kiểm soát (đã tích hợp) + baseline tuỳ chọn
**Đã có sẵn nhân tố đối chứng:** (1) **cường độ nhiễu** {0/mod/high} = IV thao tác được; (2) **nhánh
ngưỡng EMA vs `T_0` cố định** = đối chứng ghép cặp để quy nhân quả né tránh. Hai cái này đã thoả yêu cầu
"thiết kế đối chứng có kiểm soát" của rubric.

Baseline **tuỳ chọn** nếu còn thời gian (W4), để mở rộng:
- **Falco với rule mặc định** trên cùng syscall stream + cùng tấn công → so recall/FPR rule-based vs
  anomaly-based; hoặc
- **Mô hình full-syscall** (không sensor-alphabet) → minh hoạ tại sao chọn alphabet 23 syscall
  (repo đã ghi nhận mô hình full-syscall suy biến: benign ≈ attack, `docs/auto-iter-log.md` iter-2).
Khi đó IV = cơ chế phát hiện; giữ nguyên threat model + workload (ceteris paribus).
