# 01 — Giả thuyết & Biến số

> Đầu ra Tuần 1. Mọi tuyên bố cần nêu dưới dạng **có thể bác bỏ (falsifiable)** — nguyên tắc cốt lõi
> của Hypothetico-deductive / Applied Experimentation (Edgar & Manz, Ch.9, Ch.11).

## 1. Phát biểu vấn đề

Trong cụm **Multi-tenant Kubernetes**, nhiều tenant chia sẻ kernel của cùng một node. Tấn công **leo
thang đặc quyền** (privilege escalation, MITRE **TA0004**) và **container escape** biểu hiện qua một
nhóm *system call đặc quyền* hiếm gặp ở workload thường: `setuid/setgid/capset`, `ptrace`, `unshare`,
`setns`, `pivot_root`, `mount`, nạp module (`init_module/finit_module`), `bpf`. Cơ chế dựa **chữ ký**
(Falco/seccomp) phụ thuộc danh sách luật cố định → bỏ sót biến thể mới. DeSFAM huấn luyện **chỉ trên
hành vi lành tính** (normal-only VAE + Isolation Forest) và gắn cờ cửa sổ syscall **lệch khỏi phân
phối bình thường** → kỳ vọng bắt được cả biến thể chưa biết. Câu hỏi: cơ chế này **phát hiện privesc
hiệu quả đến đâu**, và **bền vững ra sao** dưới nhiễu multi-tenant.

## 2. Giả thuyết

> **RQ1** mang tính đặc tả (ước lượng điểm + CI, theo từng loại tấn công); **RQ2** là **thực nghiệm
> đối kháng có kiểm soát** với biến độc lập thao tác được (cường độ nhiễu noisy-neighbor) + giả thuyết
> bác bỏ được.

### RQ1 — Năng lực phát hiện (đặc tả)
- Ước lượng và báo cáo **recall, precision, FPR, detection latency, AUC-ROC, AP** kèm **khoảng tin cậy
  bootstrap** trên tập cửa sổ {benign workload} ∪ {7+ kịch bản tấn công}.
- Tiêu chí "đạt" định trước (tránh HARKing): privesc/container-escape có **recall ≥ 0.80** ở ngưỡng
  vận hành `T_0` với **FPR ≤ 0.05** trên workload lành tính. (Ngưỡng tham chiếu, không phải kết luận.)

### RQ2 — Né tránh qua noisy-neighbor đẩy ngưỡng EMA (đóng góp chính)
> Cơ chế giả định: DeSFAM dùng **EMA** nâng ngưỡng bất thường để dập false-positive do "benign drift"
> (repo: `EMA_TMIN` trong `inference/.env.uat`). Tenant lân cận bơm **nhiễu benign hỗn loạn** có thể
> **đẩy `T_EMA` lên** → tấn công "lén" (ít syscall đặc quyền) rơi dưới ngưỡng → **bị bỏ sót**.
>
> **Làm rõ (đính chính khung):** lỗ hổng chỉ tồn tại nếu ngưỡng cập nhật **vô điều kiện** (cách đọc theo
> nghĩa đen công thức (3) của DeSFAM). **Algorithm 2** của DeSFAM lại đặc tả cập nhật **có điều kiện** (chỉ
> trên cửa sổ không bị gắn cờ) — vốn miễn nhiễm. Vậy H1₂b kiểm định *dạng vô điều kiện*; phần đối chứng
> `ema_cond` chính là Algorithm 2, không phải "biện pháp do ta đề xuất".

- **H0₂ (null):** Cường độ nhiễu noisy-neighbor **không** ảnh hưởng recall/FPR/`T_EMA`:
  `Recall(noise=0) = Recall(noise=high)`, `T_EMA` không tăng theo nhiễu.
- **H1₂a (suy giảm):** Tăng cường độ nhiễu làm **recall giảm** và/hoặc **FPR tăng** đơn điệu, có ý nghĩa
  thống kê (xu hướng theo mức nhiễu — Jonckheere–Terpstra/hồi quy; hoặc Mann–Whitney từng cặp mức).
- **H1₂b (né tránh qua EMA — mạnh hơn):** Nhiễu đẩy `T_EMA(t)` lên, và **escape "lén"** bị bỏ sót **nhiều
  hơn hẳn** dưới EMA so với **ngưỡng cố định** `T_0` cùng dữ liệu → né tránh *do cơ chế EMA*, không chỉ do
  nhiễu lấn đặc trưng. Đo **Evasion Success Rate** (cửa sổ tấn công bị bỏ sót do ngưỡng nâng).
- Kiểm định ở **mức trial** (α = 0.05); bootstrap CI cho hiệu số recall/evasion giữa các mức nhiễu.

## 3. Biến số

| Loại | Biến | Mức / Đo lường |
|---|---|---|
| **IV chính (RQ2)** | **Cường độ nhiễu noisy-neighbor** | {none=0; moderate; high} — pod đồng-trú bơm memory churn / CPU stress / I/O syscall benign tần suất cao (xem `07`) |
| **Nhân tố A (factor)** | Loại tấn công | loud {privesc-setuid, mount, rootkit/module} vs **stealthy** {container-escape nsenter/ptrace} — xem `07` |
| **Phụ thuộc (DV1)** | Recall / TPR | tỷ lệ cửa sổ tấn công có điểm ≥ ngưỡng (per-scenario × per-noise) |
| **Phụ thuộc (DV2)** | FPR | tỷ lệ cửa sổ lành tính có điểm ≥ ngưỡng (+ số alert/giờ tuyệt đối) |
| **Phụ thuộc (DV3)** | Detection latency | giây, từ syscall đặc trưng đầu tiên → window alert đầu tiên |
| **Phụ thuộc (DV4)** | AUC-ROC / AP | độc lập ngưỡng, trên điểm bất thường benign vs attack |
| **Phụ thuộc (DV5, RQ2)** | **`T_EMA(t)`** | giá trị ngưỡng động EMA theo thời gian (chứng minh nhiễu có đẩy ngưỡng lên) |
| **Phụ thuộc (DV6, RQ2)** | **Evasion Success Rate** | tỷ lệ cửa sổ tấn công bị bỏ sót *do* ngưỡng EMA bị nâng (vs ngưỡng cố định) |
| **Kiểm soát** | Mô hình & cơ chế ngưỡng | DeSFAM cố định: sensor-alphabet 23 syscall, window 15/stride 3, 43-dim; chạy **song song 2 nhánh ngưỡng**: EMA động vs `T_0` cố định (cùng dữ liệu) để tách cơ chế |
| **Kiểm soát** | Đích tấn công & tài nguyên | tenant bị tấn công cố định; cấp tài nguyên (cpu/mem limits) pod tấn công giữ nguyên mọi mức nhiễu |
| **Kiểm soát** | Phiên bản / node / kernel | pin image Tetragon + model (`model/`) + manifest; cùng node, ghi `uname -r`, CPU/RAM |

## 4. Confounder cần khử (chi tiết `04_metrics_and_validity.md`)

- **Sự kiện khởi động pod** (image pull, runtime init: `execve/clone/mmap/openat` dồn dập) tạo cửa sổ
  "bất thường" lành tính ngay đầu mỗi pod → **loại trừ cửa sổ warm-up** hoặc gắn nhãn riêng; nếu không
  sẽ thổi phồng cả recall lẫn FPR.
- **Cộng tuyến nhân tố:** vài kịch bản (rootkit) tự thân đã dùng nhiều syscall đặc quyền → recall cao
  "dễ"; phải tách riêng kịch bản privesc *tinh vi/ít syscall* để đo năng lực thực.
- **Rò rỉ dữ liệu:** ngưỡng và scaler **chỉ** fit trên cửa sổ lành tính của tập train/validation, không
  bao giờ chạm cửa sổ tấn công (đã đảm bảo trong `train.py`, normal-only).
- **Định danh tenant:** detector chấm điểm theo namespace/pod — đảm bảo nhiễu từ tenant lân cận không bị
  gán nhầm cho tenant bị tấn công khi đo RQ2.
- **(RQ2) Tách cơ chế né tránh:** recall giảm khi có nhiễu có thể do (a) **EMA bị đẩy lên** (lỗ hổng cơ
  chế) hoặc (b) **nhiễu lấn át đặc trưng cửa sổ** (suy giảm tín hiệu). → chạy **song song nhánh ngưỡng cố
  định `T_0`**: nếu recall **chỉ** giảm dưới EMA mà không giảm dưới `T_0` → đúng lỗ hổng EMA (H1₂b); nếu
  giảm cả hai → là (b). Đây là điều kiện then chốt để quy kết nhân quả.
