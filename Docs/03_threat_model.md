# 03 — Mô hình kẻ tấn công (Threat Model)

## Tác nhân: workload tấn công KỊCH BẢN HOÁ (không phải con người)

Đề tài giới hạn "kẻ tấn công" là **pod chạy kịch bản tấn công** (manifest) để (a) hoàn thành khối
lượng trong 4 tuần và (b) **tránh rào cản xét duyệt đạo đức** (không chủ thể người, không tấn công hệ
thống thật). Đây là quy ước phổ biến của các benchmark IDS dựa syscall (ADFA-LD, DongTing, Falco
test-suite).

## Giả định về kẻ tấn công

| Khía cạnh | Đặc tả |
|---|---|
| Vị trí | **Đã có chỗ đứng ban đầu** trong một pod tenant (post-exploitation) — đúng tình huống privesc |
| Mục tiêu | Leo thang đặc quyền trong container → chiếm quyền node / thoát container / sang tenant khác |
| Năng lực | Thực thi lệnh trong pod; thử các kỹ thuật privesc/escape phổ biến |
| Tri thức | RQ1: **không** biết có DeSFAM (cơ sở). RQ2: **biết có giám sát anomaly dùng ngưỡng động** và chủ động khai thác (đối kháng) |
| Đồng phạm (RQ2) | Kiểm soát thêm **một tenant lân cận** trên cùng node để bơm nhiễu benign → đẩy ngưỡng EMA (xem dưới) |

## Bề mặt tấn công theo MITRE ATT&CK & dấu hiệu system call

| Kịch bản | Tactic | Syscall đặc trưng (trong sensor-alphabet) |
|---|---|---|
| Shell spawn | TA0002 Execution | `execve`, `clone` |
| Credential access | TA0006 | `openat`(/etc/shadow, SA token), `getdents64` |
| Discovery | TA0007 | `execve` liệt kê, `openat` |
| **Privesc — setuid/capability** | **TA0004** | `setuid`, `setgid`, `capset`, `prctl` |
| **Container escape** | **TA0004 / TA0005** | `unshare`, `setns`, `pivot_root`, `mount`, `ptrace` |
| Rootkit / kernel module | TA0005 Defense Evasion | `init_module`, `finit_module`, `bpf`, `memfd_create` |
| Lateral movement | TA0008 | `socket`, `connect` |
| Exfiltration | TA0009/TA0010 | `socket`, `connect`, `splice` |

> 7 manifest hiện có (`Kubernetes/attack/01..07`) phủ TA0002/05/06/07/08/09/40 nhưng **chưa có TA0004
> thuần** — đề tài bổ sung kịch bản **privesc setuid/capability** và **container escape** (xem
> `07_attack_suite.md`) để đúng trọng tâm "leo thang đặc quyền".

**Phân loại loud vs stealthy (then chốt cho RQ2):**
- **Loud:** dùng nhiều syscall đặc quyền dồn dập (`setuid/capset`, `mount`, nạp module) → dấu hiệu đậm,
  khó che bằng nhiễu.
- **Stealthy:** ít syscall đặc quyền, rải mỏng (`nsenter`/`setns`, `ptrace` đơn lẻ) → **dễ bị nhiễu
  noisy-neighbor che** khi ngưỡng EMA bị đẩy lên. Đây là lớp tấn công đề tài kỳ vọng thấy né tránh được.

## Vector đối kháng RQ2 — noisy-neighbor đẩy ngưỡng EMA

DeSFAM nâng ngưỡng bất thường bằng **EMA** để dập false-positive do "benign drift". Kẻ tấn công khai
thác chính cơ chế tự-thích-ứng này: từ **một pod đồng-trú** (tenant lân cận, hợp lệ về hình thức) bơm
**nhiễu benign hỗn loạn** (memory churn, CPU stress, I/O syscall tần suất cao). Vì các pod chia sẻ
kernel/node, nhiễu này làm **phân phối syscall nền dịch chuyển** → `T_EMA` tăng → cửa sổ của tấn công
*stealthy* rơi dưới ngưỡng đã nâng → **bỏ sót (né tránh)**. Đây là **kịch bản đe doạ thực tế trong
multi-tenant** (tenant không tin cậy lẫn nhau) và là **đóng góp đối kháng** của đề tài (xem `01` H1₂b).

> **Lưu ý khung (đính chính):** kịch bản này chỉ thành công nếu ngưỡng cập nhật **vô điều kiện** (cách đọc
> theo nghĩa đen công thức (3)). **Algorithm 2** của DeSFAM đặc tả cập nhật **có điều kiện** (chỉ trên cửa
> sổ không bị gắn cờ) → miễn nhiễm. Vì vậy đóng góp thực chất là **vạch ra sự thiếu nhất quán Eq.(3) ↔
> Algorithm 2** và chứng minh dạng có điều kiện là thiết yếu về an ninh — không phải đề xuất một cơ chế mới.

## Mô hình quan sát của người phòng thủ

Người phòng thủ (DeSFAM) **chỉ thấy chuỗi system call** qua eBPF, theo cửa sổ trượt 15/stride 3, **không**
thấy nội dung file/network. Mọi quyết định phát hiện dựa trên độ lệch phân phối syscall so với hành vi
lành tính đã học. Đây là điều kiện **ceteris paribus** giữa các nhân tố: cảm biến và mô hình giống hệt,
chỉ đổi **loại tấn công** và **cường độ nhiễu noisy-neighbor** (IV chính của RQ2).

## Giới hạn đã biết (khai báo trung thực — External Validity)

> Workload kịch bản hoá **lặp lại đều đặn** (`while true; do … sleep`) → dấu hiệu syscall **đậm và dễ
> hơn** so với một privesc thật, thực hiện một lần, lẩn trong hoạt động bình thường. Kết quả vì vậy
> **chỉ khái quát cho lớp privesc rõ ràng/lặp lại**, là **cận trên (upper bound)** của recall, không
> phải năng lực với kẻ tấn công lén lút (low-and-slow). (Chi tiết: `04_metrics_and_validity.md`.)

## Ranh giới đạo đức

- Chạy hoàn toàn trong cụm K8s lab cô lập, không route Internet.
- Không thu thập dữ liệu người thật; không tấn công tài sản bên thứ ba.
- Không công bố payload/khai thác kernel có thể tái sử dụng gây hại ngoài phạm vi học thuật; kịch bản
  privesc dùng kỹ thuật **mô phỏng dấu hiệu** (gọi syscall đặc quyền) thay vì 0-day thực.
