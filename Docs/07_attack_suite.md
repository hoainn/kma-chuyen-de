# 07 — Bộ kịch bản tấn công (Attack Suite) & mở rộng privesc

> Tương đương "thiết kế attacker" của Topic 03. Ở đây attacker = **bộ workload kịch bản hoá** trong
> `Kubernetes/attack/`. Đề tài **bổ sung kịch bản leo thang đặc quyền thuần** để đúng trọng tâm đề bài.

## Bộ kịch bản hiện có (7 manifest — `Kubernetes/attack/01..07`)

| Manifest | Tactic (MITRE) | Hành vi & syscall đặc trưng |
|---|---|---|
| `01-shell-spawn.yaml` | TA0002 Execution | spawn shell — `execve`, `clone` |
| `02-credential-access.yaml` | TA0006 | đọc `/etc/shadow`, SA token, `find *.key` — `openat`, `getdents64` |
| `03-discovery.yaml` | TA0007 | liệt kê hệ thống/mạng — `execve`, `openat` |
| `04-cryptominer.yaml` | TA0040 Impact | tải CPU, kết nối pool — `mmap`, `socket`, `connect` |
| `05-exfiltration.yaml` | TA0009/10 | gom + đẩy dữ liệu — `socket`, `connect`, `splice` |
| `06-rootkit.yaml` | TA0005 Defense Evasion | nạp module/ẩn — `init_module`, `finit_module`, `bpf`, `memfd_create` |
| `07-lateral-movement.yaml` | TA0008 | quét/kết nối nội bộ — `socket`, `connect` |

> Các kịch bản chạy **lặp** (`while true; do … sleep`) → dấu hiệu đậm, là **cận trên** recall (xem `03`).

## Bổ sung BẮT BUỘC — leo thang đặc quyền thuần (TA0004)

7 manifest trên **chưa có TA0004 thuần**. Đề tài thêm các kịch bản sau để đúng đề bài "leo thang đặc
quyền" (chỉ **mô phỏng dấu hiệu syscall**, không khai thác 0-day):

| Kịch bản mới | Tactic | Syscall đặc trưng (đều trong sensor-alphabet) | Cách mô phỏng (mô tả, không payload) |
|---|---|---|---|
| `08-privesc-setuid` (**loud**) | TA0004 | `setuid`, `setgid`, `capset`, `prctl`, `execve` | tiến trình hạ/đổi UID-GID, thao tác capability rồi exec |
| `09-container-escape` (**stealthy**) | TA0004/TA0005 | `unshare`, `setns`, `pivot_root`, `mount`, `ptrace` | thao tác namespace/mount đặc trưng thoát container; bản **rải mỏng/one-shot** để mô phỏng lén |

- **Loud vs stealthy (then chốt RQ2):** `08` là **loud** (nhiều syscall đặc quyền dồn dập, khó che);
  `09` có bản **stealthy** (ít syscall, rải mỏng, one-shot/low-and-slow) — lớp dễ bị nhiễu noisy-neighbor
  che khi EMA bị đẩy lên. Profile **độ né tránh theo loud vs stealthy** là kết quả chính của RQ2.
- **Đối chứng lành tính cần kề**: một số syscall đặc quyền cũng xuất hiện hợp lệ (vd `mount` ở init
  container, `prctl` ở runtime). Phải có workload lành tính dùng các syscall này để FPR phản ánh thực tế.

## Noisy-neighbor workload (IV của RQ2 — `10-noisy-neighbor.yaml`)

Pod **đồng-trú** trên cùng node với tenant bị tấn công, sinh **tải benign hỗn loạn** ở các mức cường độ
{none, moderate, high} để đẩy ngưỡng EMA của DeSFAM:

| Thành phần nhiễu | Cách tạo (benign, không tấn công) | Syscall nền bơm vào |
|---|---|---|
| Memory churn | `stress-ng --vm` cấp/giải phóng vùng nhớ lớn lặp lại | `mmap`, `mprotect` |
| CPU/scheduler stress | `stress-ng --cpu` nhiều worker | (context-switch, `clone`) |
| I/O syscall tần suất cao | vòng lặp mở/ghi/đóng file tạm, exec ngắn | `openat`, `execve`, `clone` |

- **Cường độ** điều khiển bằng số worker / kích thước vùng nhớ / tần suất vòng lặp → 3 mức rời rạc.
- **Hoàn toàn benign**: không gọi syscall leo thang đặc quyền (`setuid/capset/ptrace/unshare/mount`) —
  để né tránh (nếu có) đến **thuần tuý từ việc đẩy ngưỡng**, không phải do nhiễu tự trông giống tấn công.
- Ghi `T_EMA(t)` của detector suốt mỗi trial để gắn cường độ nhiễu ↔ mức ngưỡng (DV5, xem `04`).

## Quy ước nhãn (ground truth cho recall/FPR)

- Mỗi cửa sổ gắn nhãn `(namespace, pod, scenario, attack_class∈{loud,stealthy}, noise_level∈{0,mod,high}, threshold_arm∈{ema,fixed}, t_start, t_end, label∈{benign,attack})`.
- Khoảng tấn công xác định bằng **timestamp Tetragon** của syscall đặc trưng đầu tiên/cuối cùng của pod
  tấn công; cửa sổ giao với khoảng này = `attack`, còn lại = `benign`.
- **Loại cửa sổ warm-up** (vd 30s đầu mỗi pod) khỏi cả TP và FP (confounder #1, `04`).

## Đồng nhất giữa các điều kiện (ceteris paribus)
- Cùng image nền, cùng requests/limits cho pod tấn công, cùng namespace, cùng model/cảm biến cho mọi
  điều kiện. Chỉ đổi **cường độ nhiễu noisy-neighbor** (IV chính) và **loại tấn công** (loud/stealthy).
- Chạy **song song 2 nhánh ngưỡng** (EMA động vs `T_0` cố định) trên **cùng luồng syscall** để tách
  "né tránh do EMA" khỏi "nhiễu lấn đặc trưng" (xem `01` §4 và `04`).

## Đạo đức
- Kịch bản dùng kỹ thuật **mô phỏng dấu hiệu** (gọi syscall đặc quyền trong container lab), không nhúng
  exploit kernel tái sử dụng được. Chạy trong cụm cô lập. Không công bố payload gây hại ngoài học thuật.
