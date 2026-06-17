# Report V2 — Outline

**Đề tài:** Phát hiện tấn công leo thang đặc quyền trong Kubernetes dựa trên lời gọi
hệ thống — Nghiên cứu mô-đun phát hiện bất thường **SyscallAD (SysAD)** của framework **DeSFAM**.

**GVHD:** TS. Vũ Thị Vân — **HV:** Nguyễn Ngọc Hoài (CHAT4P006)

---

## Nguyên tắc định khung (khác V1)

- **Đối tượng nghiên cứu = mô-đun SyscallAD của DeSFAM** (không phải "một cơ chế tự đề xuất").
- DeSFAM (toàn framework) = **bối cảnh**; Mô-đun 1 (access-list/seccomp) và Mô-đun 3
  (adaptive enforcement) = **chỉ điểm qua** (mỗi cái ~½ trang).
- Hai phần "xương sống": (§4) SyscallAD — kiến trúc & thuật toán đầy đủ; (§5) SyscallAD
  phát hiện leo thang đặc quyền như thế nào (ánh xạ syscall→kỹ thuật→đặc trưng).
- Phần triển khai trực tiếp/Docker/Tetragon **giảm tải**, đặt lại vai trò: bằng chứng
  *SyscallAD hoạt động khi phục vụ*, không phải trọng tâm.
- Giữ kết quả thực (DongTing offline + phân tách trực tiếp), đối chiếu **trung thực** với
  số liệu công bố của SyscallAD.

---

## Bố cục chi tiết

### Tóm tắt + Từ khóa
Nêu rõ: nghiên cứu **mô-đun SyscallAD** của DeSFAM, áp dụng cho phát hiện leo thang đặc
quyền qua syscall trong K8s; tái lập + đánh giá trên DongTing; kết quả số chính.

### 1. Mở đầu
- 1.1 Bối cảnh: Kubernetes, container chia sẻ nhân, syscall là bề mặt tấn công.
- 1.2 Vấn đề: leo thang đặc quyền / thoát container; vì sao phát hiện dựa trên syscall.
- 1.3 **Đối tượng & phạm vi**: object = SyscallAD module; *focus SyscallAD, chỉ điểm qua
  Mô-đun 1 & 3* (tuyên bố tường minh ngay đây).
- 1.4 **Mục tiêu** (KHÔNG dùng research-question — đây là báo cáo học thuật dựa trên công
  trình đã công bố): nghiên cứu/hiểu SyscallAD; **tái lập trung thực để cung cấp bằng chứng
  nó hoạt động đúng như công bố**; tạo nền tảng tái lập cho nghiên cứu mở rộng.
- 1.5 Bố cục.

### 2. Cơ sở lý thuyết
- 2.1 Leo thang đặc quyền & thoát container trong K8s; nhóm syscall đặc quyền; họ CVE
  (PwnKit 2021-4034, Dirty Pipe 2022-0847, runc 2019-5736).
- 2.2 Giám sát syscall: seccomp (tĩnh) vs eBPF/Tetragon (động) — ngắn gọn, làm nền cho việc
  SysAD cần luồng syscall thời gian thực.
- 2.3 Phát hiện bất thường dựa trên syscall: từ "sense of self" → BoSC → học sâu một lớp;
  nền tảng **VAE** (lỗi tái cấu trúc) và **Isolation Forest** (cô lập ngoại lệ).
- 2.4 Bộ dữ liệu DongTing (nguồn huấn luyện/đánh giá của SyscallAD).

### 3. Tổng quan framework DeSFAM *(thiết lập bối cảnh, rồi thu hẹp)*
- 3.1 Kiến trúc 3 mô-đun (hình kiến trúc).
- 3.2 **Mô-đun 1** — Lập danh sách truy cập syscall lai + hồ sơ seccomp: *điểm qua* (~½ trang).
- 3.3 **Mô-đun 3** — Thực thi thích ứng (risk scoring, MITRE, chặn eBPF/LSM): *điểm qua* (~½ trang).
- 3.4 Vị trí của SyscallAD trong framework & lý do chọn làm trọng tâm.

### 4. ★ Mô-đun SyscallAD — kiến trúc & thuật toán *(xương sống #1)*
- 4.1 Tổng quan luồng phát hiện (sơ đồ: stream syscall → cửa sổ → đặc trưng → VAE+iForest → điểm → ngưỡng → cờ).
- 4.2 Thu thập & tiền xử lý; **sensor alphabet** (23 syscall an ninh) — quyết định then chốt.
- 4.3 Kỹ thuật đặc trưng: cửa sổ trượt (15,3) → véc-tơ 43 chiều
  `[freq8 | disc15 | stats8 | cat10 | prefixspan2]` (hình feature-engineering).
- 4.4 PrefixSpan — Cơ sở dữ liệu mẫu lành tính (giảm dương tính giả).
- 4.5 VAE (encoder/decoder 32→8→32, MSRE, đa hạt giống, xử lý posterior collapse).
- 4.6 Isolation Forest (300 cây, contamination 0.02).
- 4.7 Hợp nhất tập hợp: `A = α·VAE + (1−α)·iForest`, α=0.7.
- 4.8 Ngưỡng: chọn không-rò-rỉ trên validation; ngưỡng động EMA của DeSFAM.
- 4.9 Thuật toán phát hiện SyscallAD (algorithm box).

### 5. ★ SyscallAD phát hiện leo thang đặc quyền như thế nào *(xương sống #2 — bám sát đề tài)*
- 5.1 Ánh xạ syscall đặc quyền → kỹ thuật leo thang/thoát container + ví dụ CVE
  (ptrace, setuid/capset, unshare/setns, mount/pivot_root, init_module, bpf) — hình privesc-syscalls.
- 5.2 Vì sao đặc trưng **disc (hiện diện nhị phân)** là tín hiệu cốt lõi; bền với pha loãng.
- 5.3 Ví dụ minh họa: một chuỗi tấn công (clone→setuid→execve / ptrace) đi qua featurizer và
  được SyscallAD chấm điểm cao như thế nào.

> **Thực nghiệm (§6–§7): KHÔNG lấy số từ `results.json`.** Phải **chạy lại** trên DongTing
> đầy đủ (bản hiện có chỉ là mẫu) rồi chỉ ghi **kết quả quan sát được** + nhận xét. §7 hiện
> để placeholder `\TODOexp{...}`.

### 6. Thực nghiệm & thiết lập (tái lập SyscallAD)
- 6.1 Nguyên tắc tái lập: mọi thứ chạy bằng Docker; results.json lưu vết.
- 6.2 Huấn luyện SyscallAD trên DongTing — cấu hình (bảng config từ results.json).
- 6.3 Phục vụ trực tiếp qua Tetragon trên K8s *(rút gọn so với V1 — chỉ đủ để chứng minh
  vận hành)*; căn chỉnh sensor-alphabet train↔serve (test_feature_parity).
- 6.4 Kịch bản tấn công leo thang đặc quyền mô phỏng theo họ CVE.

### 7. Kết quả & đánh giá
- 7.1 Hiệu năng offline DongTing: iForest / VAE / Ensemble (bảng: AUC, AP, F1, P, R).
- 7.2 **Đối chiếu trung thực** với số liệu công bố của SyscallAD (bảng so sánh + nhận định).
- 7.3 Phân tách điểm khi phục vụ trực tiếp: lành tính ≤0.19 vs tấn công 0.278–0.930 (hình live-separation).
- 7.4 Điều kiện thành công: căn chỉnh sensor alphabet (bài học chính).
- 7.5 (nếu có) Độ nhạy α / ngưỡng.

### 8. Thảo luận
SyscallAD đạt/chưa đạt gì so với công bố; ý nghĩa khoảng cách ngưỡng (AUC/AP cao vs F1 thấp);
hạn chế & đe dọa tính hợp lệ (mô phỏng vs khai thác thật, tắt đặc trưng thời gian, mất cân bằng lớp).

### 9. Kết luận & hướng phát triển
Trả lời lại 3 RQ. Hướng: ngưỡng tối ưu F1, bổ sung đặc trưng thời gian, kết nối Mô-đun 3 để
chuyển từ *phát hiện* sang *ngăn chặn*.

### Phụ lục: Tái lập kết quả (Docker)

---

## Tài sản tái sử dụng từ V1
- Hình: `desfam-architecture.png` (→§3.1, §4.1), `feature-engineering.png` (→§4.3),
  `privesc-syscalls.png` (→§5.1), `live-separation.png` (→§7.3).
- `references.bib`, scaffold `main.tex` + `build.sh` (XeLaTeX qua Docker).
- Số liệu: `DeSFAM/model/results.json` (offline), `.env.uat` (T_0=0.25, live).

## Khác biệt then chốt so với V1
| | V1 | V2 |
|---|---|---|
| Đối tượng | "cơ chế tự đề xuất", DeSFAM = tham chiếu | **SyscallAD module của DeSFAM** là đối tượng nghiên cứu |
| SysAD | rải rác trong methodology | **2 mục riêng, là xương sống** (§4, §5) |
| Mô-đun 1 & 3 | caveat rải rác | **§3.2/§3.3 điểm qua sạch sẽ** |
| Triển khai trực tiếp | nặng | rút gọn, đặt lại vai trò bằng chứng vận hành |
| Privesc↔syscall | mờ | **§5 chuyên sâu** |
