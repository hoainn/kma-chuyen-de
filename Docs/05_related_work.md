# 05 — Lược khảo tài liệu (Related Work)

> Định vị đề tài trong dòng nghiên cứu **phát hiện bất thường runtime dựa system call cho
> container/cloud**. Corpus đầy đủ + URL: `09_literature_corpus.md`. Trích dẫn nền tảng (VAE 2013,
> Isolation Forest 2008, Edgar & Manz 2017) giữ trong `references.bib` bất kể năm.

## A. Phát hiện xâm nhập dựa system call (nền tảng)

Phát hiện bất thường trên chuỗi syscall có truyền thống dài (Forrest et al., "self/non-self"; ADFA-LD).
Ý tưởng cốt lõi: tiến trình lành tính sinh **chuỗi/tần suất syscall ổn định**; lệch khỏi phân phối đó
là dấu hiệu xâm nhập. **DongTing** (dataset đề tài dùng) hiện đại hoá hướng này bằng syscall thu từ
kernel fuzzing/syzbot ở quy mô lớn (chuỗi normal vs bug-PoC). Hạn chế chung: nhạy với nhiễu nền và **dịch
chuyển phân phối** giữa môi trường huấn luyện và vận hành (→ chính là RQ2 của đề tài).

## B. DeSFAM — cơ chế cơ sở (base)

**DeSFAM** (Zehra et al., IEEE Access 2025) là khung **eBPF + AI** cho an ninh runtime container đám
mây. Lõi phát hiện **SyscallAD** kết hợp **VAE** (lỗi tái tạo) + **Isolation Forest** (điểm cô lập)
theo **ensemble**, huấn luyện **chỉ trên hành vi lành tính** (normal-only) trên cửa sổ trượt syscall.
Đặc trưng mỗi cửa sổ gồm tần suất/độ rời rạc syscall, thống kê (entropy, unique-count…), tần suất theo
**nhóm chức năng** (process/file/memory/network/security…), và khớp **PrefixSpan** với CSDL mẫu lành
tính. Trong repo này, DeSFAM được tái lập với **sensor-alphabet 23 syscall an ninh** (gồm đúng nhóm
syscall leo thang đặc quyền/escape: setuid/capset/unshare/setns/pivot_root/mount/ptrace/bpf), triển khai
live qua **Tetragon** trong K8s. → Đề tài **không sáng tạo cơ chế mới** mà **đặc tả khách quan** năng
lực của DeSFAM trên bài toán privesc multi-tenant.

## C. eBPF + ML cho an ninh runtime K8s (SOTA cùng thời)

- **eBPF-PATROL (2025)** — agent eBPF nhận diện đe doạ runtime, **trực tiếp nhắm leo thang đặc quyền /
  container escape** qua syscall + giới hạn vượt quyền → **peer SOTA gần nhất** với đề tài; dùng để đối
  chiếu cách tiếp cận và định vị đóng góp.
- **Programmable System Call Security with eBPF (2023)** — cưỡng chế an ninh syscall bằng eBPF (so với
  seccomp); nền tảng cho luận điểm "vì sao eBPF/Tetragon là cảm biến phù hợp".
- **FedMon (2025)** — giám sát eBPF liên hợp, phát hiện bất thường phân tán đa cluster → hướng mở rộng
  multi-tenant/multi-cluster (Discussion).
- **LIGHT-HIDS (2025)** — HIDS ML nhẹ trên syscall với **novelty detection one-class** (cùng họ
  normal-only như SyscallAD) → so sánh thiết kế đặc trưng/độ nhẹ.
- **LM for Novelty Detection in System Call Traces (2023)** — mô hình ngôn ngữ phát hiện chuỗi syscall
  mới lạ → hướng hiện đại thay cho VAE/iForest (Discussion: future work).

## D. Chỉ số & cách đánh giá trong văn liệu

| Metric | Ý nghĩa | Liên hệ đề tài |
|---|---|---|
| **AUC-ROC / AP** | năng lực phân tách độc lập ngưỡng | DV4 chính (base-rate cao → ưu tiên AP) |
| **Recall @ FPR cố định** | độ phủ ở mức báo động giả chấp nhận được | DV1 + DV2 |
| **Detection latency** | độ kịp thời cảnh báo runtime | DV3 (đặc thù live, ít paper báo cáo → đóng góp) |
| **Robustness/Evasion** | bền vững trước né tránh đối kháng | Discussion (đối kháng, `advrobustseq2025`) |

## E. Khoảng trống nghiên cứu (Research Gap) — đề tài lấp

1. **Thiếu đặc tả live cho privesc cụ thể:** DeSFAM báo cáo headline tổng hợp trên DongTing
   (AUC≈0.94…), **không** tách năng lực theo **loại tấn công** (đặc biệt privesc/escape) trong môi
   trường K8s **live**. Đề tài đo per-scenario + latency live.
2. **Ảnh hưởng multi-tenant chưa định lượng:** noisy-neighbor làm dịch chuyển phân phối syscall nền →
   tác động lên recall/FPR chưa được đo có hệ thống (đúng RQ2).
3. **Khoảng cách dataset↔vận hành:** mô hình học trên DongTing (syzbot) rồi suy luận trên TTP K8s thật —
   mức khái quát hoá là câu hỏi mở mà đề tài kiểm chứng trực tiếp bằng 7+ manifest.

---

## Bảng claim → nguồn (kiểm chứng khi viết báo cáo)
| # | Tuyên bố | Nguồn |
|---|---|---|
| a | DeSFAM = eBPF + VAE+iForest ensemble, normal-only, trên syscall window | DeSFAM 2025 (`desfam_original`) |
| b | eBPF agent phát hiện privesc/container-escape qua syscall | eBPF-PATROL 2025 |
| c | eBPF cưỡng chế an ninh syscall (vs seccomp) | ProgSyscall 2023 |
| d | One-class novelty detection trên syscall nhẹ | LIGHT-HIDS 2025 |
| e | LM phát hiện chuỗi syscall mới lạ | LM-Novelty 2023 |
| f | Robustness của sequence anomaly detection trước né tránh | AdvRobustSeq 2025 |

> Khi viết: chỉ đưa claim **đã đọc/kiểm chứng**; ghi đúng năm/venue; phân biệt phát hiện của nguồn vs
> của đề tài; tránh over-claim (đặc biệt: KHÔNG coi headline DeSFAM là mục tiêu per-model — xem CLAUDE.md).
