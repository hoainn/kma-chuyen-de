# 10 — Lược khảo grounded từ NotebookLM (Q1: landscape + research gap)

> Nguồn: NotebookLM (Gemini 2.5) grounded trên **7 PDF** trong `../Reference/` (notebook
> `561b9e74-…`), truy vấn 2026-06-18, `source_format=footnotes`, session `888c743e`.
> ⚠️ Trích đoạn citation do trình trích xuất của NotebookLM lặp/nhiễu — **tin cậy `sourceName`
> (gán đúng file), nhưng kiểm lại số liệu trực tiếp trong PDF trước khi đưa vào báo cáo.**

## So sánh cách tiếp cận (7 nguồn)

| Nguồn | Phương pháp phát hiện | Dữ liệu | Live? | Multi-tenant? |
|---|---|---|---|---|
| **DeSFAM** (`desfam_original`) | Hybrid **VAE + Isolation Forest** trên chuỗi syscall + rule-based profiling + risk-scoring CVE/MITRE | **DongTing** (18,966 chuỗi) + CVE thực | **Live** (eBPF + LSM hooks, sub-ms) | **Có** (nhấn mạnh chống lan ngang trên shared kernel) |
| **Comparative eBPF tools** (`syairozi2025ebpfcompare`) | Rule-based + forensic: so sánh **Falco / Tetragon / Tracee** | Tấn công live (nsenter escape, stress-ng DoS, xmrig mining) | **Live** (đo MTTD) | **Không** (K8s chung, không nhắm multi-tenant) |
| **FedMon** (`fedmon2025`) | **Federated VAE + iForest** (FL: global VAE, local iForest) | Telemetry tự tạo (nginx/redis) trên kind + tấn công inject | **Live** (eBPF DaemonSet, inference local) | **Có** (thiết kế riêng cho multi-cluster, privacy-preserving) |
| **Multi-class threat + deception** (`aly2025k8sthreat`) | **Supervised**: PCA + Autoencoder (feature) → **Naive Bayes**; kèm deception (decoy pods) | Dataset K8s tự tạo từ **NetFlow** (Grafana SSRF, RunC injection) + OWASP ZAP | **Live** (KSniff→CICFlowMeter→KServe) | **Không** (⚠️ dùng **network packet**, KHÔNG phải syscall) |
| **LIGHT-HIDS** (`lighthids2025`) | **Unsupervised**: DeepSVDD feature + Isolation Forest novelty | **LID-DS** (brute-force, SQLi) | Thiết kế live, đo offline (Jetson Orin NX) | **Không** (nhắm edge/IoT, resource-constrained) |
| **SafeBPF** (`safebpf2024`) | **Cơ chế cô lập** (không phải detector ML/rule): SFI + ARM Memory Tagging sandbox eBPF | microbench + macrobench (Netperf/Apache/Nginx) + CVE | **Live** (instrument lúc JIT) | **Có** (cho phép eBPF unprivileged an toàn qua cgroups) |
| **Transformers/LLM IDS survey** (`translmidssurvey`) | Khảo sát: CNN/LSTM-Transformer, ViT, GAN-Transformer, LLM (BERT/GPT) | Hàng chục dataset public (CICIDS-2017, NSL-KDD, UNSW-NB15…) | Cả hai; nhấn **live bằng Transformer/LLM rất khó do latency** | Có (một phần: cloud/SDN) |

## Khoảng trống nghiên cứu (NotebookLM tổng hợp)

> Dù DeSFAM & FedMon đã kết hợp eBPF + hybrid unsupervised (VAE+iForest) cho phòng thủ live
> multi-tenant, vẫn còn các gap:

1. **Thiếu soi đối số syscall (argument inspection):** hầu hết chỉ dùng **chuỗi/tần suất syscall ID**,
   bỏ qua **đối số** (đường dẫn file, tham số exec) — vốn cốt yếu để bắt privesc tinh vi / 0-day escape.
2. **Khái quát hoá trên workload multi-tenant dị thể (non-IID):** workload đa dạng + **"benign
   behavioral drift"** → khó tránh false positive cao khi dựa hồ sơ baseline cụ thể.
3. **Chi phí tính toán cho phân tích ngữ cảnh sâu:** mô hình sequence sâu (Transformer/LLM) bắt được
   bất thường tinh vi nhưng **quá nặng cho luồng syscall tần suất cao, sub-ms** ở mức kernel.

## Hệ quả cho đề tài (định vị đóng góp)

- **Cảnh báo novelty:** DeSFAM (base) **tự nhận** đã "chống privesc/container escape" + "multi-tenant",
  FedMon cũng "multi-tenant". → đóng góp đề tài KHÔNG thể chỉ là "dùng DeSFAM phát hiện privesc
  multi-tenant" (đã được claim). Phải sắc hơn.
- **Hướng đóng góp phòng thủ được (bám gap):**
  - **Đo lường thực nghiệm theo từng loại tấn công** (privesc setuid/capability vs container escape
    namespace/mount) — các nguồn báo cáo *tổng hợp*, không tách per-attack → đề tài lấp bằng
    descriptive eval per-scenario (đúng `01_hypothesis.md` RQ1).
  - **Định lượng suy giảm do multi-tenant noisy-neighbor** (recall↓/FPR↑) — DeSFAM/FedMon *tuyên bố*
    multi-tenant nhưng **chưa đo độ suy giảm dưới nhiễu tenant lân cận trên cùng node** → đúng
    `01_hypothesis.md` RQ2 (gap #2 "benign drift / non-IID").
  - **Khoảng cách dataset↔vận hành**: DeSFAM huấn luyện *chỉ* trên DongTing (syzbot) rồi suy luận
    live → đề tài kiểm chứng mức khái quát sang TTP K8s thật (đúng threats-to-validity `04`).
- **Cần thận trọng (đối kháng):** gap #1 (argument inspection) là điểm yếu của chính DeSFAM (chỉ
  syscall ID, sensor-alphabet) → nêu trong Discussion như giới hạn cơ chế base + hướng mở rộng.

---

# Q2 — Số liệu định lượng + định vị đóng góp (NotebookLM, 2026-06-18)

> ⚠️ Body answer mạch lạc; **trích đoạn citation của NotebookLM bị nhiễu/lệch** (vd bảng timing
> LIGHT-HIDS bị gán nhầm sang safebpf). **Bắt buộc kiểm lại số trực tiếp trong PDF** trước khi đưa vào báo cáo.

## PART A — Metric báo cáo của từng nguồn ✅ ĐÃ VERIFY trực tiếp trong PDF (2026-06-18)

> Kiểm chứng bằng `pdftotext` trên `../Reference/*.pdf`. Kết quả khớp, kèm bổ sung:
> - **DeSFAM:** "94% precision, 90% recall, sub-ms enforcement, <1% overhead" (abstract) ✓; body:
>   **AUC = 0.94, AP = 0.87** (Fig.15 ROC/PR), **FPR = 1.6%** ("Lowest false positive rate at 1.6%") ✓;
>   có mục "ATTACK DETECTION LATENCY" (1.82 ms). NB: **AP = 0.87** (không phải ~0.99), thêm vào khi trích.
> - **FedMon:** "94% precision, 91% recall, F1 0.92, cutting bandwidth by 60%" ✓; xác nhận ngôn ngữ
>   **non-IID clusters** (đúng gap ta dùng) ✓.
> - **Comparative tools:** "All tools detect attacks with full accuracy without false positives"
>   → **100% DR / 0% FPR** ✓ (lưu ý: trên **bộ test nhỏ, có kiểm soát** — không phải vận hành quy mô lớn).
> - **Multi-class deception:** "F1 0.95, detection accuracy up to 91%" ✓ (cũng có DT macro-F1 0.969 cho 28-class).
> - **LIGHT-HIDS:** "75× faster", "highest F1 across both datasets" ✓ (F1 0.973/0.766).
> - **SafeBPF:** "up to 4% overhead", "0%–7% overhead", "less than 685 ns" ✓.
| Nguồn | Precision | Recall | F1 | AUC | FPR | Latency/MTTD | Overhead |
|---|---|---|---|---|---|---|---|
| **DeSFAM** | 94% (attack 94 / normal 97) | 90% (attack) | 0.92 attack / 0.97 normal (acc 96%) | **0.94** | **1.6%** | **1.82 ms** phát hiện; enforce 0.54 µs | <1% (CPU +1.74%, RAM −6.1%) |
| **Comparative (Falco/Tetragon/Tracee)** | — | TPR **100%** mọi tool | — | — | **0%** | MTTD ~0.42–1.62 s (theo tool/tấn công) | Tetragon 6.5→6.9 mcore; Falco ~432 mcore |
| **FedMon** | 94% | 91% | 0.92 | — | — | gần baseline | −60% bandwidth liên-cluster |
| **Multi-class + deception** | 0.94 | 0.97 | 0.95 | — | ~5.5% (4.8–6.5%) | 0.8–1.2 s | CPU 39–55%; RAM 480–896 MB |
| **LIGHT-HIDS** | — | — | 0.973 (brute-force) / 0.766 (SQLi) | — | — | 0.004–0.293 s inference | nhanh hơn tới **75×** CNN-RNN |
| **SafeBPF** (cô lập, không phải detector) | — | — | — | — | — | <685 ns/lần kiểm tra | 0–4% (web), 0–7% (Netperf) |
| **Transformers/LLM survey** | trích ngoài: F1 96.9 (FlowTransformer); acc 99.5–99.75% (ViT); FPR 0.22% (CNN-GRU) | | | | | nhấn live khó do latency | |

> Lưu ý so với repo: DeSFAM gốc công bố **headline tổng hợp** (P94/R90/AUC0.94/FPR1.6%/1.82ms). KHÔNG
> coi đây là mục tiêu per-model cho bản tái lập của ta (CLAUDE.md). Bản tái lập repo: ensemble test
> AUC 0.835 / recall ~0.26 — khác xa headline → đúng "khoảng cách dataset↔vận hành".

## PART B — Đóng góp phòng thủ được (SẮC HƠN "descriptive eval" ban đầu)

NotebookLM đề xuất một hướng **mạnh và bảo vệ được**: thay vì chỉ "đo DeSFAM phát hiện privesc
multi-tenant" (DeSFAM/FedMon đã claim), hãy **đo tính dễ tổn thương của NGƯỠNG động (EMA) dưới
nhiễu noisy-neighbor** — một góc đối kháng (adversarial) mà cả hai chưa thử.

**Đóng góp đề xuất:** Định lượng **độ suy giảm phát hiện của DeSFAM** khi bị **noisy-neighbor**
(pod đồng-trú bơm tải benign hỗn loạn: cấp phát memory chồng lấn, CPU stress, I/O syscall tần suất
cao). Đo **recall↓ / FPR↑**, **tách theo loại tấn công**: privesc "ồn" (setuid, mount) vs escape
"lén" (nsenter, ptrace) ẩn trong nhiễu benign.

**Điểm phân biệt với DeSFAM/FedMon:**
1. **Nhiễu đối kháng thay vì benchmark chuẩn:** DeSFAM/FedMon huấn luyện+đánh giá trên workload
   "đại diện chuẩn" (wrk/sysbench/ApacheBench; nginx/redis) — **không** test resource-contention thù
   địch/hỗn loạn (vốn làm lệch tần suất syscall, đặc trưng thời gian, context-switch trên shared kernel).
2. **Phơi bày lỗ hổng ngưỡng động (EMA):** DeSFAM dùng **EMA** để tự nâng biên bất thường nhằm dập
   false-positive do "benign drift". Đề tài kiểm: **noisy-neighbor có thể bị dùng làm chiến thuật né
   tránh** — bơm syscall benign hỗn loạn → **đẩy EMA threshold lên** → escape lén "chui" dưới ngưỡng
   bị nâng. (Khớp repo: live detector có `EMA_TMIN` — xem CLAUDE.md.)
3. **Hồ sơ suy giảm theo từng kỹ thuật:** DeSFAM gộp metric "Attack" vs "Normal"; đề tài profile
   **CVE/kỹ thuật cụ thể nào dễ bị nhiễu che nhất** (vd Dirty Pipe CVE-2022-0847 vs RunC escape
   CVE-2019-5736) → bổ sung độ chi tiết còn thiếu cho đánh giá eBPF-anomaly.

**Giới hạn của nguồn mà đề tài lấp (trích chính lời các tác giả):**
- **DeSFAM tự nhận:** "accuracy nhạy với tính đại diện của dữ liệu huấn luyện"; future work cần
  "workload-aware profiling / container role classification" để khái quát hoá; hiện **lọc theo chuỗi
  KHÔNG soi đối số (without argument inspection)**. → Đề tài cung cấp **bằng chứng thực nghiệm** cho
  chính các giới hạn này (nếu noisy-neighbor che được escape lén → biện minh nhu cầu argument-level).
- **FedMon tự nhận:** đánh giá chỉ trên cụm kind "co-located trên một host"; thích ứng **non-IID**
  (workload tenant dị thể, biến động độc lập) là **vấn đề chưa giải**. → Đo suy giảm dưới multi-tenant
  nhiễu **trực tiếp chạm gap non-IID** này.

## ✅ Xác nhận cơ chế EMA trong PDF DeSFAM (nền tảng của RQ2 — KHÔNG phải NotebookLM bịa)
Trích nguyên văn DeSFAM (mục "DYNAMIC THRESHOLDING: ADAPTIVE ANOMALY"): *"employs dynamic thresholding
based on an **Exponential Moving Average (EMA)**, rather than a fixed cutoff. The initial threshold T0
is set to the **99.5th percentile of VAE reconstruction errors on benign validation data**... **Suppresses
false positives from benign behavioral drift**."* → đúng cơ chế RQ2 nhắm tới; giả thuyết "noisy-neighbor
đẩy EMA để né tránh" có cơ sở trong chính thiết kế DeSFAM. (T0 = p99.5 — khớp `01` nhánh ngưỡng cố định.)

## Hệ quả cho tài liệu yêu cầu (cân nhắc nâng cấp)
- Hướng này **mạnh hơn** "descriptive eval, no baseline" đã chốt ở `00_requirements.md`. Nếu chấp nhận,
  **RQ2 nâng cấp thành câu hỏi đối kháng**: *"Noisy-neighbor có làm suy giảm/né tránh được DeSFAM
  (đẩy EMA threshold) không, và mức độ theo từng loại privesc/escape?"* — có **giả thuyết bác bỏ được**
  + biến độc lập rõ (cường độ nhiễu) → đạt rubric "Experiment/Data Quality" & "Critical Thinking" cao hơn.
- Cần cập nhật `01_hypothesis.md` (RQ2), `03_threat_model.md` (thêm noisy-neighbor như biến), `07_attack_suite.md`
  (kịch bản escape lén + workload nhiễu) nếu chốt hướng này.

---

# Q3 — Cơ chế DeSFAM (EMA + ensemble + đặc trưng) ✅ VERIFY verbatim trong PDF (2026-06-18)

> Tất cả công thức/tham số dưới đây **trích nguyên văn `desfam_2025.pdf`** (đã `pdftotext` xác nhận),
> không phải NotebookLM suy diễn. Đây là **spec để dựng thí nghiệm né tránh RQ2**.

## Công thức (verbatim)
- **Ngưỡng động EMA** (cập nhật **sau mỗi cửa sổ syscall** `W_i`):
  `T_{t+1} = max(T_min, β·T_t + (1−β)·A(W_i))`, với **β = 0.9** ("controls update sensitivity"),
  **T_min = 0.5** ("enforces a lower bound"). Theo dõi **chính điểm ensemble** `A(W_i)` (không phải percentile).
- **Cờ phát hiện:** `Flag(W_i) = 1 nếu A(W_i) > T`.
- **Ngưỡng khởi tạo:** `T0 = percentile(E_val, 99.5)` — **99.5th percentile của lỗi tái tạo VAE trên
  benign validation** (Algorithm 2).
- **Điểm ensemble:** `A(W_i) = 0.7·A_VAE + 0.3·A_iForest` (**α = 0.7**, validated qua sensitivity analysis
  α∈{0,0.25,0.5,0.7,0.85,1.0}, đỉnh tại 0.7). **Chuẩn hoá A_VAE/A_iForest trước khi cộng: KHÔNG nêu trong paper.**
- **Cửa sổ:** length **15**, stride **3** (Algorithm 2 còn thử {5,10,15}).
- **Đặc trưng/cửa sổ (theo paper):** (1) Categorical Frequencies (syscall→nhóm chức năng), (2) Temporal
  deltas (mean/std/max khoảng thời gian giữa syscall), (3) PrefixSpan benign-pattern matching.
- **Cap trên / giới hạn ngưỡng tăng: KHÔNG nêu** (chỉ có sàn `T_min`). **Danh sách syscall sensor-alphabet
  chính xác: KHÔNG nêu trong paper** (paper chỉ mô tả tokenization qua `unistd64.h`).

## Hệ quả thiết kế thí nghiệm né tránh (then chốt)
1. **EMA β=0.9 ⇒ thích ứng nhanh:** mỗi cửa sổ kéo `T` 10% về phía `A(W_i)`. Đạt ~63% bước nhảy sau
   ~**10 cửa sổ**, ~90% sau ~**22 cửa sổ** (`0.9^n`). Stride 3 → cửa sổ trượt mỗi 3 syscall ⇒ noisy-neighbor
   phải **duy trì nhiễu liên tục** đủ ~vài chục cửa sổ *trước* khi tấn công lén để đẩy `T` lên.
2. **KHÔNG có cap trên ⇒ lỗ hổng khai thác được:** điểm benign cao kéo dài đẩy `T` lên **không chặn**;
   chỉ có sàn `T_min=0.5`. Đây chính là điểm yếu RQ2 nhắm.
3. **Cửa sổ né tránh:** escape lén thành công khi `T0 < A(attack) < T_inflated`. Nếu `A(attack) < T0` →
   bỏ sót *không phải* do EMA (không tính evasion-via-EMA); nếu `> T_inflated` → vẫn bị bắt. Thí nghiệm đo
   tỷ lệ cửa sổ tấn công rơi vào **dải khai thác** `(T0, T_inflated)`.
4. **Giả định ngưỡng GLOBAL:** paper cập nhật `T` "sau mỗi cửa sổ" (một ngưỡng chung cho luồng) ⇒ nhiễu
   từ pod lân cận kéo `T` chung lên, che cửa sổ tấn công. **Cần xác minh khi implement** detector chấm
   ngưỡng *toàn cục* hay *per-container*; nếu per-container thì vector yếu hơn (ghi rõ là giả định).

## ⚠️ Lệch giữa PAPER và bản tái lập trong repo (phải khai báo trung thực trong báo cáo)
- **Đặc trưng:** paper = categorical-freq + temporal + PrefixSpan; **repo = 43-dim** (freq8+disc15+stats8+
  cat10+prefixspan2), **temporal=0** vì DongTing (`strace`) không có timestamp per-syscall (CLAUDE.md).
- **Sensor alphabet 23 syscall** là **lựa chọn kỹ thuật của repo** (CLAUDE.md), **paper KHÔNG liệt kê**;
  đừng nhầm với `S_blocked` (danh sách syscall chặn theo CVE — mục đích *enforcement*, không phải feature).
- **Thang điểm/ngưỡng:** paper `T_min=0.5`, `T0=p99.5`; repo dùng chuẩn hoá khác (ensemble threshold 0.81
  trong `results.json`; live `T_0=0.25`, `EMA_TMIN=0.20` trong `.env.uat`). → Khi dựng thí nghiệm né tránh,
  dùng **tham số EMA THỰC của repo** (`EMA_TMIN`…), trích **công thức chuẩn** của paper.

## Truy vấn tiếp theo (đề xuất)
- Q4: các nguồn định nghĩa/đo "detection latency" và "FPR" thế nào? (thống nhất metric cho báo cáo).
- Q5 (verify khi code): detector chấm ngưỡng EMA **toàn cục** hay **per-container/namespace**? (quyết định độ mạnh vector né tránh).
