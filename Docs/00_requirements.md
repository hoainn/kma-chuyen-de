# Yêu cầu đề tài — Phát hiện leo thang đặc quyền trong Multi-tenant Kubernetes qua System Calls

> Nguồn: *Danh sách đề tài làm project môn PP NCKH trong ATTT — Lớp CHAT4P (KMA)*.
> TLTK chính của cả môn: **Edgar, T.W.; Manz, D.O. — Research Methods for Cyber Security, Syngress 2017**.
> Thời lượng: **4 tuần** · Độ khó: **Cao** · Mỗi đề tài **1 học viên**.
> Cơ sở (base) của đề tài: **DeSFAM** — khung phát hiện bất thường runtime trên cloud
> container bằng **eBPF (Tetragon) + AI** (VAE + Isolation Forest) trên chuỗi *system call*.
> Bộ tái lập DeSFAM đã có sẵn trong repo (`DeSFAM/`, dataset DongTing, detector live qua Tetragon).

## 1. Đề tài & loại hình
**"Nghiên cứu cơ chế phát hiện tấn công leo thang đặc quyền trong môi trường Multi-tenant
Kubernetes dựa trên phân tích System Calls"**

Loại hình: **Applied Experimentation** — tái lập cơ chế DeSFAM rồi (1) **đặc tả** năng lực phát hiện
privilege escalation / container escape theo từng loại tấn công (RQ1), và (2) **thực nghiệm đối kháng
có kiểm soát** (RQ2): đo độ suy giảm/né tránh phát hiện khi một **tenant lân cận bơm nhiễu**
(noisy-neighbor). **Biến độc lập = cường độ nhiễu noisy-neighbor**; biến phụ thuộc = recall/FPR/độ trễ
+ **trạng thái ngưỡng động EMA** + **tỷ lệ né tránh thành công**. Đây là thiết kế có **biến độc lập
thao tác được + giả thuyết bác bỏ được** (đúng Edgar & Manz Ch.9/11), không chỉ đo mô tả.

> 💡 **Định vị novelty (grounded từ NotebookLM — `10_related_work_grounded.md`):** DeSFAM & FedMon
> **đã tuyên bố** phát hiện live, unsupervised, multi-tenant — nên đóng góp KHÔNG phải "dùng DeSFAM
> phát hiện privesc multi-tenant". Đóng góp **sắc và phòng thủ được**: phơi bày **lỗ hổng ngưỡng động
> EMA** của DeSFAM dưới nhiễu noisy-neighbor (nhiễu benign hỗn loạn **đẩy EMA threshold lên** → escape
> "lén" chui dưới ngưỡng), **profile theo từng kỹ thuật** (loud setuid/mount vs stealthy nsenter/ptrace).
> Chạm trực tiếp giới hạn **tác giả DeSFAM tự nhận** ("nhạy với dữ liệu huấn luyện", "không soi đối
> số — without argument inspection") và gap **non-IID** của FedMon. DeSFAM & FedMon chỉ test workload
> "chuẩn" (wrk/sysbench/nginx/redis), **chưa** test resource-contention thù địch → đây là khoảng trống.

## 2. Câu hỏi nghiên cứu
- **RQ1.** Cơ chế phát hiện bất thường trên system call của DeSFAM (VAE+iForest, cửa sổ 15/stride 3)
  phát hiện tấn công **leo thang đặc quyền** (TA0004) và **container escape** trong K8s với **độ phủ
  (recall), độ chính xác (precision), tỷ lệ báo động giả (FPR) và độ trễ phát hiện (detection
  latency)** là bao nhiêu?
- **RQ2 (đối kháng — đóng góp chính).** Một **tenant lân cận bơm nhiễu** (noisy-neighbor: cấp phát
  memory hỗn loạn, CPU stress, I/O syscall benign tần suất cao) có **làm suy giảm và/hoặc giúp NÉ TRÁNH**
  DeSFAM không — cụ thể bằng cách **đẩy ngưỡng động EMA** lên để tấn công "lén" chui dưới ngưỡng? Định
  lượng **recall↓ / FPR↑ / tỷ lệ né tránh** theo **cường độ nhiễu** và **theo từng kỹ thuật** (loud:
  setuid/mount vs stealthy: nsenter/ptrace).

## 3. Ràng buộc phạm vi (bắt buộc)
- Giới hạn ở **một cụm Kubernetes thực nghiệm cô lập** (kind/minikube hoặc cluster lab), namespace
  `demo`; **không** thao tác trên cụm production hoặc tài sản bên thứ ba.
- Nguồn tín hiệu **duy nhất** là *system call* thu qua eBPF (Tetragon TracingPolicy) — **không** dùng
  log ứng dụng/network như tín hiệu phát hiện (để cô lập đóng góp của phân tích syscall).
- **Cơ chế phát hiện cố định** = DeSFAM đã huấn luyện (normal-only, sensor-alphabet 23 syscall). Mọi
  khác biệt kết quả phải quy về **nhân tố quan sát** (loại tấn công, mức multi-tenant), không phải do
  đổi mô hình giữa chừng.
- Tấn công chạy bằng **workload kịch bản hoá** (manifest pod) — không có chủ thể người, gói trong 4 tuần.

## 4. Chỉ số đo lường (phải ghi nhận)
- **Recall / TPR** trên cửa sổ tấn công leo thang đặc quyền; **Precision**; **FPR** trên workload lành tính.
- **Detection latency**: khoảng thời gian từ syscall đặc trưng đầu tiên của hành vi leo thang đến lúc
  điểm bất thường vượt ngưỡng (`session/window` đầu tiên bị alert).
- **AUC-ROC / Average Precision** trên tập cửa sổ benign vs attack (đánh giá độc lập ngưỡng).
- **(RQ2) Giá trị ngưỡng động EMA theo thời gian** — ghi `T_EMA(t)` để chứng minh nhiễu noisy-neighbor
  có đẩy ngưỡng lên hay không; **Evasion success rate** = tỷ lệ cửa sổ tấn công bị bỏ sót *do* ngưỡng bị
  nâng (so điều kiện không nhiễu).
- **Phân tích theo loại tấn công** (per-scenario) **× theo cường độ nhiễu** (per-noise-level).
- Báo cáo **phân phối điểm bất thường** (không chỉ trung bình) để thể hiện phương sai trung thực.

## 5. Mối đe doạ tính hợp lệ (chỉ đích danh)
- **Lớn nhất (External):** DongTing là syscall sinh từ **kernel fuzzing/syzbot**, **không** phải TTP
  leo thang đặc quyền do người vận hành → mô hình huấn luyện trên phân phối "bất thường kernel" có thể
  **không khái quát** sang privesc thực tế. → bắt buộc thảo luận; giảm thiểu bằng đánh giá live trên 7
  kịch bản tấn công K8s thật (manifest) thay vì chỉ test-set DongTing.
- **Base-rate / FPR:** trong vận hành, tỷ lệ window lành tính áp đảo → FPR nhỏ vẫn có thể tạo "biển
  cảnh báo". Phải báo cáo FPR tuyệt đối + số alert/giờ, không chỉ tỷ lệ.
- **Construct:** "điểm bất thường cao" ≠ "leo thang đặc quyền" — có thể chỉ là workload hiếm gặp lành
  tính. Củng cố bằng phân tích định tính syscall trace của các true/false positive.
- **(RQ2) Nhân quả né tránh:** recall giảm khi có nhiễu có thực sự do **EMA bị đẩy lên** không, hay do
  nhiễu lấn át đặc trưng cửa sổ? → phải **ghi `T_EMA(t)`** và so điều kiện ngưỡng cố định vs EMA để tách
  hai cơ chế (nếu recall giảm cả khi ngưỡng cố định thì không phải lỗ hổng EMA).

## 6. Công cụ & TLTK chốt
- Nền tảng: **Kubernetes + Tetragon (eBPF)**, **DeSFAM** (VAE+iForest, repo này), **DongTing dataset**,
  Prometheus/Grafana/Loki cho quan sát.
- TLTK chủ chốt: **DeSFAM (Zehra et al., IEEE Access 2025, DOI 10.1109/ACCESS.2025.3592192)**;
  **eBPF-PATROL (2025)** — phát hiện privesc/container escape qua syscall; **DongTing dataset**;
  nền tảng phương pháp **VAE (Kingma & Welling 2013)**, **Isolation Forest (Liu et al. 2008)**.
  (Corpus đầy đủ: `09_literature_corpus.md`.)

## 7. Kế hoạch 4 tuần (xem `week_plan.md`)
| Tuần | Nội dung |
|---|---|
| W1 | Lược khảo phát hiện bất thường syscall; **dựng cụm K8s + Tetragon**; tái lập DeSFAM (train + load model). |
| W2 | Triển khai **bộ kịch bản tấn công leo thang đặc quyền** (manifest); cấu hình detector live + logging điểm bất thường. |
| W3 | Chạy nhiều trial trên **đơn tenant** và **multi-tenant (noisy-neighbor)**; thu điểm bất thường + nhãn cửa sổ. |
| W4 | Phân tích recall/FPR/latency/AUC theo nhân tố; tư duy đối kháng (né tránh); hoàn thiện báo cáo + phụ lục tái lập. |

---

# PHỤ LỤC C — Quy chuẩn báo cáo & Khung đánh giá (BẮT BUỘC — dùng chung cả môn)

## A. Định dạng & cấu trúc
- **IEEE 2 cột**, dài **6–8 trang**.
- Các mục bắt buộc: **Abstract · Background & Related Work · Methodology (bắt buộc *biện minh lựa chọn
  phương pháp*) · Experiment Setup · Evaluation · Discussion & Threats to Validity · Conclusion**.

## B. Trình bày dữ liệu
- **CẤM pie-chart.** Phải dùng **bảng Markdown / Box-plot / CDF / Histogram / ROC** để thể hiện
  **phương sai** trung thực (vd. phân phối điểm bất thường benign vs attack).

## C. Trích dẫn
- **≥ 10 tài liệu khoa học chất lượng cao, 2020–2026**, lồng ghép logic, **phong cách IEEE**
  (ngoại lệ: trích VAE 2013 / Isolation Forest 2008 / Edgar & Manz 2017 làm nền tảng phương pháp).

## D. Điểm sáng tạo bắt buộc
- **"Phụ lục Tái lập kết quả" (Reproducibility Appendix)** — gói trong **1 trang**, chứa **dòng lệnh
  tự động** (deploy Tetragon → train/load model → chạy attack manifest → xuất điểm bất thường), để
  giảng viên xác thực **trong ≤ 15 phút**.

## E. Văn phong
- **Ngôi thứ ba khách quan**, định dạng IEEE, biểu đồ trung thực, **bỏ ngôn ngữ cường điệu/tiếp thị**.

## F. Rubric (100 điểm — chấm *tư duy lý luận bảo vệ bằng chứng*, KHÔNG chấm "chạy trơn tru")
| Tiêu chí | Điểm | Yêu cầu mức Xuất sắc |
|---|---:|---|
| **Scientific Rigor & Methodology** | 20 | Tường minh loại hình nghiên cứu (**tham chiếu trực tiếp Edgar & Manz**), khoanh vùng triệt để biến phụ thuộc/nhân tố, liệt kê đầy đủ validity threats nội/ngoại tại. |
| **Experiment/Data Quality** | 20 | **Nhiều trial**, báo cáo **phương sai/CI**; phân tích theo nhân tố (loại tấn công × tenancy); FPR tuyệt đối + base-rate; thay suy luận trung bình cảm tính bằng phân phối/kiểm định. |
| **Critical Thinking** | 15 | Không lặp kết luận cũ; **tấn công lại chính giả định nền tảng** (mô hình normal-only né tránh được không? DongLing ≠ privesc thật?). |
| **Interpretation & Limitations** | 15 | Phân tích khách quan; thảo luận **lỗ hổng của chính bài kiểm thử** (dataset, base-rate, n kịch bản). |
| **Reproducibility** | 15 | Công khai **toàn bộ mã, siêu tham số, manifest, ngưỡng**; nhà nghiên cứu độc lập khôi phục y hệt. |
| **Writing Quality** | 15 | Ngôi thứ ba, IEEE chuẩn, biểu đồ trung thực, bỏ hype. |

---

## Đối chiếu nhanh với báo cáo (mục tiêu — `paper/` / `ReportV2/`)
| Yêu cầu | Trạng thái mục tiêu |
|---|---|
| IEEE 2 cột, 6–8 trang | ⏳ dựng bản PP-NCKH (IEEEtran) từ `paper/` |
| Đủ 7 mục bắt buộc | ⏳ Abstract→Conclusion |
| Methodology biện minh chọn Applied Experimentation (Edgar & Manz) | ⏳ §III trích Ch.9/11/12 |
| Recall/Precision/FPR/Latency/AUC theo nhân tố | ⏳ DV chính (xem `04`) |
| RQ2 đối kháng: noisy-neighbor đẩy EMA → né tránh (recall↓/evasion theo cường độ nhiễu × kỹ thuật) | ⏳ biến độc lập = cường độ nhiễu; log `T_EMA(t)` |
| Threats to Validity (DongTing ≠ privesc thật; base-rate) | ⏳ §IV + §VI |
| CẤM pie-chart, dùng CDF/Box-plot/ROC | ⏳ phân phối điểm bất thường + ROC |
| ≥10 paper 2020–2026, IEEE | ⏳ corpus `09` (8 paper SOTA + nền tảng) |
| Reproducibility Appendix ≤1 trang, ≤15 phút | ⏳ deploy.sh + detect + manifest |
| Văn phong khách quan, bỏ hype | ⏳ |
| **Lưu ý lệch:** đề bài nhấn "leo thang đặc quyền"; 7 manifest hiện tại phủ TA0002/05/06/07/08/09/40, **chưa có TA0004 thuần** | ⚠️ Bổ sung kịch bản privesc thuần (setuid/capset/ptrace/container-escape) — xem `07_attack_suite.md`. |
