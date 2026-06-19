# 06 — Pilot run (xác minh end-to-end)

> Mục tiêu: **xác minh đường ống** train → model → detector live → điểm bất thường, KHÔNG phải kết quả
> cuối. Dữ liệu pilot lấy từ bản tái lập DeSFAM hiện có trong repo (`model/results.json` + ghi chú
> CLAUDE.md). Quy mô đầy đủ (nhiều trial × tenancy) thực hiện ở Tuần 3.

## Thiết lập pilot
- **Train**: DongTing, sensor-alphabet 23 syscall, window 15 / stride 3 → 43-dim; normal-only;
  VAE multi-seed (0,1,2) giữ seed val-AUC tốt nhất; ensemble `0.7·VAE + 0.3·iForest`.
- **Ngưỡng**: chốt từ **validation** (không chạm test) → `results.json`.
- **Live**: detector qua Tetragon trong K8s; ngưỡng vận hành `T_0=0.25` (`inference/.env.uat`).

## Kết quả tái lập (DongTing test-set — từ `model/results.json`)

| Mô hình | AUC-ROC | AP | F1 | Precision | Recall/TPR | FPR |
|---|---|---|---|---|---|---|
| Isolation Forest | 0.851 | 0.991 | 0.474 | 0.999 | 0.311 | 0.0077 |
| VAE | 0.734 | 0.985 | 0.421 | 0.999 | 0.267 | 0.0079 |
| **Ensemble** | **0.835** | **0.990** | **0.417** | **0.999** | **0.264** | **0.0076** |

- Validation: VAE val-AUC ≈ **0.944** (đã khử posterior collapse), ensemble val ≈ **0.948**.
- **Phân tách live** (qua Tetragon, ns demo): workload lành tính (mongodb) điểm **≤ 0.19**; các pod tấn
  công demo **0.278–0.930** → tách được benign/attack ở `T_0=0.25`.

## Diễn giải & bài học phương pháp luận (QUAN TRỌNG)

1. **Pipeline đã thông**: train → artifacts (`model/`) → detector live (Tetragon gRPC) → điểm bất
   thường per-window, chạy trọn vẹn; `test_feature_parity.py` xanh (train ↔ serve cùng không gian đặc trưng).
2. **Precision rất cao (0.999) nhưng recall thấp (~0.26–0.31) trên DongTing test**: mô hình **rất ít
   báo động giả** nhưng **bỏ sót nhiều** trên phân phối DongTing test. ĐÂY LÀ TÍN HIỆU then chốt cho
   đề tài: cần đo **recall trên privesc K8s thật** (live), vì DongTing test (syzbot) **có thể không** đại
   diện cho hình dạng privesc thực tế → đúng mối đe doạ external validity ở `docs/04`.
3. **Khoảng cách dataset↔live là THẬT, không phải lo xa**: AUC test DongTing 0.835 nhưng phân tách live
   (benign ≤0.19 vs attack 0.278–0.930) **rõ hơn** — gợi ý dữ liệu live K8s có cấu trúc dễ tách hơn
   syzbot. → phải báo cáo **cả hai** (DongTing test + live K8s) và **không** dùng headline DeSFAM gốc làm
   chuẩn (CLAUDE.md cảnh báo các số "paper" bị mislabel).
4. **Giới hạn pilot (sửa cho Tuần 3)**: kết quả trên là **một** bản train + đo định tính phân tách live;
   chưa có **nhiều trial**, chưa **tách per-scenario/per-tenancy**, chưa có **CI** ở mức trial, chưa đo
   **detection latency**. → Tuần 3 chạy ≥5–10 trial/(kịch bản×tenancy), loại cửa sổ warm-up, dựng ROC +
   box-plot điểm + bảng nhân tố.

## Tái lập (skeleton — hoàn thiện thành Phụ lục ≤15 phút)
```bash
# 1) Cảm biến + cluster
cd DeSFAM/Kubernetes && bash tetragon/install.sh
kubectl apply -f tetragon/tracing-policy.yaml -f tetragon/nodeport.yaml
# 2) Mô hình (dùng model/ có sẵn, hoặc train lại)
cd ../training && docker compose -f docker-compose.pipeline.yml run --rm train   # tuỳ chọn
# 3) Detector live (ngưỡng calibrate)
cd ../inference && docker compose --env-file .env.uat up -d --build
# 4) Workload: benign + tấn công
cd ../Kubernetes && kubectl apply -f normal/app.yaml
bash deploy.sh        # 7 kịch bản tấn công (+ kịch bản privesc bổ sung, xem 07)
# 5) Thu điểm bất thường → CSV → phân tích (recall/FPR/latency/AUC + box-plot/ROC)
```

## Lưu ý cấu hình (đã biết)
- Live TracingPolicy chỉ emit **23 syscall** sensor-alphabet → train PHẢI bật `SENSOR_ALPHABET=1`
  (mặc định) để cùng không gian đặc trưng; nếu không, mô hình full-syscall **suy biến** (benign ≈
  attack, `docs/auto-iter-log.md` iter-2).
- `load_syscall_table` bỏ dòng x32 ABI để `execve/ptrace` dùng đúng ID như extractor live.
- DongTing không có timestamp per-syscall (`strace -v -f`) → `temporal_T=0` khi train; **detection
  latency** đo từ timestamp **Tetragon** ở live, không từ feature temporal.
