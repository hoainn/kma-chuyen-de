# Báo cáo Chuyên đề — Tái lập DeSFAM (tiếng Việt)

Mã nguồn LaTeX cho báo cáo chuyên đề: *Tái lập và mở rộng DeSFAM — phát hiện bất
thường lời gọi hệ thống cho container bằng eBPF/học máy, kiểm chứng trực tiếp qua
Tetragon/Kubernetes*.

**Học viên:** Nguyễn Ngọc Hoài — CHAT4P006.
**Định dạng:** báo cáo học thuật một cột, tiếng Việt (class `article`, XeLaTeX).

## Biên dịch (Docker — không cài LaTeX trên máy)
```bash
./build.sh          # XeLaTeX qua image texlive/texlive -> main.pdf
```
Engine XeLaTeX bắt buộc (dùng `fontspec` cho tiếng Việt; phông Latin Modern phủ
đầy đủ dấu). BibTeX chạy tự động trong `build.sh`.

## Cấu trúc
```
main.tex                  # tiền tố + khung tài liệu (\input các mục)
references.bib            # thư mục tham khảo (kế thừa từ paper/ + Tetragon/Cilium)
build.sh                  # biên dịch bằng Docker (XeLaTeX + BibTeX, 3 lượt)
sections/
  abstract.tex            # Tóm tắt
  01-introduction.tex     # Giới thiệu — câu hỏi NC (CH1–CH3), nghiên cứu tái lập
  02-related-work.tex     # Nền tảng + công trình liên quan
  03-methodology.tex      # Phương pháp tái lập — dữ liệu, đặc trưng 43-D, mô hình
  04-experiment-setup.tex # Thiết lập Docker/Kubernetes/Tetragon
  05-evaluation.tex       # Kết quả: tái lập ngoại tuyến + phân tách trực tiếp
  06-discussion.tex       # Thảo luận + đe dọa tính hợp lệ
  07-conclusion.tex       # Kết luận
  08-appendix-repro.tex   # Phụ lục tái lập (lệnh Docker)
figures/
  desfam-architecture.png # sao chép từ DeSFAM/diagram/
  feature-engineering.png # sao chép từ DeSFAM/diagram/
```

## Nguồn dữ liệu/số liệu
- Số liệu ngoại tuyến (Bảng 1–2) lấy trực tiếp từ `DeSFAM/model/results.json`.
- Số liệu trực tiếp (phân tách điểm) theo cấu hình `DeSFAM/inference/.env.uat`
  (`T_0 = 0.25`) và ghi chú trong `kma-chuyen-de/CLAUDE.md`.
- Báo cáo trình bày **trung thực** điểm tái lập được và điểm lệch so với bài báo
  gốc DeSFAM (IEEE Access, 2025).

## Ghi chú
- Báo cáo là **nghiên cứu tái lập** (không đề xuất phương pháp mới); mọi tuyên bố
  bám theo kết quả thực tế của dự án, không sao chép số liệu công bố như của mình.
- Hành văn khách quan ngôi thứ ba; ≥20 tài liệu tham khảo (2016–2025).
