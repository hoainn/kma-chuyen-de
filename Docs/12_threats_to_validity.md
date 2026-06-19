# 12 — Threats to Validity (Các mối đe doạ đến tính hợp lệ)

> Tài liệu phương pháp luận kèm theo báo cáo. Nội dung này trước nằm trong §6 của
> báo cáo học thuật; được tách ra đây để báo cáo gọn, vẫn giữ đầy đủ phần tự-phản-biện.
> Số liệu theo nhánh VAE (TensorFlow/Keras): AUC 0.832; evasion 74.2% (noise cao, stealthy);
> Spearman ρ=−0.74; dạng cập nhật có điều kiện (Algorithm 2) = 0% evasion.

## Construct validity (tính hợp lệ cấu trúc)

- **(a) Scorer là bản tái hiện độc lập.** VAE đúng theo kiến trúc DeSFAM (Gaussian encoder +
  reparameterization + KL), cài bằng TensorFlow/Keras, nhưng *không* dùng trọng số gốc (công
  trình gốc không phát hành). Do đó các giá trị **tuyệt đối** (T_op, độ lớn inflation 74%) phụ
  thuộc bản cài đặt; **chiều hướng và ý nghĩa thống kê** mới là điều được khẳng định.
- **(b) DongTing ≠ TTP privesc thật.** DongTing là dữ liệu kernel-fuzzing; "loud/stealthy" là các
  *dải anomaly score*, không phải kỹ thuật privilege-escalation cụ thể. RQ1 do đó đo trên *proxy*,
  không phải privesc/K8s trực tuyến.

## Circularity của mô hình nhiễu (mối lo quan trọng nhất)

"Cường độ nhiễu" được định nghĩa bằng các cửa sổ benign lấy theo **phân vị anomaly score** tăng
dần. Vì EMA bám theo anomaly score, một phần hiệu ứng inflation có thể *do chính cách định nghĩa
nhiễu* chứ chưa hẳn do tải co-tenant thực. Cần một định nghĩa nhiễu **độc lập** (theo loại/khối
lượng workload, rồi *đo* phân phối score sinh ra) để khẳng định mạnh hơn → hướng nghiên cứu tiếp.

## Fidelity với hệ cơ sở — ĐÃ GIẢI QUYẾT

Trước đây để mở: "liệu DeSFAM thực có cập nhật vô điều kiện hay đã có guard?". **Đã đối chiếu trực
tiếp Algorithm 2 của DeSFAM:** phép cập nhật nằm trong nhánh `else` (chỉ khi `A ≤ T`) → DeSFAM
đặc tả **cập nhật có điều kiện**. Chỉ phần lời quanh công thức (3) ("cập nhật sau mỗi cửa sổ") đọc
ra vô điều kiện. Đây không còn là một "threat" mà trở thành **phát hiện trung tâm** của báo cáo
(sự thiếu nhất quán Eq.(3) ↔ Algorithm 2). Rủi ro còn lại: các *bản triển khai* đi theo nghĩa đen
công thức (3) thay vì Algorithm 2 — chính nhóm đối tượng mà báo cáo cảnh báo.

## Internal validity (tính hợp lệ nội tại)

- CAL ∩ TEST = ∅ (chống rò rỉ calibration).
- Giả định **ngưỡng toàn cục**; nếu ngưỡng theo từng container thì hiệu ứng khu trú.
- Chưa đo **FPR của chính noisy neighbor**. *(Đo bổ sung: ở noise cao ~43.5% cửa sổ của hàng xóm
  vượt T_op — tức kẻ tấn công gây nhiễu mạnh sẽ tự tạo nhiều cảnh báo; đây là một tín hiệu phòng
  thủ và là giới hạn về tính lén của vector tấn công.)*
- Chưa đối chứng cập nhật có điều kiện với các phương án đơn giản khác (cap trên, median-EMA).
- Thứ tự dòng (warm-up → noise → attack) là một trừu tượng hoá của co-tenant scheduling thực.

## External validity (tính hợp lệ ngoại tại)

Đây là **thí nghiệm cơ chế** trên anomaly score DongTing; xác thực online trên cụm K8s + Tetragon
với privilege escalation thực là công việc tiếp theo. Kết quả khái quát cho **lớp cơ chế ngưỡng
thích nghi cập nhật vô điều kiện**, không tự động cho mọi cấu hình triển khai.
