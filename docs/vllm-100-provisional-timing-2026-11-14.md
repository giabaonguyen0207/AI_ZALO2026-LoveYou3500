# Thời gian vLLM100 câu — số tạm từ output Colab ngày14/11/2026

Trạng thái: **chẩn đoán/tạm tính, không phải benchmark đã hoàn tất**. Nguồn là cell output chủ dự án cung cấp; lượt `recovery-20261114-114601-004845`, A10040GB, generator FP16. Log có đủ100 chỉ số từ1→100,100 ID riêng biệt và100 dòng ĐÃ LƯU. Chủ dự án cũng đã kiểm100 bản ghi trên Drive; thiếu `benchmark_metrics.json` vì ngắt lúc engine đang đóng.

Cập nhật sau khi chủ dự án cung cấp folder kết quả đầy đủ: đã đọc raw timing100 câu và chấm gold cho cả ba cấu hình. Tổng elapsed vLLM chính xác hơn là1476,9915s, còn span từ câu1 đến hết câu100 là1479,8632s. Xem [báo cáo mới](comparison-100-review-2026-11-14.md); bảng dưới giữ nguyên số tạm lấy từ cell để bảo toàn lịch sử nguồn.

| Đại lượng | Giây | Phút:giây | Cách xác định |
|---|---:|---|---|
| Khởi tạo đến model sẵn sàng | 391,1 | 6:31,1 | Dòng Model đã sẵn sàng; gồm xác minh/load/khởi tạo/compile liên quan trong application |
| Tổng thời gian xử lý100 câu | 1476,8 | 24:36,8 | Cộng100 giá trị elapsed đã in; có Agent, SQL/tools, nhiều model call |
| Trung bình mỗi câu | 14,768 | — | 1476,8/100; không phải thời gian mỗi token/model call |
| Khởi tạo + tổng elapsed câu, tạm tính | 1867,9 | 31:07,9 | 391,1+1476,8; chưa tính khoảng trống/I/O giữa các câu và shutdown |
| Cửa sổ sẵn sàng → bắt đầu đóng engine | khoảng1480 | khoảng24:40 | Timestamp11:52:32→12:17:12, độ phân giải giây |
| Khởi tạo + cửa sổ theo timestamp | khoảng1871,1 | khoảng31:11,1 | Ước lượng đến lúc bắt đầu shutdown; không phải thời gian process kết thúc |
| Torch baseline, số tham khảo trước đó | 3362,798657995 | 56:02,8 | process_timing.json người dùng đã gửi; còn measurement warnings |

Có281 model call theo100 dòng tiến độ. Phần thời gian từng câu có độ chính xác hiển thị0,1s; tổng có sai số làm tròn tích lũy (tối đa khoảng5s nếu mỗi dòng làm tròn gần nhất), ngoài chi phí chưa nằm trong elapsed. Chênh lệch khoảng3,2s giữa cửa sổ timestamp và tổng elapsed có thể chứa runtime-info/I/O/khoảng trống và sai số timestamp; chưa phân rã bằng raw telemetry.

Các phase sau là **thành phần nằm trong startup, không cộng lại**: đọc weights9,17s; model loading14,381355s; torch.compile35,78s; engine init276,95s. Docker vẫn cần tính startup, kernel warmup/index và đóng kết quả trong ngân sách chạy.

Không so trực tiếp tổng elapsed1476,8s với process wall của Torch để gọi là speedup: hai số có mốc đo khác nhau. Chưa xác nhận cùng ID/điều kiện từ file identity; chưa có raw timing baseline để tách startup/inference; baseline còn cảnh báo. Không tạo file benchmark_metrics.json giả hoặc sửa gate để coi lượt bị ngắt là complete. Chưa có kết luận chất lượng nếu chưa chấm gold.

Artifact local, không qua Git: `outputs/runs/Bảo/vllm-100-cell-timing-20261114-114601/` gồm bản sao output nguyên byte,100 dòng timing JSONL và `timing_provisional.json`. SHA256 output: `b343f3b8def203afb3ef1b047c888c9d111d0349d8a7a2b6989c04f252c892ad`.

Công thức có thể kiểm lại bằng cách cộng trường displayed_duration_seconds trong JSONL. Tài liệu này lưu kết quả tạm, không thay báo cáo so sánh hoàn chỉnh. Xem [nhật ký lỗi](runtime-issues-2026-11-14.md).
