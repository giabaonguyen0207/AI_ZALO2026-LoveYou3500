# Đọc log từng bước của agent

Yêu cầu được làm rõ ngày **03/11/2026**: cần xem câu trả lời nguyên văn của LLM, SQL mà LLM sinh và kết quả/lỗi ở từng bước để debug. Không cần bật chế độ thinking để ghi những thông tin thực thi này.

## Mở ngay log của lượt inference lần 2

Đã xuất từ 72 JSON log người dùng cung cấp, không gọi model hoặc đọc lại dữ liệu doanh nghiệp:

- Danh sách: `outputs/diagnosis-2026-11-03/gemma4-12b-stress72-pipeline-v2-01-readable-logs/index.md`.
- Ví dụ có SQL do LLM sinh và lỗi sửa đi sửa lại: `q_9be15002fb02.md` trong cùng thư mục.
- Ví dụ LLM chọn metric và công cụ sinh SQL: `q_1cc70c12aa27.md`.
- Bản ZIP cùng tên thư mục, thêm đuôi `.zip`, chứa index và 72 file Markdown.

JSON nguồn, ZIP chẩn đoán gốc và submission của lượt cũ được giữ nguyên. Các bản Markdown chỉ trình bày lại những gì đã ghi; không tạo thêm bước, kết quả hay lý do cho model.

## Nội dung mỗi bước

1. **LLM trả về — nguyên văn:** toàn bộ `trace[].raw`, kể cả JSON hỏng, token đặc biệt và nội dung thừa gây lỗi parse.
2. **Action LLM sinh:** object đã parse trước khi validator bổ sung tham số. Log mới lưu riêng `llm_action`; `action` tiếp tục là action được xử lý. Với log cũ, trình xuất thử parse raw để lấy action gốc; raw không hợp lệ được giữ nguyên và không suy đoán action.
3. **SQL do LLM sinh:** chỉ hiện khi action gốc là `sql`. SQL được tách khỏi chuỗi JSON để đọc được dấu nháy và xuống dòng; parameters giữ riêng.
4. **SQL do công cụ metric tạo:** hiện khi LLM gọi `metric` và kết quả công cụ có SQL. Đây là SQL của chương trình, không được ghi nhãn như SQL model tự viết.
5. **Kết quả công cụ / phản hồi gửi lại LLM:** observation, gồm columns, rows, query ID, tham số, lỗi và hướng sửa đã ghi trong trace.
6. **Đáp án cuối và thông tin sinh:** đáp án được chấp nhận, thời gian, token và lý do dừng khi có trong log nguồn.

Chỉ những dòng công cụ thực sự trả về trong log mới có thể hiển thị. Nếu query đã bị giới hạn số dòng, trạng thái `truncated` và số dòng trả về vẫn nằm trong observation; bản Markdown không khôi phục các dòng chưa được ghi.

## Các lượt chạy mới

- Batch CLI/Docker ghi thêm file `.md` cạnh mỗi JSON trong `logs/`; tên file ở batch chuẩn vẫn theo hash ID như trước.
- Batch chẩn đoán Colab ghi `logs/<id>.json` và `logs/<id>.md`, kèm trong `diagnostic_bundle.zip`.
- Notebook inference đã đồng bộ ô chẩn đoán với mã mới; các ô chọn/nạp model được giữ nguyên. Gói `outputs/datamate-colab-source.zip` cần được thay trên Drive và kernel khởi động lại khi dùng code mới. Chọn tên lượt chạy mới vì source hash đã đổi; không resume vào lượt cũ.
- Không thay model, prompt, chế độ thinking, rules tính đáp án hoặc submission schema trong thay đổi logging này.

Xuất lại một lượt có sẵn mà không inference:

```powershell
.\.venv\Scripts\python.exe tools/export_debug_logs.py --run-dir 'outputs/duong-dan-luot-chay' --zip
```

Mặc định xuất vào `readable_logs/` bên trong thư mục lượt chạy. Dùng `--output-dir` để chọn nơi khác; thư mục JSON nguồn không được dùng làm thư mục xuất.

## Kiểm chứng

- Đã chạy **54 kiểm thử CPU** liên quan agent, batch, chẩn đoán, notebook bootstrap và pipeline nghiệp vụ: tất cả qua.
- Đã xuất và kiểm tra 72 log thực tế. Đây là thay đổi ghi/đọc log, chưa chạy inference GPU mới và không thiết lập điểm benchmark mới.
- Báo cáo phân tích lần 2 lưu kết quả kiểm tra source tại thời điểm **trước** thay đổi logging này. Source hash của lượt inference mới sẽ khác; điều đó không sửa hay làm thay đổi kết quả lượt cũ.
