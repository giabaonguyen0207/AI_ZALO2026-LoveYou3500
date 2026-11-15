# Đọc log từng bước

CLI/notebook chính/Docker ghi submission.jsonl, run_identity.json, run_summary.json và logs/ vào result_dir đã chọn, thường dưới outputs/runs/<người>/<run-id>/.

Mỗi log JSON trong `logs/` có ID, câu hỏi/as_of_date, đáp án cuối, elapsed_ms, thinking nguyên văn/null, trace và prompt ban đầu; một record một dòng theo guideline BTC. Bản Markdown dễ đọc của lượt mới nằm trong `debug/`, trình bày raw output, action, SQL/parameters, tool result/lỗi và các bước chấp nhận/sửa lại. Lượt cũ có thể để Markdown cạnh JSON. SQL model sinh và SQL công cụ tạo được phân biệt; lỗi cấp run nằm ngoài folder log từng câu.

Không suy thêm rows hoặc reasoning không có trong log. Giữ trạng thái truncated nếu công cụ giới hạn kết quả. Generation tokens/stop reason chỉ xuất khi backend thực sự cung cấp.

Để xuất lại log dễ đọc mà không gọi model:

```bash
python tools/export_debug_logs.py --run-dir outputs/runs/hai/run-01 --zip
```

Các lượt cũ nằm trong archive/2026-11-09/outputs/; thay đường dẫn tương ứng. Không ghi đè raw log hoặc gói frozen. Team200 tùy chọn có diagnostic bundle/review riêng; nhãn chỉ được mở sau generation.
