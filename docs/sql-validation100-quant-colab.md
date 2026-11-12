# Chọn none, 8bit, 4bit khi đối chứng

Bản L4 v1.2 thêm QUANTIZATION ở ô đầu, mặc định "4bit". Chọn "none" (BF16), "8bit" (LLM.int8), hoặc "4bit" (NF4 double quantization). Hai model luôn cùng lựa chọn; context16K, token1536 và100 IDs giữ nguyên. Backend/scorer/runner không đổi.

Upload `datamate-qwen35-sql-validation100-source-l4-v1.2.zip` vào MyDrive/DataMateSQLSFT, tái sử dụng `qwen35-sql-validation100-data-v1.zip`. Mở `DataMate_Colab_Qwen35_SQL_Validation100_L4_v1_2.ipynb`. Adapter/base/model merge cũ giữ nguyên. Merge CPU/High RAM nếu chưa có bản merge hợp lệ, giống bản v1.1.

Thay QUANTIZATION rồi chạy lại notebook từ ô đầu. RUN_NAME tự thêm hậu tố -none / -8bit / -4bit, giúp kết quả/cấu hình/cache của từng lựa chọn tách riêng. Không gán thủ công RUN_NAME trở về thư mục của precision khác để resume. Với mỗi lựa chọn giữ nguyên source, data, model, GPU và cấu hình khi nối lại.

L4: 4bit vẫn là mặc định để tiết kiệm VRAM. 8bit/none dùng nhiều VRAM hơn; không có bảo đảm16K vừa L4 cho mọi câu. Nếu OOM, không tự chuyển precision hoặc cắt prompt; chọn precision khác từ ô đầu hoặc dùng GPU lớn hơn. So baseline/fine-tune phải cùng precision. Chưa có điểm/VRAM mới được đo trong bước chuẩn bị này.
