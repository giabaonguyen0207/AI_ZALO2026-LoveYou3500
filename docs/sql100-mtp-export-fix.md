# Sửa xuất checkpoint merge — validation100 L4 v1.3

Nguồn ngày10/11/2026: notebook v1.2 do người dùng cập nhật, CPU merge đã ghi6shard nhưng seal báo9,409,813,744 phần tử khác inventory9,653,104,368. Đây là lỗi xử lý xuất model của bộ công cụ, không phải bằng chứng adapter train bị hỏng.

Đã đọc index và header safetensors tại revision gốc `c202236235762e1c871ad0ccb60c8ee5ba337b9a` của Qwen/Qwen3.5-9B (không tải full trọng số). Có15tensor `mtp.*` tổng243,290,624 phần tử, đúng chênh lệch. Transformers5.22.1 khai báo bỏ qua `^mtp.*` trong Qwen3_5PreTrainedModel, nên các tensor này không thuộc state_dict khi save_pretrained. Metadata đọc được ở `.cache/qwen35-merge-metadata/`.

Bản v1.3 sửa riêng CPU exporter: đối chiếu đầy đủ tên/shape tensor xuất với base đã verify; chỉ cho khôi phục tensor thiếu có prefix `mtp.` bằng giá trị/dtype gốc. Missing active tensor, extra tensor hoặc sai shape vẫn bị từ chối. Thêm shard MTP gốc và cập nhật index để loader đọc đủ. Giữ nguyên registry/count/checksum guard; không chèn trọng số giả hoặc thay số tham số. CPU/GPU inference vẫn dùng runtime/scorer cũ; MTP tiếp tục được Transformers bỏ qua giống baseline, tổng checkpoint vẫn được kiểm kê đầy đủ theo giới hạn model của dự án.

Upload source mới `datamate-qwen35-sql-validation100-source-l4-v1.3.zip` vào MyDrive/DataMateSQLSFT. Mở `DataMate_Colab_Qwen35_SQL_Validation100_L4_v1_3.ipynb` trong kernel mới; giữ data ZIP cũ. Source v1.2/notebook lỗi và artifact người dùng được giữ nguyên. Adapter đã train không đổi, không cần train lại. Model merge cũ chưa seal được đổi tên bảo toàn rồi merge lại từ base+adapter đã verify. Bản giao sửa lỗi giữ QUANTIZATION="none" như notebook người dùng cập nhật; vẫn chọn được none/8bit/4bit. None16K chưa được bảo đảm vừa L4; lỗi thiếu MTP xảy ra ở export, chưa liên quan precision inference.

CPU test dùng model Qwen3.5 nhỏ và PEFT thật: merge, khôi phục MTP, seal, nạp lại index và xác nhận active LoRA đã thay đổi, auxiliary giữ nguyên. Có test đầu ra shard/unsharded và từ chối thiếu tensor active, extra, sai shape. Không load full9B trên máy local; cần chạy lại merge/inference trên Colab để kiểm end-to-end thực tế.
