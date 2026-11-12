# Validation100 bằng L4 23 GB

Ngày 10/11/2026, chủ dự án yêu cầu đổi bộ đối chứng sang L4; đã từng chạy baseline Qwen3.5-9B trên L4. Chuẩn bị bản **L4 v1.1** riêng, giữ nguyên notebook/source v1 A100 và dữ liệu. Chưa chạy full merge hoặc inference GPU mới ở máy local.

## File cần dùng

- Notebook: `notebooks/training/DataMate_Colab_Qwen35_SQL_Validation100_L4_v1_1.ipynb`.
- Source mới: `outputs/packages/datamate-qwen35-sql-validation100-source-l4-v1.1.zip`.
- Data giữ nguyên: `outputs/packages/qwen35-sql-validation100-data-v1.zip`.

Chép source mới vào **MyDrive/DataMateSQLSFT/**; data ZIP cũ nếu đã upload thì giữ nguyên. Mở notebook mới, chọn **L4 và High RAM**, chạy các ô theo thứ tự. Không sửa notebook training hoặc đổi hash thủ công. Adapter full và base trên Drive được tái sử dụng tại các đường dẫn của v1. Kết quả mới ở `DataMateSQLSFT/runs/qwen35-9b-sql-v1.3-validation100-l4-01/` để không trộn với lượt BF16 v1.

## Thay đổi

Cả baseline lẫn fine-tune dùng **4-bit NF4, double quantization, BF16 compute** theo backend hiện có. Giữ context16384, sinh1536 token, mode single/code_only, seed42, cùng100 IDs và cùng ngân sách. Hai model nạp lần lượt trong hai process. Không giảm dữ liệu, context hay số câu; không thêm nhãn vào prompt. Chưa có số đo VRAM thực tế cho lượt này trên L4, nên không tuyên bố đã kiểm vừa mọi câu.

Nếu model merge đã tồn tại và khớp lineage/hash, tái sử dụng ngay. Nếu chưa có, notebook gọi **CPU merge trong process riêng trên Colab**, ẩn CUDA chỉ trong process đó. Base BF16 và adapter được merge, seal, lưu trên Drive rồi process kết thúc. Các lượt inference sau đó vẫn nhìn thấy L4 và mới nạp model ở 4-bit. Không cần thuê/chuyển sang A100 để merge.

CPU merge cần chính sách tối thiểu **32 GiB RAM hệ thống còn trống**, có kiểm thêm hạn mức container khi đọc được. Đây là RAM thường, không phải VRAM. Chọn High RAM; nếu thiếu, ô merge dừng trước khi nạp trọng số và báo lượng RAM cần. Ngưỡng này có margin nhưng không phải cam kết peak RAM cho mọi môi trường. Merge CPU có thể lâu hơn GPU, chỉ cần làm một lần. Nếu đã có bản merge hợp lệ thì bỏ qua yêu cầu32GiB này.

Model merge trên Drive vẫn là BF16 (khoảng20GB); lượng tử hóa4-bit diễn ra khi nạp inference. SSD `/content` vẫn cần chỗ cho cả base và model merge, khoảng40GB cộng phần trống. Hai bản merge bằng CPU/GPU có thể khác đôi chút do làm tròn số; artifact thực tế luôn được seal/hash, không trộn kết quả giữa artifact khác nhau.

## Chạy và khôi phục

Chạy baseline100 → fine-tune100 → report. Bộ100 và nhãn giống hệt v1. Scorer và49 file runtime giữ nguyên; chỉ thay orchestration/profile và thêm launcher CPU. Các record hoàn tất được lưu trên Drive như v1. Mất runtime: mở kernel mới, cùng GPU L4, cùng RUN_NAME/source/data/model/cấu hình rồi chạy lại; không đổi sang tag BF16 cũ để resume. Xem [hướng dẫn đối chứng chung](sql-validation100-colab.md) cho ý nghĩa PA, giới hạn validation và các file kết quả.

Sau khi xong, lấy `review/comparison.json`, `paired_review.json`, `regressions.json` hoặc ZIP báo cáo trong thư mục kết quả. Chưa có điểm baseline/fine-tune mới cho đến khi notebook chạy thật trên Colab.
