# Đối chứng SQL SFT trên 100 câu validation

Nguồn ngày 10/11/2026: chủ dự án tải run full từ Drive về `outputs/runs/Bảo/qwen35-9b-sql-v1.3-full-01`. Chuẩn bị đánh giá trên Colab Pro, chưa chạy merge hoặc inference GPU tại máy local.

## File và đường dẫn

Chép `datamate-qwen35-sql-validation100-source-v1.zip` và `qwen35-sql-validation100-data-v1.zip` từ `outputs/packages/` vào **MyDrive/DataMateSQLSFT/**. Hai ZIP mới là source đối chứng và dữ liệu nghiệp vụ 100 câu, khác gói SFT train/validation JSONL.

Mở `notebooks/training/DataMate_Colab_Qwen35_SQL_Validation100_v1.ipynb`, chọn **A100 40 GB / High RAM** và kernel mới. Ô đầu đã đặt:

- Adapter: `MyDrive/DataMateSQLSFT/runs/qwen35-9b-sql-v1.3-full-01/adapter`.
- Base: `MyDrive/DataMateSQLSFT/models/qwen35-9b-c202236`.
- Model merge: `MyDrive/DataMateSQLSFT/models/qwen35-9b-sql-v1.3-full-01-merged`.
- Kết quả: `MyDrive/DataMateSQLSFT/runs/qwen35-9b-sql-v1.3-validation100-01/`.

Nếu Drive vẫn giữ run đúng đường dẫn training thì không cần upload lại adapter vừa tải về. Không dùng smoke_adapter, pilot, hoặc thay model_dir bằng thư mục chỉ có adapter.

## Quy trình

Chạy các ô theo thứ tự. Notebook kiểm hash source/data/adapter, giữ Torch/Triton hiện có và dùng dependency đã pin như training. Dùng base chính xác theo manifest của lượt train. Merge/seal trong process GPU riêng; tái sử dụng bản merge hợp lệ khi chạy lại. Bản merge ghi dở được đổi tên bảo toàn, không coi là model hoàn tất.

Chép model base và model merge sang SSD `/content`, xác minh rồi chạy **baseline 100 câu trước, fine-tune đúng 100 câu đó sau** trong hai process khác nhau để giải phóng VRAM. Không có baseline GPU hoặc kết quả chất lượng được giả lập trên local. Dự trù dung lượng cho hai model BF16 trên SSD (khoảng 40 GB cộng vùng trống) và một model merge trên Drive (khoảng 20 GB); Colab cần đủ RAM cho merge.

Cả hai dùng cùng pipeline v6 đóng băng từ source training v1.3, `single`, `code_only`, BF16/không quantization, context16384, sinh1536 token, tối đa8 bước, SQL timeout15s, 90s/câu, seed42; cache ingestion tách theo variant. Không áp dụng conv adapter training cho inference có KV cache. Đối chứng này đo khả năng sinh đáp án của deployment BF16 sau merge, không so trực tiếp train loss của base NF4.

Chỉ khi cả hai đã trả lời đủ 100 câu mới đọc nhãn để chấm. PA = (đúng − sai)/100; abstain được 0. Report có PA, đúng/sai/abstain, số calls, thời gian câu trung bình/p95, chi tiết cải thiện/giảm và raw trace SQL. Thời gian copy/verify model được ghi riêng; tốc độ còn chịu ảnh hưởng cache/JIT và thứ tự chạy, không mặc định là speedup thuần của model.

## Bộ 100 câu

Lấy từ đủ 630 ID validation gốc (6 doanh nghiệp tách khỏi train), không chỉ 618 câu đã nhận SFT. Phân bổ theo tỷ lệ số câu của doanh nghiệp bằng largest remainder, chọn theo SHA256(seed42:id), không dùng nhãn/độ khó/kết quả để chọn. IDs và hash cố định trong ZIP. Không có câu thuộc partition train.

Input giữ nguyên toàn bộ bảng/README của doanh nghiệp được chọn, rulebook, calendar và 100 câu. Nhãn ở `evaluation/references.jsonl`, bên ngoài `input/`, chỉ mở sau hai lượt generation. Đây là tập phát triển đã từng xem khi curate, không phải điểm public/private test hoặc holdout mù. 100 câu là phép đo tiết kiệm; chênh lệch nhỏ chưa đủ kết luận tổng quát.

## Ngắt runtime và lấy kết quả

Mỗi câu ghi record nguyên tử kèm checksum trên Drive. Khi mất runtime, mở kernel mới và chạy lại notebook với **cùng RUN_NAME, cấu hình, source/data/model và loại GPU**; câu có record hợp lệ không chạy lại. Nếu lượt đã đủ 100 câu, ô inference chỉ tái tạo summary, không load model lại. Chỉ một server cho một RUN_NAME. Đổi cấu hình hoặc GPU thì dùng RUN_NAME mới; không trộn kết quả.

Phần timing chưa ghi được khi runtime chết cứng không được dựng lại thành thời gian đo đầy đủ: `wall_seconds=null` và `timing_incomplete_after_runtime_loss=true`. Record câu đang tính dở có thể phải chạy lại. Khi hai lượt đủ, ô cuối tạo `review/comparison.json`, `paired_review.json`, `regressions.json`, trace đọc được và ZIP trong thư mục kết quả trên Drive. Giữ báo cáo đó để quyết định chọn adapter; không train thêm chỉ vì chưa có điểm generation.

Bản tải về checkpoint295 thiếu optimizer.pt; không dùng bản đó để resume training. Adapter cuối và checkpoint298 được kiểm riêng trong bước đóng gói. Giữ nguyên artifact người dùng và các release training cũ.
