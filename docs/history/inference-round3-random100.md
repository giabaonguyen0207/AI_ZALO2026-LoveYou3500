# Inference lần 3 — 100 câu ngẫu nhiên

**Yêu cầu ngày 03/11/2026:** chọn 100 câu ngẫu nhiên, không chọn theo tiêu chí loại đáp án, ngôn ngữ hoặc độ khó. Phạm vi lấy mẫu lần này là **toàn bộ 3.000 câu train**; không giới hạn trong 24 doanh nghiệp development của hai lượt trước.

## Danh sách đã chuẩn bị

Trong `outputs/inference-round3-random100/`:

- `questions.jsonl`: đúng 100 câu, mỗi dòng chỉ có `id`, `business_id`, `question`, `as_of_date`; không chứa nhãn hoặc đáp án.
- `question_ids.json`: đúng 100 ID để dùng khi chạy bằng CLI hoặc đối chiếu.
- `selection_manifest.json`: thuật toán, seed, hash nguồn câu hỏi, hash mẫu và thứ tự ID đã chọn.
- `verification.json`: kiểm tra số lượng, tính duy nhất và đối chiếu với cách chọn trong notebook.

Cách bốc là `random.Random(20261103).sample(...)` trên danh sách toàn bộ câu hỏi sắp theo ID để có thứ tự đầu vào ổn định. Lấy mẫu đều, không hoàn lại; mọi câu có cùng xác suất được chọn. Không dùng đáp án/type, nội dung câu, độ khó, quota doanh nghiệp, lỗi cũ hoặc kết quả inference để chọn mẫu. Seed dùng lặp lại danh sách, không thay seed của model.

**Quan sát sau khi chọn, không dùng để chọn lại:** mẫu có 25 doanh nghiệp, 1 câu trùng mẫu 72 cũ (`q_27eebcccdf1f`) và 18 câu thuộc các doanh nghiệp holdout trước đây. Không lọc hoặc bốc lại vì các thống kê này. Do lần này lấy cả train, kết quả không được mô tả như phép đánh giá trên một tập holdout vẫn hoàn toàn chưa được quan sát. Phân chia training/validation đã lưu trước đây không bị sửa.

## Chạy trên Colab

1. Thay `datamate-colab-source.zip` trên Drive bằng bản mới tại `outputs/datamate-colab-source.zip`.
2. Dùng `notebooks/DataMate_Colab_Inference.ipynb` đã cập nhật. Giữ model và quantization bạn đang muốn đánh giá. Khi thay source trong runtime đang dùng, khởi động lại kernel/runtime và nạp lại theo notebook.
3. Chạy ô batch cuối. Mặc định đã đặt:

```python
SAMPLE_SIZE = 100
SAMPLING_MODE = "random"
SAMPLING_SEED = 20261103
KNOWN_CASE = None
RUN_NAME_SUFFIX = "random100-round3-01"
```

Notebook tự chọn đúng danh sách đã xuất khi dùng train nguyên bản; không cần tải riêng `questions.jsonl` lên Colab. Đã đối chiếu thứ tự đủ 100 ID giữa file xuất và hàm dùng trong notebook.

Thư mục kết quả là `<model>-random100-round3-01`, ví dụ `gemma4-12b-random100-round3-01`. Giữ trần mềm 90 phút như trước; nếu lượt chạy dừng sớm, có thể chạy lại cùng ô với cùng model/config/code để tiếp tục. Không resume vào thư mục lượt 72 câu cũ.

Sau dòng REPORT, ZIP gồm submission, review, manifest và log JSON/Markdown từng câu. Log thể hiện đầu ra LLM, SQL, tham số, kết quả hoặc lỗi mỗi bước. Xem [hướng dẫn debug log](../debug-step-logs.md).

## Trạng thái kiểm chứng

- Đã xác nhận nguồn có 3.000 câu với ID duy nhất; mẫu có đúng 100 ID duy nhất và đúng bốn trường đầu vào.
- File ID, file câu hỏi, manifest và kế hoạch chọn của notebook khớp nhau; hash nội dung mẫu đã được kiểm tra.
- **12 kiểm thử CPU đã qua**, gồm lựa chọn ngẫu nhiên không truy cập type/profile/holdout, không ép câu đã biết, không cần quota doanh nghiệp và thực thi toàn bộ ô batch trên dữ liệu giả.
- Chỉ thay cách chọn mẫu/cấu hình lượt chẩn đoán. Không sửa logic trả lời, model hoặc rules nghiệp vụ trong yêu cầu này.
- **Chưa inference 100 câu bằng LLM**, chưa có điểm số hoặc thời gian chạy GPU cho lượt 3.
