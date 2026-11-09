# Đánh giá dữ liệu và ngân sách Colab — 31/10/2026

Đây là đề xuất cập nhật theo ảnh và dữ liệu người dùng vừa cung cấp; chưa chọn model chính thức, chưa chạy model hoặc huấn luyện trên Colab.

## GPU và units

Ảnh cho thấy Colab Pro, 400 compute units, khoảng 10,05 units/giờ, GPU RAM 95,6 GB, system RAM 176,9 GB, disk 235,7 GB (đã dùng 42,6 GB). Ở mức tiêu hao này, 400 units tương đương khoảng **39,8 giờ runtime được cấp tài nguyên**, gồm cả thời gian chuẩn bị và nhàn rỗi, không phải 39,8 giờ huấn luyện thuần.

Người dùng gọi accelerator là G4. Google Cloud [liệt kê G4 là RTX PRO 6000 Blackwell 96GB](https://docs.cloud.google.com/compute/docs/gpus); dung lượng trong ảnh phù hợp với loại này. Ảnh chưa hiển thị tên GPU, vì vậy đây vẫn là suy luận, cần xác nhận bằng:

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

Không coi 39,8 giờ là một phiên liên tục được bảo đảm. Colab có giới hạn phiên và tài nguyên thay đổi: [Colab FAQ](https://research.google.com/colaboratory/faq.html). Cần lưu checkpoint để tiếp tục khi phiên dừng.

## Kết quả kiểm tra dữ liệu thực tế

Chạy `tools/inspect_dataset.py` bằng Python 3.13.5. Script chỉ đọc dataset, tạo `docs/dataset-audit.json` và `docs/rulebook-diff.txt`.

| Thuộc tính | Train | Public test |
|---|---:|---:|
| Đường dẫn | `train/train` | `public_test/public_test` |
| Doanh nghiệp | 30 | 14 |
| Câu hỏi | 3.000 | 1.000 |
| Đáp án chuẩn trong gói | 3.000 | Không có `answers.jsonl` |
| CSV trong thư mục data | 20 | 10 |
| XLSX trong thư mục data | 107 | 69 |
| SQLite trong thư mục data | 1 | 1 |
| Sheet XLSX phát hiện | 302 | 136 |
| Dung lượng thư mục data, MB thập phân | 193,5 | 108,9 |

Mỗi doanh nghiệp có README. Train không có ID câu hỏi hoặc đáp án trùng; hai tập ID khớp nhau. Câu hỏi có bốn trường `id`, `business_id`, `question`, `as_of_date`. Đáp án có `id`, `type`, `answer`, `alias`; không có SQL, code, tool calls hoặc lời giải trung gian được gán nhãn.

Phân bố đáp án train: number 1.884; percent 578; string 88; list 193; unanswerable 70; date 145; table 19; boolean 23. Number và percent chiếm **82,07%**.

Đã kiểm kê file, đếm CSV/SQLite và xem cấu trúc/số sheet XLSX, các tóm tắt README, một số README chi tiết, câu hỏi/đáp án mẫu và diff rulebook. Chưa kiểm tra toàn bộ giá trị/ô Excel, chưa giải câu hỏi và chưa đo accuracy. `dimension` trong audit chỉ là vùng dùng của sheet, không phải số bản ghi nghiệp vụ.

Các dạng dữ liệu quan sát từ README: bảng phẳng; header/detail nối theo mã; nối bằng tên khi thiếu mã; sheet theo tháng; file theo năm/tháng; sổ tay không mã đơn/khách; nhiều nguồn dữ liệu cùng một doanh nghiệp. Ví dụ `biz_123` có 17 bảng SQLite. Những đặc điểm này ưu tiên cơ chế đọc schema, đọc README và thực thi truy vấn.

## Rulebook thực tế làm rõ hoặc bổ sung ngữ cảnh cũ

- Lãi gộp dùng **giá vốn ghi trên dòng bán**, không tự tính lại từ phiếu nhập/lô. Phần mô tả phương pháp giá vốn trong bản tổng hợp cũ không đủ để triển khai phép tính này.
- Tiền có số lẻ làm tròn về đồng; phần trăm và trung bình khác làm tròn hai chữ số, trừ quy định riêng.
- Có quy tắc điền thông tin đơn cho các dòng sau trong nhóm và loại dòng xuất trùng của cùng một đơn; không được xóa trùng mọi bảng một cách chung chung.
- Public có quy tắc bổ sung: khách trung thành theo số ngày mua; khách sộp theo trung bình đơn với ít nhất ba đơn; nhà cung cấp chính theo số phiếu; các ngành mẹ và bé, sửa xe, điện thoại; trạng thái có thể nằm trong ghi chú.

Đây là thông tin để đọc đúng split ở inference, không phải nguồn để sinh training samples từ public test. Chỉ dùng train và tài nguyên được phép để huấn luyện.

## Baseline đề xuất

1. **B0: Qwen3.5-4B chưa fine-tune + SQL/Python**, kiểm tra ingestion, rulebook, ngày tháng và output.
2. **B1: Qwen3.5-9B chưa fine-tune + cùng bộ công cụ**, so sánh trên doanh nghiệp train được giữ riêng.
3. **B2 ưu tiên: Qwen3.5-9B + BF16 LoRA**, rank khởi điểm 16–32, train cách kiểm tra schema, gọi công cụ, sửa truy vấn và xuất JSON. Dùng quantization khi xuất inference chỉ sau khi đo chất lượng.
4. Gemma 4 12B hoặc Ministral 3 14B là thử nghiệm sau nếu B1/B2 chưa đủ; mỗi phương án dùng riêng một checkpoint.

Tất cả model vẫn phải có trọng số công khai trước **01/10/2026**, pin revision và kiểm kê tổng tham số <=15B. Xem nghiên cứu ngày 30/10 trong `baselines-and-costs.md`; tài nguyên GPU lớn hơn không thay đổi giới hạn inference của cuộc thi.

Với ~96GB VRAM, BF16 LoRA 9B là lựa chọn đầu tiên hợp lý vì chưa cần nén base model xuống 4 bit để tiết kiệm bộ nhớ. QLoRA vẫn hữu ích khi tăng context/batch hoặc cần tiết kiệm VRAM; không bảo đảm nhanh hơn LoRA. [Hướng dẫn Unsloth](https://unsloth.ai/docs/models/qwen3.5/fine-tune) hỗ trợ LoRA cho Qwen3.5, nhưng số VRAM trong ví dụ không bảo đảm mọi cấu hình đều fit.

## Full fine-tuning so với LoRA

- Full: cập nhật toàn bộ tham số; bộ nhớ gradient và optimizer lớn.
- LoRA: giữ base model BF16 cố định, chỉ cập nhật adapter.
- QLoRA: giữ base model nén 4 bit, chỉ cập nhật adapter; giảm VRAM nhưng có chi phí giải lượng tử.

[Hugging Face](https://huggingface.co/docs/transformers/v4.50.0/model_memory_anatomy) nêu quy tắc ước lượng **18 byte/tham số cộng activations** cho một cấu hình mixed precision + AdamW điển hình. Đây không phải định luật áp dụng cho mọi optimizer.

| Checkpoint | Bộ nhớ trạng thái theo 18 byte/tham số, GB thập phân, chưa activations | Full tiêu chuẩn trên ~96GB |
|---|---:|---|
| Qwen3.5-4B, toàn checkpoint ~4,66B | ~84GB | Có thể thử, khá sát bộ nhớ; cần cấu hình nhỏ và đo peak VRAM |
| Qwen3.5-9B, toàn checkpoint ~9,65B | ~174GB | Không fit cấu hình này |
| Gemma 4 12B ~11,95B | ~215GB | Không fit cấu hình này |
| Ministral 3 14B ~13,9B | ~250GB | Không fit cấu hình này |

Optimizer 8 bit, BF16 states hoặc CPU offload có thể thay đổi khả năng fit, nhưng cần cấu hình và benchmark riêng. Không diễn giải bảng thành kết luận rằng full 9B là bất khả thi với mọi kỹ thuật. Không nên mặc định RAM hệ thống cộng vào VRAM mà không có chi phí truyền dữ liệu.

## Thời gian ước tính để lập ngân sách

Chưa có dữ liệu train agent đã token hóa và chưa đo training throughput trên GPU người dùng. Các khoảng dưới đây là **giả định lập kế hoạch**, không phải benchmark mới thay thế các ước tính thuê GPU trước đây.

Giả sử giữ riêng khoảng năm doanh nghiệp train cho validation, tạo khoảng 2.500 mẫu hợp lệ, trung bình 4.096 token/mẫu, hai epoch: **20,48 triệu processed tokens**. Độ dài 4K là giả định; context/schema/tool trace thực tế có thể dài hơn nhiều.

| Phương án | Giả định aggregate training tokens/s | Tổng giờ, đã thêm khoảng 1h overhead | Units tại 10,05/h |
|---|---:|---:|---:|
| 4B BF16 LoRA | 1.000–2.500 | ~3–7h | ~30–70 |
| 9B BF16 LoRA | 500–1.200 | ~6–13h | ~60–131 |
| 9B QLoRA | 350–900 | ~8–18h | ~80–181 |
| 12B BF16 LoRA | 350–900 | ~8–18h | ~80–181 |
| 14B BF16 LoRA | 250–700 | ~10–24h | ~101–241 |
| 4B full, chỉ khi cấu hình fit | 300–800 | ~9–21h | ~90–211 |
| 9B/12B/14B full AdamW tiêu chuẩn | Không fit theo giả định trên | Không có thời gian chạy hợp lệ để báo | Không khởi chạy nguyên cấu hình này |

Throughput phụ thuộc kernel Blackwell, optimizer, batch, gradient checkpointing và sequence length. LoRA cũng phải chạy forward/backward qua mạng base; ít tham số trainable không có nghĩa thời gian giảm theo đúng tỷ lệ tham số.

Các khoảng trên không gồm chuẩn bị lời giải có kiểm chứng, sinh dữ liệu, evaluation dùng generation, cài đặt gặp lỗi kéo dài hoặc thời gian idle. Full với offload không dùng chung throughput với full nằm hoàn toàn trên GPU.

Chạy thử 50–100 bước sau warm-up trên cấu hình thật, đo thời gian step, peak VRAM và số token thực tế. Khi đó dùng:

```text
training_hours = processed_tokens / measured_tokens_per_second / 3600
units = (training_hours + setup_and_save_hours) × actual_units_per_hour
```

Ngân sách đề xuất: 20–30 units cho kiểm tra môi trường và đo tốc độ; 40–60 cho baseline/evaluation ban đầu; khoảng 100–140 cho một LoRA 9B; giữ ít nhất 100 cho validation, kiểm tra export và chạy lại. Điều chỉnh sau đo, không khởi chạy tất cả phương án trong bảng.

## Việc còn thiếu

- Tên GPU và phiên bản CUDA/PyTorch thực tế trên Colab.
- Chốt checkpoint có bằng chứng công khai trước cutoff.
- Đo baseline trên holdout theo doanh nghiệp.
- Tạo và kiểm tra tool trajectories từ train; chưa có trong gói.
- Đếm token sau template, rồi đo throughput/memory để chốt epoch và ngân sách.
- Docker guideline vẫn chưa được cung cấp trong workspace lúc kiểm tra này.
