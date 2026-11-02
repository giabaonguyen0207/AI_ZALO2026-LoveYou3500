# Kiểm tra Qwen 3.5 4B local — 01/11/2026

## Nguồn và mục tiêu

Người dùng báo sau bước B: Qwen 9B 4bit không load được, Transformers yêu cầu xử lý các module phân bổ sang CPU/disk. Đây là báo cáo lỗi của người dùng; không phải benchmark 9B.

Bổ sung `qwen35-4b-local-4bit.json` để kiểm tra luồng schema → SQL → typed answer trên laptop. Qwen 9B/Gemma vẫn là các lựa chọn đánh giá trên GPU đủ bộ nhớ; không coi 4B đã thắng hoặc thay thế model thi chính thức.

## Artifact và môi trường đã kiểm tra

- `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, commit 02/03/2026 00:52:52 UTC, Apache 2.0.
- Metadata công khai xác nhận tree có 2 shard safetensors trước cutoff. Tổng tham số đầy đủ **4.659.865.088**.
- Đã tải vào `models/qwen35-4b` trên E:, đếm tensor khớp inventory, seal 14 file và kiểm tra manifest local. Hai shard có SHA256 khớp metadata LFS đã lưu.
- RTX 4060 Laptop, 8.188 MiB VRAM; Windows, RAM tổng khoảng 16GB. Snapshot lúc chẩn đoán: RAM vật lý trống khoảng 3GB, không phải dung lượng được bảo đảm lâu dài.
- PyTorch `2.11.0+cu128`, CUDA 12.8, Transformers `5.15.0`, bitsandbytes `0.50.2`, Accelerate `1.15.0`.
- Phép thử matmul CUDA nhỏ và Linear4bit nhỏ đều chạy qua. Không cài lại/downgrade stack của người dùng trong lần kiểm tra này.

## Smoke test đã chạy thật

Câu thử trên `train/train/data/biz_101`, `as_of_date=2026-06-15`:

> Dữ liệu bán hàng trong file ban_hang.csv có tất cả bao nhiêu dòng? Đếm COUNT(*) toàn bảng, không lọc ngày hay trạng thái và không loại trùng.

Agent gọi `SELECT COUNT(*) FROM raw_0_ban_hang`, sau đó xuất `number: 56501`. Kết quả khớp COUNT(*) được chạy độc lập trên chính bảng ingest. Không đọc `answers.jsonl` để trả lời và không dùng public/private test.

| Lượt | Load model | Câu hỏi | Gọi model | Kết quả |
|---|---:|---:|---:|---|
| Standard 4bit | 24,280s | 186,389s | 2 | Khớp COUNT(*) |
| Standard 4bit, thử dọn cache rỗi sau load | 24,128s | 184,687s | 2 | Khớp COUNT(*) |

File thực tế: `outputs/qwen35-4b-local-smoke.json`, `outputs/qwen35-4b-local-smoke-cache-check.json`. Một lượt ban đầu chạy qua stdin đã dừng trước khi hoàn thành; chưa xác định nguyên nhân. Các lượt chạy file Python trực tiếp trong bảng hoàn tất với exit code 0.

Sau load, PyTorch báo bộ nhớ allocated khoảng 3.149 MiB. Khi sinh đáp án, đỉnh allocated khoảng 12.033 MiB, cao hơn VRAM vật lý. Đây là số đo allocator PyTorch trên Windows, không phải bằng chứng toàn bộ quá trình nằm trong VRAM 8GB. Chưa xác định hoàn toàn cơ chế paging/attention gây đỉnh đó. Không diễn giải con số sau load thành lời hứa model sẽ fit mọi context trên 8GB.

Thử thêm `prefill_chunk_size=512` qua cơ chế generate của Transformers 5.15: gặp lỗi kích thước tensor/cache trước khi có câu trả lời. Đã lưu ở `outputs/qwen35-4b-local-smoke-chunked.json`; **không bật tùy chọn này trong preset/code triển khai**. Việc dọn cache rỗi cũng không làm giảm thời gian đáng kể ở hai lượt trên.

## Giới hạn và cách dùng

- Đây là kiểm tra một câu đếm dòng, không phải đo accuracy/PA hay khả năng áp rulebook trên câu hỏi kinh doanh.
- Không có số liệu chất lượng hay speedup cho 9B/Gemma/assistant, chưa fine-tune.
- Local 4B dùng được để xem trace và phát triển luồng agent, nhưng thời gian thử còn chậm. Colab/GPU đủ bộ nhớ phù hợp hơn để kiểm tra thường xuyên các model lớn hoặc train.
- Tăng context, số bước hoặc output có thể làm tốn thêm tài nguyên; chưa xác minh mọi doanh nghiệp trên local.
- Đã bổ sung thông báo tiếng Việt cho lỗi GPU placement/OOM và chạy 36 kiểm thử qua.
- Có notebook `DataMate_Colab_Inference.ipynb` chỉ hỏi một câu mỗi lần. Đã kiểm tra cú pháp cell; chưa chạy trên Colab trong phiên này.

Nguồn checkpoint: [card chính thức](https://huggingface.co/Qwen/Qwen3.5-4B), [history](https://huggingface.co/Qwen/Qwen3.5-4B/commits/main), và `model-evidence-2026-11-01.json`. Hướng dẫn offload tham khảo [Hugging Face bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes#offloading); chưa tự bật offload CPU/disk cho 9B.
