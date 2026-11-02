# Qwen 3.5 4B/9B, Gemma 4 12B và assistant

Kiểm tra nguồn ngày **01/11/2026**. Metadata và lịch sử file từ repository chính thức được lưu ở `model-evidence-2026-11-01.json`. Sau lỗi load 9B do người dùng báo trên máy 8GB, bổ sung 4B làm lựa chọn kiểm tra agent local. Chưa có benchmark so sánh 9B/Gemma/assistant hoặc kết quả fine-tune.

| Cấu hình | Tổng tham số theo metadata checkpoint | Vai trò | Preset |
|---|---:|---|---|
| Qwen 3.5 4B riêng | 4.659.865.088 | Kiểm tra luồng agent trên laptop, nén 4bit khi load | `qwen35-4b-local-4bit.json` |
| Qwen 3.5 9B riêng | 9.653.104.368 | Baseline agent một model | `qwen35-9b.json` |
| Gemma 4 12B IT riêng | 11.959.730.224 | Baseline agent khác họ model | `gemma4-12b.json` |
| Gemma 4 12B IT + MTP assistant | 12.382.587.188 | Tăng tốc decoding có model chính kiểm tra | `gemma4-12b-speculative.json` |
| Gemma gọi lại chính nó để review | 11.959.730.224 | Kiểm tra đáp án bằng một lượt suy luận khác | `gemma4-12b-self-review.json` |
| Qwen 9B + Gemma 12B | 21.612.834.592 | Vượt 15B, không được dùng chung khi inference thi | Không hỗ trợ |

Quantization không giảm số tham số được khai báo. Merge một adapter LoRA vào base giữ nguyên kiến trúc/count của model xuất; các adapter chưa merge hoặc model bổ sung phải được kiểm kê lại.

## Assistant bạn chỉ định hoạt động ra sao

`google/gemma-4-12B-it-assistant` là **MTP drafter**, khoảng 0,423B tham số, ghép với Gemma 4 12B IT. Nó tận dụng hidden states/KV cache của model chính để đề xuất nhiều token; model chính xác minh token trong speculative decoding. Đây không phải model trả lời câu hỏi độc lập hoặc giám khảo kiểm tra nghiệp vụ. Xem [model card chính thức](https://huggingface.co/google/gemma-4-12B-it-assistant) và [hướng dẫn Transformers cho Gemma Unified Assistant](https://huggingface.co/docs/transformers/model_doc/gemma4_unified_assistant).

Tài liệu nhà phát hành mô tả khả năng tăng tốc và bảo toàn chất lượng của thuật toán. Mức tăng tốc trên RTX 4060, Colab hoặc RTX 5090 **chưa được đo tại dự án này**. Kết quả thực tế phụ thuộc GPU, độ dài prompt/output, bộ nhớ, tỷ lệ token được chấp nhận, quantization và runtime. Prompt dài, output SQL/JSON ngắn có thể khiến phần prefill chiếm nhiều thời gian nên drafter không giúp nhiều, hoặc tổng thời gian còn tăng; đây là giả thuyết cần đo.

Mục tiêu của MTP là giảm thời gian để thu được output của model chính. Chất lượng truy vấn sai do model chính hiểu nhầm rulebook không được drafter sửa. `self_review` có mục đích khác: thêm lượt kiểm tra, có thể tăng hoặc giảm accuracy và thường tốn thêm thời gian. Không cộng hai lần tham số nếu cùng một object/checkpoint được gọi lại.

Backend dùng native `assistant_model=` trong `generate`, batch size 1, greedy và cùng tokenizer của target. Không ghép Gemma assistant với Qwen. Với Qwen chưa có drafter tương thích đã xác minh/triển khai trong repository này. Sau fine-tune Gemma, assistant gốc có thể dự đoán kém phù hợp hơn; cần đo lại tỷ lệ chấp nhận/tốc độ và tính nhất quán output.

## Khóa checkpoint trước hạn

| Model | Revision đầy đủ | Thời điểm commit đã chọn (UTC) |
|---|---|---|
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 02/03/2026 00:52:52 |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 02/03/2026 00:51:43 |
| `google/gemma-4-12B-it` | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | 20/07/2026 16:42:08 |
| `google/gemma-4-12B-it-assistant` | `46d4c6f13f0ac0ad827b915669b8df9b81c64c51` | 20/08/2026 21:49:20 |

Đã kiểm tra tree ở từng revision có safetensors và metadata LFS SHA256, không chỉ ngày ra tên model. Bốn nguồn ghi Apache 2.0. Evidence JSON là nguồn repository công khai; manifest trọng số local được tạo riêng khi `stage-model` thực sự tải xong. `stage-model` tải đúng revision, đếm mọi tensor safetensors, tạo SHA256 từng file; inference kiểm tra lại manifest. Không dùng `main` vì Gemma target có cập nhật tokenizer ngày 30/10 và assistant có cập nhật generation config ngày 26/10. Không tự đổi sang các cập nhật sau cutoff.

Nguồn: [Qwen 4B card](https://huggingface.co/Qwen/Qwen3.5-4B), [Qwen 4B history](https://huggingface.co/Qwen/Qwen3.5-4B/commits/main), [Qwen 9B card](https://huggingface.co/Qwen/Qwen3.5-9B), [Qwen 9B history](https://huggingface.co/Qwen/Qwen3.5-9B/commits/main), [Gemma target history](https://huggingface.co/google/gemma-4-12B-it/commits/main), [assistant history](https://huggingface.co/google/gemma-4-12B-it-assistant/commits/main). Chi tiết file/hash nằm trong evidence JSON đi kèm.

## So sánh công bằng

Tách train/validation theo **doanh nghiệp**, giữ nguyên tập câu hỏi validation giữa các cấu hình. Không dùng public/private để sinh dữ liệu fine-tune. Lần đầu so Qwen single với Gemma single dưới cùng ngân sách bước/token; tiếp theo so Gemma single với Gemma speculative cùng target weights, prompt và kiểu dữ liệu. Review là thí nghiệm riêng.

```powershell
.\.venv\Scripts\python.exe -m datamate compare --configs configs/qwen35-9b.json configs/gemma4-12b.json configs/gemma4-12b-speculative.json --ids outputs/training/format/validation_ids.json --references train/train/answers.jsonl --output outputs/comparison-01
```

Mỗi cấu hình chạy trong tiến trình riêng để giải phóng VRAM; báo PA, coverage, p50/p95 thời gian từng câu và wall time có load. Local evaluator không phải bộ chấm chính thức. Nên chạy thêm lặp lại, đo peak VRAM/token throughput/acceptance nếu tối ưu speculative sâu hơn; harness hiện chưa có telemetry token-level hoặc phép thử token-for-token cho native generation.

Chỉ ưu tiên assistant khi **đo được** tổng wall time giảm, không làm PA/độ ổn định xấu đi, và vẫn fit bộ nhớ. Để đánh giá khả năng giữ output, đối chiếu từng đáp án single/speculative cùng checkpoint và lưu log; không suy ra từ một speed claim trên model card.

## Máy local và Colab

Máy hiện tại báo NVIDIA RTX 4060 Laptop, **8.188 MiB VRAM**. Trọng số BF16 theo count thô đã khoảng 19,3GB với Qwen và 23,9GB với Gemma, chưa có KV cache/activations. Preset Qwen 4bit là tùy chọn để thử trên local, chưa được chứng minh fit/tốc độ. Gemma 4bit cộng assistant BF16 còn chật hơn; không mặc định bật assistant trên 8GB.

Colab dùng notebook chọn model và LoRA/QLoRA. Không huấn luyện drafter Gemma như một model trả lời độc lập. Với GPU ~96GB như ngữ cảnh trước đây, LoRA BF16 là phương án để thử; phải đo tên GPU, peak VRAM và thời gian 50 bước trên runtime thật trước khi chạy dài.
