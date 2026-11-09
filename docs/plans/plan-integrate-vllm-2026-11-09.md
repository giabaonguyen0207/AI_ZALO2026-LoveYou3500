# Plan tích hợp vLLM — Hải

Cập nhật **09/11/2026**. Mục tiêu: tăng tốc inference của DataMate, giữ chất lượng đáp án. Đây là kế hoạch triển khai, chưa có kết quả benchmark. Phối hợp với **Thịnh** theo [plan RAG](plan-integrate-rag-2026-11-09.md).

## 1. Kiến trúc đã chốt

Dùng **Adapter + Strategy** trong ứng dụng hiện có:

- `VLLMBackend` là Adapter bọc engine local; `TransformersBackend` vẫn dùng được.
- Strategy chọn qua `Settings.backend = transformers | vllm | demo`; `build_backend()` tạo implementation, `create_agent()` truyền vào Agent.
- Agent, preparation và schema repair gọi cùng interface, không phụ thuộc trực tiếp vLLM. Giữ logic plan, SQL, kiểm tra đáp án và output hiện tại.

Hải chuẩn hóa `ChatBackend` trong `models/contracts.py`, giữ import cũ bằng re-export. Interface cần có:

```python
class ChatBackend(Protocol):
    def generate(self, messages: list[dict], remaining_seconds: float,
                 *, max_new_tokens: int | None = None) -> str: ...
    def count_tokens(self, messages: list[dict]) -> int: ...
    def runtime_info(self) -> dict: ...
```

Giữ metadata `counts_as_model`, `supports_token_budget`, `last_generation`; cập nhật cả DemoBackend, fake backend và wrapper. `generate()` trả text assistant cho parser, `count_tokens()` dùng đúng tokenizer/chat template. Thống nhất interface với Thịnh để RAG dùng được cả hai backend.

## 2. Đánh giá baseline trước

**Hải tự chọn benchmark, số câu và công cụ chạy phù hợp; không bắt buộc bộ 200 câu hoặc cùng benchmark với Thịnh.**

1. Chọn bộ câu và cách chấm, rồi chạy/chấm baseline Transformers trước khi tích hợp. Lưu tập đã chọn và kết quả.
2. Sau tích hợp, chạy vLLM trên **đúng bộ đó**, cùng model/checkpoint, prompt, ngân sách sinh và điều kiện GPU tương đương.
3. So chất lượng đáp án và tổng thời gian, tính cả load/chuẩn bị. Nếu đổi môi trường đáng kể, đo lại baseline đối chứng.

**Đạt khi:** inference nhanh hơn và chất lượng đáp án không giảm. Không áp mức tăng tốc cố định. Ghi rõ bộ đo, số câu, cấu hình và kết quả trước/sau; không chỉ báo tokens/giây hoặc chọn lại câu có lợi sau khi xem điểm.

## 3. Công việc thực hiện

1. Kiểm tra checkpoint hiện dùng và chọn phiên bản vLLM/PyTorch/CUDA tương thích trên GPU Linux; ghi lại phiên bản đã thử.
2. Thêm `models/vllm_backend.py`. Engine local được tái dùng giữa các câu, không tải lại mỗi lượt và không gọi dịch vụ bên ngoài.
3. Giữ chat template, tokenization, chế độ thinking, EOS/turn terminator, decoding và token budget tương đương baseline. Hỗ trợ override max_new_tokens cho preparation/schema repair.
4. Tôn trọng deadline và giới hạn context. Hết giờ phải hủy request thực sự, không để GPU tiếp tục chạy ngầm; không tự cắt rulebook để vừa prompt.
5. Thêm cấu hình/factory/CLI cho backend mới; lazy import để Transformers/demo vẫn chạy khi không cài vLLM.
6. Ghi token, stop reason, thời gian, cấu hình engine vào log; đổi backend/config/model phải chặn resume vào kết quả cũ.
7. Kiểm tra cơ chế bằng fixture, chạy smoke model thật rồi đo trước/sau. Prefix caching, batching hoặc grammar có thể thử tiếp và ghi rõ tác dụng từng thay đổi.

## 4. Phạm vi file và phối hợp

| Hải phụ trách | Nội dung |
|---|---|
| `src/datamate/models/` | Contract, adapter, formatter/token counter, factory |
| `config.py`, `cli.py` | Cấu hình, validation và preset vLLM |
| Dependency, runner đóng gói/diagnostic | Phiên bản, log, resume và artifact chạy được |

Các đường dẫn trên thuộc `src/datamate/` trừ phần công cụ đóng gói. Với file chung như `config.py`, `application.py`, dependency và Docker, Hải/Thịnh ghép từng patch nhỏ để tránh ghi đè công việc nhau. Thịnh có thể phát triển module RAG và chạy baseline riêng trong lúc Hải làm backend.

Khi ghép RAG, dành VRAM cho encoder/reranker và đo lại bản kết hợp trên một bộ câu do hai người chọn. Không lấy điểm từ hai benchmark khác nhau để kết luận bản ghép tốt hơn.

## 5. Bàn giao

- Code/PR, cấu hình, dependency đã pin và lệnh chạy; chuyển được về Transformers.
- Kết quả baseline và vLLM: chất lượng, tổng thời gian, lỗi/regression và thông tin GPU/model/bộ câu.
- Kiểm tra EOS, token override, timeout, context, log/resume; chạy thử Docker offline.

Tuân thủ [ngữ cảnh dự án](../competition-context.md): mọi model inference tổng ≤15B, trọng số công khai trước 01/10/2026 và có revision/hash/license; không dùng test để training hoặc thu thập đáp án test. `/data` chỉ đọc; kết quả ở `/result/submission.jsonl`, log ở `/result/logs/`; toàn bộ private test ≤6 giờ gồm load. Không sửa scorer hoặc quyết định đáp án theo ID.

Tham khảo: [kiến trúc](../architecture.md), [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/), [API AsyncLLM](https://docs.vllm.ai/en/latest/api/vllm/v1/engine/async_llm/).

