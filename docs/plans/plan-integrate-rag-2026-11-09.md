# Plan tích hợp RAG — Thịnh

Cập nhật **09/11/2026**. Mục tiêu: cải thiện chất lượng đáp án bằng truy xuất ngữ cảnh phù hợp. Đây là kế hoạch triển khai, chưa có kết quả benchmark. Phối hợp với **Hải** theo [plan vLLM](plan-integrate-vllm-2026-11-09.md).

## 1. Kiến trúc và model đã chốt

Dùng **Adapter + Strategy**:

- Adapter riêng cho embedding và reranker; không ghép encoder vào giao thức sinh text của ChatBackend.
- Strategy chọn `rag.mode = none | dense | dense_rerank` bằng factory. `none` giữ hành vi baseline, không load encoder hoặc thêm trường prompt.
- `create_agent()` lắp và truyền retriever vào Agent. RAG dùng được với cả Transformers và vLLM.

Sử dụng đúng hai checkpoint đã kiểm tra:

| Model | Revision |
|---|---|
| AITeamVN/Vietnamese_Embedding | `dea33aa1ab339f38d66ae0a40e6c40e0a9249568` |
| AITeamVN/Vietnamese_Reranker | `f536976248403314225d7fdfdbc87f0e9516a54e` |

Tổng **1.135.510.529 tham số**. Khi tải thật, kiểm hash và seal toàn bộ artifact. Chi tiết giấy phép, ngày trọng số và ngân sách ghép model ở [báo cáo đã kiểm tra](../rag-model-review-2026-11-09.md).

## 2. Đánh giá baseline trước

**Thịnh tự chọn benchmark, số câu và công cụ chạy phù hợp; không bắt buộc bộ 200 câu hoặc cùng benchmark với Hải.**

1. Chọn bộ câu và cách chấm; chạy/chấm hệ thống chưa có RAG trước khi tích hợp.
2. Chạy bản có RAG trên **đúng bộ đó**, giữ generator/checkpoint, backend, prompt nền và ngân sách tương đương.
3. Có thể đo lần lượt `none → dense → dense_rerank` để biết embedding và reranker giúp bao nhiêu. Ghi cả chất lượng và chi phí thời gian/index/load.

**Đạt khi:** chất lượng đáp án tốt hơn baseline, với thời gian/tài nguyên đáp ứng giới hạn hệ thống. Chỉ giữ reranker nếu có lợi ích đo được. Ghi rõ bộ đo và kết quả trước/sau; retrieval score cao hơn chưa chứng minh đáp án tốt hơn, không chọn lại câu có lợi sau khi xem điểm.

## 3. Công việc thực hiện

1. Tạo package `src/datamate/retrieval/`: contract, encoder adapters, corpus/index, strategies, cấu hình và factory.
2. Thống nhất interface với Hải: retriever nhận câu hỏi, scope split/doanh nghiệp, as_of_date và thời gian còn lại; trả các đoạn cùng score và nguồn bảng/cột/tài liệu. Dùng `backend.count_tokens()` khi ghép context.
3. Corpus ban đầu gồm rulebook, README, mô tả schema/binding và danh mục có nguồn. Tách đúng split/doanh nghiệp; không index đáp án benchmark, review/log hoặc lời giải của các câu đang đánh giá.
4. Embedding dùng Sentence Transformers, CLS pooling + normalize, vector 1024 chiều. Bắt đầu trong mức 2048 token đã nêu ở model card; cấu hình checkpoint 8192 không đồng nghĩa chất lượng ở độ dài đó đã được kiểm chứng.
5. Reranker dùng Transformers sequence classification: chấm từng cặp query–passage và trả một điểm/cặp. Giữ tổng input trong 2304 token gồm special tokens; không dùng nó như embedding model.
6. Chunk theo mục có nghĩa, giữ điều kiện/ngoại lệ đi cùng nhau. Top-k, chunk size, batch size và budget là cấu hình do Thịnh chọn/đo, không hard-code trong runner. Lưu index local ngoài dataset; cache phải gắn hash corpus/model/config.
7. Thêm context có provenance vào prompt, giữ rulebook/README và kiểm tra hiện có. Tính token bằng tokenizer generator, chừa output budget; không đủ chỗ thì giảm đoạn RAG. Tắt RAG hoặc không có đoạn phải giữ prompt baseline.
8. Retrieval/rerank nằm trong deadline câu. Ghi rõ lỗi/fallback; thiếu checkpoint hoặc sai hash phải báo khi khởi tạo. Không có đoạn không phải bằng chứng unanswerable.
9. Mở rộng inventory để cộng embedding/reranker theo mode, cập nhật artifact/config/log/resume. Giữ manifest generator cũ tương thích.
10. Kiểm tra cơ chế bằng fixture, chạy smoke model thật rồi đo trước/sau.

**SQL/công cụ tiếp tục tính trên đầy đủ dòng thỏa điều kiện.** Không lấy top-k giao dịch rồi cộng tiền. Tên/mã/size cần đối chiếu khóa chính xác; semantic top-1 không tự chứng minh cùng thực thể.

## 4. Phạm vi file và phối hợp

| Thịnh phụ trách | Nội dung |
|---|---|
| `src/datamate/retrieval/` | Corpus/index, encoder/reranker, strategy và factory |
| `agent/runner.py`, `agent/prompts.py`, `application.py` | Injection retriever, context và deadline |
| `models/registry.py`, `models/artifacts.py` | Inventory và artifact RAG |
| Cấu hình, dependency, diagnostic/đóng gói | Bật/tắt RAG, cache identity và log |

Các đường dẫn rút gọn thuộc `src/datamate/`. Với file chung như `config.py`, `application.py`, dependency và Docker, Thịnh/Hải ghép từng patch nhỏ, không chép đè toàn file. Thịnh có thể triển khai và đánh giá RAG trên Transformers trước khi vLLM hoàn thành.

Khi ghép với Hải, kiểm VRAM của toàn bộ model và đo lại bản kết hợp trên một bộ câu hai người chọn, có baseline tương ứng. Không bắt buộc chạy đủ ma trận thí nghiệm giữa mọi backend/chế độ.

## 5. Bàn giao

- Code/PR, preset bật/tắt RAG, manifest model/index, dependency và lệnh chạy.
- Kết quả baseline và RAG: chất lượng đáp án, thời gian, GPU/model/bộ câu và các lỗi/regression.
- Kiểm tra scope, cache invalidation, pooling/shape, rerank alignment, context/deadline/fallback, inventory và Docker offline.

Tuân thủ [ngữ cảnh dự án](../competition-context.md): tổng model inference ≤15B gồm embedding/reranker/assistant; checkpoint trước 01/10/2026, pin revision/hash/license. Hai model RAG ghép được với Qwen 9B/Gemma 12B hiện có, nhưng vượt 15B với bản upscaled 14B. Không dùng test để training hoặc thu thập đáp án test; `/data` chỉ đọc, index/cache ngoài dataset; output/log đúng đường dẫn và private test ≤6 giờ gồm load/index. Không quyết định đáp án theo ID.

Tham khảo: [kiến trúc](../architecture.md), [model evidence](../model-evidence-rag-2026-11-09.json).

