# Model hiện hành

Registry runtime không còn Qwen 3.5 4B. Máy local không chạy full-checkpoint inference; chọn checkpoint và môi trường GPU theo nhu cầu đo.

| Khóa model | Tham số theo inventory | Vai trò |
|---|---:|---|
| qwen35-9b | 9.653.104.368 | Qwen chính thức, baseline mặc định |
| gemma4-12b | 11.959.730.224 | Generator khác họ để thử |
| gemma4-12b-assistant | 422.856.964 | MTP assistant của Gemma; không phải bộ kiểm tra nghiệp vụ |
| qwen35-upscaled14b-v2 | 14.599.485.040 | Checkpoint thử nghiệm; không đủ ngân sách thêm cả hai encoder RAG |

Qwen 9B + hai encoder RAG: 10.788.614.897 tham số. Gemma 12B + hai encoder: 13.095.240.753; cộng MTP assistant là 13.518.097.717. Upscaled 14B + hai encoder vượt 15B. Self-review gọi lại cùng checkpoint không cộng thêm một model.

Revision/hash/ngày công khai và giấy phép nằm trong registry và [evidence gốc](model-evidence-2026-11-01.json), [evidence Pinkstackorg](model-evidence-pinkstack-2026-11-06.json), [evidence RAG](model-evidence-rag-2026-11-09.json). Evidence lịch sử có Qwen 4B được giữ để truy nguyên, không phải model được hỗ trợ cho run mới.

Stage-model online tải đúng revision và seal artifact; inference chỉ nhận local files đã verify. Quantization không giảm số tham số theo giới hạn cuộc thi. Embedding/reranker chưa được thêm vào registry/inventory runtime trong đợt dọn này; đó là đầu việc của Thịnh.

Chưa có benchmark GPU mới từ việc dọn bố cục. Các số PA/tốc độ đã nhận trước đó ở docs/history. Với thay đổi mới, chạy baseline trước trên bộ câu đã chọn và giữ cùng điều kiện khi so.
