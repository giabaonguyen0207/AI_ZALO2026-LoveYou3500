# Hướng dẫn làm việc trong dự án DataMate

## Đọc trước khi làm việc

- Đọc `docs/competition-context.md` để hiểu cuộc thi và trạng thái thông tin.
- `README.md` là điểm bắt đầu và mô tả bố cục/quy trình hiện hành của nhóm.
- Ngữ cảnh cuộc thi do người dùng cung cấp ngày 30/10/2026. Đây là bản tổng hợp, chưa phải bản sao tài liệu chính thức đã được xác minh độc lập.
- Khi có tài liệu chính thức hoặc dữ liệu thực tế, dùng chúng để kiểm chứng bản tổng hợp và ghi rõ các điểm khác biệt; không âm thầm coi giả định là quy định đã xác nhận.

## Phạm vi và cách hỗ trợ

- Người dùng quen Python và kiến thức lập trình cơ bản, chưa từng xây AI Agent. Trao đổi bằng tiếng Việt, giải thích thuật ngữ và lý do của các quyết định khi cần.
- Quyết định được chủ dự án duyệt ngày 09/11/2026: modular monolith theo Adapter + Strategy; Hải phụ trách vLLM, Thịnh phụ trách RAG. Đọc hai plan trong `docs/plans/`; không coi phần còn nằm trong plan là đã triển khai.
- Máy cá nhân chỉ phát triển và test CPU; full-checkpoint inference/training trên Colab/Kaggle/GPU thuê. Qwen 4B và web local đã bỏ khỏi runtime; giữ Transformers làm baseline và DemoBackend cho kiểm thử.
- Mỗi người tự chọn benchmark, chạy baseline trước rồi bản tích hợp trên cùng tập đã chọn. Không bắt buộc Team200 và không so điểm từ hai tập khác nhau để kết luận cải thiện.
- Dataset/model/output/cache/archive không qua Git. Kho `archive/2026-11-09/` là local, giữ các artifact frozen nguyên hash; dùng `outputs/runs/<người>/<run-id>/` và `outputs/packages/` cho lượt mới. Đọc README, `docs/run-gpu.md`, `docs/team-workflow.md` trước khi đổi entrypoint/đường dẫn.
- Trước khi thiết kế dựa trên dữ liệu, kiểm tra tài liệu và file thực tế trong workspace. Không suy đoán schema, tên cột, chất lượng dữ liệu, base image chính thức hoặc hiệu năng.

## Các ràng buộc phải nhớ từ bản tổng hợp

- Mỗi split có `rulebook.md` riêng và quy tắc có thể thay đổi. Các ví dụ về doanh thu, giá vốn, thời gian trong ngữ cảnh không phải quy tắc cố định cho mọi split.
- Giải nghĩa thời gian trong câu hỏi theo `as_of_date`; dùng lịch và rulebook của split tương ứng khi cần.
- Phân biệt `unanswerable` (thiếu thông tin cần thiết trong dữ liệu) với `abstain` (chọn không trả lời). Đúng +1, sai -1, bỏ qua 0; `unanswerable` sai vẫn bị -1.
- Đầu ra phải đúng cả `type` lẫn giá trị; thứ tự của `list` có ý nghĩa. Xem đầy đủ quy tắc so khớp trong tài liệu ngữ cảnh.
- Tổng tham số của tất cả model dùng khi inference không quá 15B; tính cả embedding, reranker, classifier và toàn bộ tham số MoE. Quantization không làm giảm số tham số được tính.
- Trọng số model phải phát hành công khai trước 01/10/2026 và có giấy phép phù hợp; phải xác minh trước khi chọn model. Người dùng đã nhắc lại rõ ràng điều kiện này. Ngày công bố tên model không đủ: phải kiểm tra ngày công khai của chính checkpoint trọng số được dùng và lưu revision/hash. Không mặc định dùng `main` nếu trọng số đã được thay sau thời hạn.
- Inference phải offline hoàn toàn trong Docker, không tải tài nguyên hoặc gọi dịch vụ bên ngoài, không có người can thiệp.
- Không hard-code đáp án hoặc dùng logic quyết định đáp án dựa trên question ID/business ID. ID vẫn được dùng cho định tuyến dữ liệu, ghép câu hỏi–đáp án, đầu ra và log theo định dạng yêu cầu.
- Không dùng public/private test để huấn luyện hoặc thu thập đáp án test. Không coi việc kiểm tra dữ liệu đầu vào lúc inference là quyền dùng test để huấn luyện.
- Dataset tại `/data` là chỉ đọc. Kết quả ở `/result/submission.jsonl`; log từng câu ở `/result/logs/`, tối thiểu có ID, đáp án cuối và thời gian chạy.
- Toàn bộ private test phải chạy trong 6 giờ, tính cả load model. Điểm cuối dựa trên Docker do ban tổ chức chạy lại.

## Cập nhật ngữ cảnh

- Khi nhận dữ liệu hoặc tài liệu mới, ghi nguồn, ngày nhận, nội dung đã xác minh và phần còn chưa rõ.
- Giữ tách biệt yêu cầu cuộc thi, đặc điểm quan sát từ dữ liệu và quyết định thiết kế của đội.
- Không ghi kết quả benchmark hoặc trạng thái hoàn tất nếu chưa thực sự chạy và kiểm tra.
