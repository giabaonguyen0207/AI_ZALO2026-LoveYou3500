<<<<<<< HEAD
# Zalo AI Challenge 2026 — DataMate

Thư mục làm việc của đội để chuẩn bị xây dựng AI Agent trả lời câu hỏi tiếng Việt trên dữ liệu kinh doanh có cấu trúc.

## Tài liệu hiện có

- [Ngữ cảnh cuộc thi](docs/competition-context.md): tổng hợp đầy đủ đề bài, cách chấm điểm, dữ liệu, giới hạn kỹ thuật và yêu cầu nộp bài do người dùng cung cấp ngày 30/10/2026.
- [Hướng dẫn cho agent hỗ trợ dự án](AGENTS.md): chỉ dẫn để các phiên làm việc sau đọc đúng ngữ cảnh, phân biệt thông tin đã có với điều chưa biết.
- [Đề xuất baseline và chi phí fine-tuning](docs/baselines-and-costs.md): nghiên cứu ngày 30/10/2026, lọc theo hạn công khai trọng số trước 01/10/2026; đây là các phương án đề xuất, chưa phải lựa chọn của đội.
- [Đánh giá Colab và dữ liệu ngày 31/10](docs/colab-assessment-2026-10-31.md): kết quả kiểm kê dữ liệu thực tế, ảnh tài nguyên Colab, đề xuất LoRA và giới hạn của full fine-tuning.

## Cập nhật ngày 31/10/2026

- Đã nhận train tại `train/train` (30 doanh nghiệp, 3.000 câu và đáp án) và public tại `public_test/public_test` (14 doanh nghiệp, 1.000 câu).
- Đã kiểm kê bằng `tools/inspect_dataset.py`; metadata ở `docs/dataset-audit.json`, khác biệt rulebook ở `docs/rulebook-diff.txt`.
- Chưa train hoặc benchmark model. Người dùng có Colab Pro khoảng 95,6GB GPU RAM và 400 units theo ảnh; tên GPU còn chờ xác nhận.

## Trạng thái ngày 30/10/2026

- Đã lưu ngữ cảnh; chưa chọn kiến trúc, model, framework hoặc kế hoạch triển khai.
- Chưa có bộ dữ liệu training/public, `rulebook.md`, `calendar.csv` hay Docker guideline riêng trong thư mục này.
- Chưa có mã nguồn agent, kết quả thử nghiệm hoặc benchmark.
- Bản tổng hợp chưa được đối chiếu độc lập đầy đủ với tài liệu chính thức. Khi có tài liệu gốc, cần xác minh trước khi dùng để quyết định triển khai hoặc nộp bài.

## Tiếp tục làm việc

Đọc `docs/competition-context.md` trước. Khi dữ liệu và Docker guideline được cung cấp, kiểm tra cấu trúc thực tế, đọc README của từng doanh nghiệp và rulebook của từng split, rồi mới chốt các quyết định triển khai.

Các mốc cần lưu ý theo bản tổng hợp: nộp giải pháp Docker trước **16/11/2026, 23:59 GMT+7**; nộp kết quả private trước **23/11/2026, 23:59 GMT+7**.
