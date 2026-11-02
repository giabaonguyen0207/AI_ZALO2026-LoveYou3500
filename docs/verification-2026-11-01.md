# Kiểm tra triển khai ngày 01/11/2026

Nguồn: lệnh thực sự chạy trong workspace E:, dataset đã nhận trước đó, metadata công khai từ Hugging Face. Các mục dưới đây là kiểm tra phần mềm và dữ liệu, không phải benchmark LLM.

## Đã chạy

- Python local 3.13.7, môi trường `.venv` riêng và cache trên E:.
- 32 tests qua, gồm contract/type, alias/diacritics/list/table/PA, CSV/XLSX/SQLite, nguồn không bị sửa, SQL chỉ đọc/timeout, query repair, chiếu đáp án từ SQL, self-review, batch/resume, model hash, nguồn train/holdout và API. Có một cảnh báo deprecation từ Starlette TestClient/httpx; test vẫn qua.
- Ingest đủ **30 doanh nghiệp train**, không có exception, **3.997.042 dòng raw** không tính calendar. Catalog có **383 bảng/view**, gồm view tổng hợp và calendar của từng doanh nghiệp. Đây không phải số bảng nguồn duy nhất hoặc số giao dịch đã làm sạch.
- Có 115 sheet với header sau dòng 1: dòng 2 (34), dòng 3 (42), dòng 4 (39). Kiểm tra đơn giản không thấy header là mã ORD/HD của bản ghi. Chưa chứng minh toàn bộ ô/header/semantics đều đúng.
- Tokenizer thật của Qwen/Gemma tại revision đã khóa tải về `.cache/tokenizers`; kiểm tra template/prefix/loss boundary cho một câu ở từng doanh nghiệp train. Qwen prompt ban đầu 5.867–10.989 token; Gemma 6.128–11.527. Chưa gồm tool observations. Đã xử lý BPE ở biên prompt và empty thought primer của Gemma.
- Tạo format-only dataset: 2.370 train / 630 validation, chia 24/6 doanh nghiệp, seed 42. Đây là dữ liệu thử pipeline, không phải trajectory suy luận đã học.
- Các cell Python trong notebook được parse bằng AST; chưa chạy notebook trên Colab GPU.
- Web local đã được mở bằng trình duyệt: chạy demo qua API, hiển thị abstain, 0 lượt model/0 tham số thật trong demo; thử chọn Qwen khi thiếu checkpoint và thấy thông báo chuẩn bị model; mở trace trên màn hình nhỏ. Đã kiểm tra API bằng TestClient. Screenshot ở `outputs/web-desktop-preview.png` và `outputs/web-mobile-preview.png`.
- Docker Desktop 29.5.2 build được `datamate-dev:local` từ Python 3.13 slim đã pin digest. Smoke test 3 câu train dùng `/data` chỉ đọc, rootfs chỉ đọc, `/work`/`/tmp` tmpfs và mạng `none`; xuất đủ 3 dòng JSONL abstain và log. Đầu ra ở `result/docker-smoke-final/`; đây là CPU/demo, không có model weights.
- Đã đọc metadata/checkpoint tree ở full SHA trước 01/10/2026 cho Qwen 9B, Gemma 12B IT và Gemma MTP assistant; lưu LFS SHA256 và license trong `model-evidence-2026-11-01.json`. Đây là metadata công khai, chưa là checksum trọng số local đã tải.

Các output kiểm tra lớn nằm trong `outputs/` hoặc `result/`, được gitignore. Có thể chạy lại bằng tools/CLI trong README. Không dùng public/private test để tạo dataset training; audit mới của phiên này chỉ ingest train.

## Chưa chạy hoặc chưa xác nhận

- Chưa tải full weights; chưa kiểm tra tensor count/hash trên full artifacts local.
- Chưa generate bằng Qwen/Gemma, chưa chạy native MTP assistant, chưa đo peak VRAM, speedup, acceptance rate hoặc accuracy/PA của LLM.
- Chưa fine-tune/merge adapter; chưa có trajectory baseline đúng được thu từ model thực.
- Chưa kiểm tra CUDA/torch/bitsandbytes/trainer trên Colab hay RTX 5090; chưa chạy private test hoặc chứng minh hoàn tất trong 6 giờ.
- Chưa có Docker guideline chính thức, nên image thi và giao diện chấm cuối còn phải đối chiếu.
- Local evaluator dựa trên ngữ cảnh/rulebook đã cung cấp; không phải bộ chấm chính thức.

## Sự cố môi trường và cách xử lý

- Python hệ thống có các package metadata nhưng thiếu dependency. Đã dùng `.venv` riêng và cài dependency trên E:.
- Docker không tìm thấy bind mount đường dẫn Unicode của dự án. Smoke test dùng Docker cp với đúng subset train và volume tạm, rồi chép kết quả về E:.
- Hệ thống duyệt tự động từ chối phép thử mount toàn ổ E: vì phạm vi đọc rộng hơn cần thiết. Không thực hiện phép thử đó; kiểm tra đã hoàn thành bằng subset dự án.

Không diễn giải các kiểm tra trên thành kết luận model nào tốt hơn trên DataMate. Dùng harness holdout để đo sau khi chọn/tải checkpoint.
