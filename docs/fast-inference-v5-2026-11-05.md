# Inference fast v5 — 05/11/2026

Người dùng đã cho phép triển khai các hướng giảm độ trễ sau lượt 4. Mặc định mới là **code ánh xạ trước → SLM lập plan → code tính và chiếu đáp án**. Tên lượt notebook là fast v5; `Agent.pipeline_version=4` là phiên bản code, khác số lần inference.

## Đã thay đổi

- `schema_mode="code_first"`, `adaptive_preparation=False`: bỏ yêu cầu SLM mô tả toàn bộ rulebook và từng bảng lúc gặp doanh nghiệp mới. Code vẫn đọc/chuẩn hóa dữ liệu và tính profile; bước này vẫn có chi phí lần đầu.
- `require_plan=True` độc lập với chuẩn bị schema. Tắt SLM ánh xạ vẫn phải lập plan về đại lượng, phép toán, kỳ, population, điều kiện và type trên đường chạy model thật.
- `auto_execute_plan=True`: plan hợp lệ được code biên dịch thành SQL, chạy và chiếu kết quả thành final ngay. Không cần thêm lượt model chỉ để gọi `calculate` và `final`. Nghiệp vụ khác có thể gửi `query: {sql, parameters, columns}` trong cùng plan. Action SQL riêng vẫn trả kết quả về model để tiếp tục khám phá hoặc chọn `final`, tránh chốt nhầm truy vấn thăm dò.
- Chỉ thử SLM sửa schema khi plan/phép tính bị chặn và có phần còn thiếu: **tối đa một lần/câu, hai bảng, 20 giây, 512 token**, trong ngân sách câu hiện có. Nhãn lạ ngoài kỳ cần tính không tự gây thêm lượt sửa. Cache cả đề xuất thành công và thất bại theo nguồn, nhu cầu, model/config và phiên bản code; cache lỗi policy chung có thể dùng giữa các doanh nghiệp cùng split.
- Bản bổ sung tham chiếu ID dòng README/rulebook và ID cột. Kiểm tra từng trường độc lập; lỗi cột phụ không xóa mapping đang hoạt động. Không ghi đè vai trò đã rõ bằng đề xuất xung đột. Nhãn mơ hồ cần nguồn có nhắc đến nhãn; header hoặc riêng nhãn đó chưa đủ làm căn cứ.
- Trích dẫn xử lý Markdown có kiểm soát. Plan/điều kiện có thể dùng `evidence_id="Q"` để code gắn lại câu hỏi gốc vào log, giảm việc chép câu dài. Căn cứ và kiểm tra cấu trúc/kiểu/khóa vẫn không bảo đảm diễn giải ngôn ngữ luôn đúng.
- Bổ sung CamelCase, acronym Unicode như `ĐVT`, mô tả cột trong bảng Markdown và tên chung khi có khóa định danh rõ. Đây là quy ước dùng chung, không có logic đáp án theo question ID/business ID.
- Đưa gợi ý viết tắt vào prompt, giữ nhiều ứng viên cho từ đa nghĩa như `ck`, `dt`; giữ nguyên câu gốc, tên, mã, ngày và số tiền.
- Rút gọn hướng dẫn công cụ và phản hồi lặp. Toàn bộ rulebook/README vẫn có trong prompt ban đầu để giữ các ngoại lệ; chưa đổi engine hoặc triển khai prefix caching.
- Log JSON/Markdown giữ output model, plan, SQL do model sinh hoặc code biên dịch, tham số, kết quả/lỗi, sửa schema/cache và `auto_final`. Kết quả SQL bị chặn khi chiếu vẫn có ở `execution` để debug.

Các kiểm tra trạng thái, parse, khoảng ngày, khóa nối nhân bản, đơn vị, giá vốn và type vẫn chạy. Quy tắc định nghĩa tiền dòng chỉ chặn phép tính cần tiền dòng; không tự chặn phép đếm đơn/số lượng đã đủ căn cứ. Nguồn chưa ánh xạ không được coi là số 0 hay bằng chứng thiếu dữ liệu.

## Cấu hình

Notebook và `configs/submission.json` cùng dùng 6 bước tối đa, 768 token đầu ra/lượt, timeout câu 90 giây và các cờ trên. Các điểm vào khác dùng chung code; giới hạn bước/token có thể khác profile notebook/Docker.

Muốn bỏ hoàn toàn SLM sửa ánh xạ, đặt `schema_mode="code_only"` và giữ `adaptive_preparation=False`. SLM vẫn lập plan/SQL; schema chưa hỗ trợ có thể làm tăng bỏ qua. Chế độ `schema_mode="llm"` hoặc cờ cũ `adaptive_preparation=True` vẫn giữ để nghiên cứu luồng chuẩn bị toàn bộ có chi phí cao của v3.

Không đổi checkpoint/quantization hoặc thêm model phụ. Notebook giữ lựa chọn model của người dùng; Docker giữ model trong cấu hình submission. Ràng buộc offline, số tham số và thời điểm phát hành trọng số vẫn theo tài liệu dự án. Đây là cập nhật triển khai, không phải xác minh quy định mới của ban tổ chức.

## Kiểm tra thực tế

- **142 kiểm thử phần mềm đạt**, gồm 17 ca cho luồng nhanh. Backend giả lập kiểm tra một lượt cho phép tính thường; hai lượt khi bổ sung cột/trạng thái; cache cả lỗi; SQL sai/ghi dữ liệu bị chặn; SQL khám phá không tự chốt; code-only không gọi sửa schema; self-review vẫn có lượt kiểm tra riêng.
- **22/22 ca hồi quy tính toán train** với plan/SQL thủ công khớp nhãn tham chiếu. Nhãn được dùng sau tính toán. Đây không phải PA mới của SLM.
- Phát lại đúng plan model đã sinh trong trace tương tác lượt 4 cho câu “Hôm nay cửa hàng bán được bao nhiêu tiền?”: luồng mới cho **13.479.900 VND**, một lần gọi backend phát lại plan. Không chạy trọng số Gemma trong phép kiểm tra này; thời gian CPU đó không phải tốc độ SLM.
- Seed `20261103`, mẫu 100 câu và đúng **bảy ID đầu tiên của lượt 4** được kiểm tra lại; không lấy mẫu lại theo kết quả.
- Tokenizer offline trên bảy câu: Qwen 9B có 7.732–10.864 token đầu vào; Gemma 12B có 8.189–11.584. Cộng 768 token đầu ra đều trong 16.384. Riêng SYSTEM giảm 255/272 token khi giữ cùng user context. Lợi ích chính là bỏ lượt chuẩn bị/điều khiển thừa; chưa có số đo tốc độ GPU mới.

Chi tiết tại `outputs/verification-fast-v5-2026-11-05/`: `tests.xml`, `regressions.json`, `actual-plan-replay.json`, `sample-check.json`, `fast-tokenizers.json`, `verification.json`. Có thể lặp kiểm tra local bằng `python -m tools.verify_fast_pipeline`; cần artifact lượt 4 trong outputs. Plan/nhãn kiểm tra thủ công tách khỏi runtime agent.

## Chạy lại Colab

1. Thay source ZIP trên Drive bằng `outputs/datamate-colab-source.zip` mới, mở notebook inference cập nhật. Khởi động lại runtime/kernel rồi chạy lại chuẩn bị/load model để loại code cũ trong RAM.
2. Ô load phải hiện `Pipeline: 4 | schema: code_first | auto_execute_plan: True`. Giữ cùng model/quantization lượt 4 khi so sánh.
3. Ô diagnostic mặc định tên `random100-fast-v5-01`, `SAMPLE_SIZE=100`, `RUN_LIMIT=7`, seed `20261103`. Chạy bảy câu rồi kiểm tra đúng/sai/bỏ qua, thời gian và số lượt model trong ZIP xuất ra.
4. Sau khi xem kết quả, đổi `RUN_LIMIT=None`, giữ nguyên tên lượt/model/config rồi chạy lại để resume đến 100. Đổi config như `schema_mode` thì đổi tên lượt để tránh trộn kết quả.

`AUTO_DISCONNECT=True` trước đây của người dùng được giữ lại. Muốn xem bảy câu rồi chạy tiếp cùng runtime thì đặt False trước khi chạy. Các ô chọn model/quantization/Drive và các ô khác được giữ nguyên. Notebook trước cập nhật, gồm log lượt 4, đã sao lưu ở `outputs/notebook-backups/inference-before-fast-v5-4473b4f2.ipynb`.

**Chưa chạy inference GPU mới.** Chưa có PA hoặc cam kết số giây/câu cho fast v5. Cần đo lại với Gemma thật, tính cả chi phí chuẩn bị doanh nghiệp lần đầu, rồi mới quyết định chạy đủ mẫu hoặc tắt hẳn sửa schema bằng SLM.
