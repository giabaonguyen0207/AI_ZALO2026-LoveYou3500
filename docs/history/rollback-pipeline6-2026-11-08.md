# Khôi phục pipeline 6 theo yêu cầu người dùng — 08/11/2026

Người dùng yêu cầu quay về hệ thống trước interface v7 vì lượt inference mới chậm, sau đó nhắc rõ cập nhật cả notebook infer 200 câu. Đã khôi phục engine **pipeline 6**, backend protocol **3**, diagnostic 100 câu **8**. Không áp dụng thêm tối ưu hay sửa suy luận trong đợt rollback.

## Mốc đã khôi phục

Source đối chiếu: `outputs/datamate-colab-source-before-interface-v7-2026-11-08.zip`, SHA256 `38407bd0bbdf97489f3de59d11c88c1f604a061f81b30914e8bba43ddbfbed64`. Toàn bộ file Python dưới src khớp byte với bản này. Các file giao tiếp/ledger/planning/prompts/tools/debug_log và helper Colab được khôi phục; module parameters và test riêng của v7 chuyển vào bản sao lưu, không còn là module/test đang hoạt động.

Giữ cấu hình trước v7: self_review, code_first, context 16K, output 768, tối đa sáu bước, timeout 90 giây; schema repair giới hạn 20 giây/512 token. Không đổi checkpoint, dataset, rulebook, calendar, as_of_date hay cách chấm.

## Hai workflow Colab đã đồng bộ

- **100 câu:** `notebooks/DataMate_Colab_Inference.ipynb` cùng `outputs/datamate-colab-source.zip`. Seed 20261103 và thứ tự 100 câu giữ nguyên. Suffix mới `random100-rollback-pipeline6-Nov8th-01`. Model/quantization/tự ngắt và output lịch sử trong notebook được giữ.
- **200 câu:** `notebooks/DataMate_Colab_Team200.ipynb` cùng `outputs/team200-colab/datamate-team200-colab-kit.zip`. Source bên trong là pipeline 6; notebook khóa hash mới và preflight yêu cầu >=6/backend 3. Suffix mới `team200-rollback-pipeline6-Nov8th-01`. Giữ 200 câu/seed 4000930841, participant/coordinator ZIP nguyên hash, submission/metadata sáu trường, diagnostic logs, resume và tự ngắt như trước. Runner vẫn ghi pipeline vào identity để không resume lẫn các engine khác nhau; thay đổi này chỉ phục vụ nhận diện phiên chạy.

Để có hiệu lực trên Colab, tải lại notebook và gói tương ứng lên Drive, **restart kernel** rồi chạy từ ô đầu. Ô load phải hiện **pipeline 6**. Model đã seal trên Drive được dùng lại. Không resume vào thư mục v7 hoặc lượt pipeline 6 đã chạy xong. Gói 200 câu dùng kit riêng, không dùng source ZIP của workflow 100 câu.

## Sao lưu và kiểm chứng

Trước khi hoàn tác đã sao lưu source workspace và notebook/gói hiện tại ở `outputs/rollback-pipeline6-2026-11-08/`. Kit 200 câu v7 trước rollback được giữ ở `outputs/team200-colab/history/27fbb37ba527/`; kit 6 gốc trước đợt nâng v7 vẫn ở history/f709f78e1bf0. Log inference không bị xóa hay ghi đè. Hai gói SFT Gemma frozen giữ nguyên hash.

Đã chạy toàn bộ suite: **225 passed**, một cảnh báo deprecation của TestClient. Kiểm tra bổ sung so sánh source, compile notebook, preflight import độc lập, CRC/per-file/hash các gói và mẫu câu được ghi ở `outputs/rollback-pipeline6-2026-11-08/verification.json`.

Đây là khôi phục mã và cấu hình, **chưa chạy lại checkpoint trên GPU, chưa có số đo tốc độ/PA mới sau rollback**. Không suy từ việc rollback rằng môi trường Colab hoặc mọi câu hỏi sẽ đạt một tốc độ cố định.
