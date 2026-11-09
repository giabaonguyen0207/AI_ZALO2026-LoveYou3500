# Artifact, dữ liệu và lịch sử

Git chỉ chia sẻ code, cấu hình và tài liệu. Dataset, checkpoint, wheelhouse, cache và kết quả chạy được chia sẻ riêng; README không chứa link công khai tới nhãn benchmark.

## Bố cục mới

- outputs/runs/hai/<run-id>/ và outputs/runs/thinh/<run-id>/: submission, log, cấu hình và số đo của mỗi người.
- outputs/packages/: source ZIP + manifest, kit benchmark mới hoặc build context Docker.
- models/: chỉ stage model cần dùng khi build; không đưa weights vào Git.
- archive/2026-11-09/: kho lưu **local**, không được upload tự động và không có trong git pull.

Đợt dọn đã chuyển nguyên bản outputs/ và result/ cũ vào archive/2026-11-09/. Gói benchmark v1/v2, SFT, source ZIP frozen, log/rollback và lịch sử kit được giữ nguyên. Danh sách và hash nằm trong artifacts-before.json; preservation-verification.json ghi kết quả đối chiếu sau chuyển.

Checkpoint Qwen 9B được chuyển vào archive/2026-11-09/models/qwen35-9b và kiểm hash đầy đủ theo manifest. Khi build Docker, truyền đường dẫn này cho --model-source; không cần tải lại. Qwen 4B đã nghỉ hỗ trợ; chỉ giữ metadata/license/cấu hình trong archive, weights được dọn.

Các tên cũ trong docs/history là đường dẫn tại thời điểm chạy. Phần outputs/... cũ hiện nằm dưới archive/2026-11-09/outputs/...; result/... cũng tương tự.

## Chia sẻ cho đồng đội

Chủ dự án cung cấp dataset/checkpoint từ kho riêng đã chọn. Người nhận kiểm SHA256/manifest rồi đặt đường dẫn trong config/notebook. Đợt dọn này không tự tạo link Drive, upload Kaggle hoặc công khai kho artifact.

Không sửa gói benchmark/SFT frozen để đổi source/runtime. Bộ Gemma seed cũ khóa theo source/pipeline riêng; phải dùng đúng source đi kèm hoặc chuẩn bị một release mới đã kiểm chứng. Không bỏ các guard hash để ép nó chạy trên pipeline mới.

Gói source hiện hành được tạo bằng tools/package_colab.py, loại archive/data/models/cache và docs/history/references/research. Bản mới có manifest; hãy giữ manifest cạnh artifact khi gửi cho nhóm.
