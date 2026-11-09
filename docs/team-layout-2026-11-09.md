# Dọn bố cục làm việc nhóm · 09/11/2026

Thực hiện theo phương án người dùng đã duyệt. Nhánh local: **codex/team-gpu-layout**, tách từ qwen35-9B-pre-finetune/pipeline 6. Chưa commit, push hoặc merge GitHub trong đợt này.

## Đã thay đổi

- Bỏ Qwen 4B khỏi registry/preset/notebook hiện hành và xóa weights local; giữ metadata/license trong archive. Bỏ preset laptop 9B 4bit nhưng giữ hỗ trợ quantization/QLoRA trên GPU từ xa.
- Lưu web/API và script khởi động cũ trong archive; bỏ lệnh serve và dependency FastAPI/Uvicorn. Giữ CLI, Transformers baseline, DemoBackend và Docker dev smoke.
- Tách test API khỏi test training/artifact; giữ kiểm tra dữ liệu training, hash và inventory.
- Chuyển báo cáo cũ vào docs/history, paper vào docs/references, roadmap vào docs/research và plan vào docs/plans. Notebook training ở notebooks/training.
- Notebook GPU chính được sinh từ một tool duy nhất, có cấu hình Colab/Kaggle/GPU thuê, gọi chung application.run_batch. Loại outputs cũ trong notebook; lịch sử nguyên bản nằm trong source backup.
- Thay môi trường CUDA Windows bằng môi trường CPU. Môi trường kiểm thử thực tế: PyTorch **2.15.0+cpu**, CUDA **None**, Transformers **5.22.2**. requirements-dev.lock.txt chỉ khóa phần development nhẹ.
- Gom output mới theo outputs/runs và outputs/packages; gói source có manifest/hash, loại data/model/archive/history. Thêm helper tạo Docker context với model được chọn, không tự build/nộp image.
- Cập nhật README, AGENTS, hướng dẫn GPU/GitHub/Docker/artifact và chống ghi đè evidence lịch sử từ tool verify_model_sources.

## Bảo toàn

Kho **local**, không qua Git: archive/2026-11-09/.

- pre-cleanup-source.zip: 177 file, SHA256 **8a8443f8c09e0cbf1a9af6d992c63987bc62bd9e0cb72b705d91e7b8b904b5e9**; gồm cả các thay đổi tài liệu chưa commit trước khi dọn.
- outputs/ và result/ cũ: **2.388 file** được đối chiếu SHA256 trước/sau chuyển. Các ZIP benchmark/SFT/rollback không bị viết lại.
- Qwen 9B chuyển sang archive/2026-11-09/models/qwen35-9b, kiểm hash đầy đủ theo artifact manifest, 9.653.104.368 tham số. Chưa upload kho này lên Drive/Kaggle.
- **34 file Python lõi** trong agent/data/evaluation/training, application và domain khớp byte với bản trước dọn. Không thay thuật toán nghiệp vụ để lấy điểm mới.
- Phần quy định từ mục 1 trở đi trong competition-context giữ nguyên nội dung; provenance dài được chuyển thành tài liệu history.

## Dung lượng

Đã dọn weights 4B, cache tạo lại được và môi trường CUDA cũ; giữ tokenizer. Chênh dung lượng trống đo sau dọn khoảng **38,14 GB** (có thể dao động do file tạm/Docker). Tại lần đo: .venv CPU khoảng 0,88 GB; archive khoảng 20,74 GB. Di chuyển Qwen 9B trong cùng ổ không được tính là giải phóng thêm dung lượng.

Manifest local: source-backup-manifest.json, move-plan.json, artifacts-before.json, preservation-verification.json, deleted-generated-paths.json và cleanup-storage.json trong archive/2026-11-09/.

## Kiểm chứng

- **229 tests passed** trong môi trường CPU mới (lượt đầy đủ sau thay đổi: 20,92 giây). Có kiểm tra package không chứa data/model/archive, notebook sinh đúng source, CLI bỏ serve, cache/model và Docker staging bằng fixture. Có regression test chặn kernel dùng source cũ khi ZIP đổi.
- Chạy CLI demo trên ba câu TRAIN, ghi đủ ba submission/log. Demo luôn abstain; đây là kiểm tra luồng, không phải PA hoặc hiệu năng model.
- Tạo kit Team200 mới từ source hiện tại; hai gói participant/coordinator giữ nguyên hash đã phát. Gói source và notebook mới được kiểm CRC, hash và compile cell.
- Không chạy full-checkpoint GPU/Colab/Kaggle, không train hay đo PA mới. Cấu hình ba nền tảng cần được chạy trên runtime GPU thực tế.
- Docker Desktop API trả lỗi truy cập trong phiên. Thử build dev chưa hoàn thành và đã dừng; chưa có image Docker mới được xác nhận, chưa build/nộp image thi. Helper staging được kiểm bằng fixture, không thay rehearsal GPU.

## Bước tiếp theo

Chủ dự án làm theo [team-workflow](team-workflow.md) để commit/push nhánh và PR vào main. Nhóm pull source rồi nhận dataset/checkpoint riêng. Khi chốt bản thi, đối chiếu base image BTC và thực hiện [quy trình Docker](docker.md).
