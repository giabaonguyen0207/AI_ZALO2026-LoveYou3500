# Training trên GPU

Training source được giữ trong src/datamate/training và được đưa vào build context Docker. Full-checkpoint training/inference chạy trên GPU từ xa, không cài stack CUDA vào môi trường phát triển Windows.

Đường chạy hiện hành của Bảo là **Qwen3.5-9B SQL stage1 + stage2**, đã fine-tune1+0,5epoch. Source/dataset/config/report frozen đi cùng mỗi release; inference dùng [full model merged](colab-finetuned-compare.md). Các notebook LoRA tổng quát, Gemma Seed72 và SQL validation L4 phiên bản cũ đã nghỉ hỗ trợ; dùng bản frozen trong archive nếu cần tái lập, không trỏ vào notebook đã bỏ.

Bộ SQL SFT mới cho Qwen3.5-9B, nguồn 3.000 câu TRAIN và notebook Colab riêng: [hướng dẫn và phạm vi kiểm chứng ngày 09/11/2026](training-sql-qwen35-9b-2026-11-09.md). Chỉ các mẫu đã nhận mới dùng train; giữ validation theo doanh nghiệp và hàng chờ sửa riêng.

## Nguồn và đánh giá

Chỉ dùng nguồn TRAIN được allowlist trong configs/training_sources.json, synthetic hoặc nguồn hợp lệ khác theo quy định. Không dùng public/private test để training. Bộ train đã nhận gồm 30 doanh nghiệp/3.000 câu; split seed 42 giữ riêng 6 doanh nghiệp/630 câu validation. ID phục vụ chọn partition/ghép nhãn, không quyết định đáp án.

Format warm-up chỉ dạy định dạng; trajectories được lọc theo đáp án cuối vẫn cần kiểm query/tool, không mặc định là ground truth reasoning. Fine-tune phải so baseline trước/sau trên cùng tập đánh giá, khai báo overlap nếu tập từng được dùng debug/curate.

## Quy trình tổng quát trên máy GPU

Chuẩn bị source và checkpoint local đã pin, cài môi trường GPU tương thích:

```bash
python -m pip install -e '.[train]'
python -m datamate prepare-training --split-root train/train --kind format --output outputs/training/format
python -m datamate run --config configs/qwen35-9b.json --ids outputs/training/format/train_ids.json --result-dir outputs/runs/hai/teacher-train-01
python -m datamate train --config configs/lora-qwen35-9b.json
python -m datamate merge --model qwen35-9b --base models/qwen35-9b --adapter checkpoints/qwen35-9b-lora --output models/qwen35-9b-datamate
```

Các lệnh là bước riêng, không tự tạo đủ trajectories từ format. Sửa đường dẫn theo runtime và đọc --help/config trước khi chạy dài. LoRA/QLoRA vẫn được giữ; quantization hữu ích trên GPU nhỏ và không giảm số tham số phải kiểm kê.

## Bộ Gemma seed đã khóa

Bộ 136 câu/144 quyết định, ZIP dataset và source riêng được giữ nguyên trong archive/2026-11-09/outputs/. [Tài liệu gốc](history/gemma-seed72-100-colab-pro.md) ghi source/pipeline của bộ này.

Không dùng source mới để ghi đè gói source frozen hoặc bỏ guard hash để ép seed cũ chạy. Dùng đúng dataset/source/notebook đi kèm đã lưu, hoặc chuẩn bị release mới đã replay/kiểm chứng. Module training/seed.py vẫn được giữ vì pipeline SQL dùng chung phần tạo messages; bỏ generator notebook/packager Gemma cũ khỏi tools hiện hành.

Khi chốt checkpoint cuối: giữ training config/report/provenance, merge đúng base/revision, seal artifact, đo generation quality và kiểm Docker offline.
