# Huấn luyện trên Colab

**Bộ Gemma seed 72+100, 05/11/2026:** đã chuẩn bị 136 câu/144 quyết định assistant theo fast v5, 17 câu hàng chờ và 18 câu validation seed giữ ngoài train. Dùng notebook `DataMate_Colab_Seed72_100_QLoRA.ipynb` cùng ZIP dataset/source riêng; không cần chạy lại baseline toàn bộ train. Xem [hướng dẫn Gemma trên Colab Pro, kiểm chứng và giới hạn](gemma-seed72-100-colab-pro.md). Chưa chạy GPU fine-tune. Các mục bên dưới giữ quy trình training tổng quát trước đó.

Nếu cần hướng dẫn thao tác từng bước trên PowerShell/Drive/Colab, đọc **phần D** trong [Hướng dẫn chạy từng bước](huong-dan-chay-tung-buoc.md) trước. Tài liệu này giải thích kỹ thuật chuẩn bị dữ liệu và training.

Triển khai ngày 01/11/2026. Chưa chạy GPU fine-tune. Các command dưới đây là quy trình sau khi chuẩn bị trọng số và runtime tương thích.

## Nguồn và holdout

Gói thực tế `train/train` có 30 doanh nghiệp, 3.000 câu hỏi/đáp án; không có SQL/tool calls gán nhãn. Không tự tạo lời giải trung gian từ đáp án rồi gọi đó là ground truth.

`configs/training_sources.json` lưu SHA256 mọi file train và fingerprint `5d598c5d603020c8077df7802cc6fb06b364af9914eb88425bd7664c639978ea`. Mang lên Colab, gói giữ nguyên nội dung được nhận diện dù đổi đường dẫn. Nguồn khác bị từ chối. Không dùng public/private hoặc test đổi tên.

Seed 42 giữ riêng 6 doanh nghiệp: `biz_107`, `biz_111`, `biz_115`, `biz_120`, `biz_123`, `biz_127` (630 câu). 24 doanh nghiệp còn lại có 2.370 câu. ID chỉ dùng chọn partition/ghép nhãn. Không trộn câu của cùng doanh nghiệp vào train và validation.

## Dataset

**Format warm-up**: đầu vào chứa typed result đã có, đầu ra chép/chuẩn hóa JSON. Dùng để thử pipeline training. Không dạy lập SQL, hiểu rulebook hay giải câu hỏi; không coi loss bài này là chất lượng agent.

**Verified trajectories**: baseline chạy trên train questions/data/rulebook, không nhìn nhãn trong prompt. Trình chuẩn bị ghép log với train answers, chỉ giữ đáp án cuối khớp local evaluator và tái dựng action/tool observations thành hội thoại. Đây là lọc theo **đáp án cuối**; query vẫn có thể tình cờ đúng. Nên kiểm tra chất lượng một mẫu trace train.

```powershell
.\.venv\Scripts\python.exe -m datamate prepare-training --split-root train/train --kind format --output outputs/training/format
.\.venv\Scripts\python.exe -m datamate run --config configs/qwen35-9b.json --ids outputs/training/format/train_ids.json --result-dir result/teacher-train-01
.\.venv\Scripts\python.exe -m datamate prepare-training --split-root train/train --kind trajectories --logs result/teacher-train-01/logs --output outputs/training/trajectories
```

Nếu chưa có trajectory đúng, lệnh báo lỗi, không sinh SQL giả. Khi chỉ thu trace trên 24 doanh nghiệp train, validation trajectory file có thể rỗng; sau training phải đánh giá **generation** trên 630 câu giữ riêng.

## LoRA / QLoRA

LoRA giữ base weights cố định, học các ma trận nhỏ; QLoRA thêm nén base 4bit. Script dùng Hugging Face Trainer + PEFT, không cần TRL. Chọn linear layers trong `language_model`; thiếu module phù hợp thì báo lỗi để kiểm tra kiến trúc.

- Rank 16, alpha 32, batch 1, gradient accumulation 8, learning rate 1e-4 là cấu hình khởi điểm, chưa tối ưu.
- Gradient checkpointing giảm peak VRAM bằng tính lại activations.
- Loss chỉ tính trên assistant output. Đã kiểm tra ranh giới với tokenizer thật, xử lý BPE và empty thought primer Gemma đã pin.
- Dùng 16K context. Prompt thực tế dài tới ~11,5K trước tool results; 4K/8K sẽ bỏ nhiều mẫu. Không tự cắt rulebook; báo số mẫu quá dài trong report.
- Chạy 50 bước đầu để đo thời gian/peak VRAM trước khi chọn epoch. Checkpoint định kỳ hỗ trợ resume; lưu version, config, seed, dataset hash và metrics.

```powershell
.\.venv\Scripts\python.exe -m datamate train --config configs/lora-qwen35-9b.json
.\.venv\Scripts\python.exe -m datamate train --config configs/lora-qwen35-9b.json --resume checkpoints/qwen35-9b-lora/checkpoint-50
```

Notebook sửa đường dẫn theo Drive và cho chọn `gemma4-12b`. Không fine-tune MTP assistant như model chat độc lập. Kiểm tra GPU/CUDA/PyTorch thực tế; mã chưa phải xác nhận stack này chạy trên mọi GPU.

## Merge và đánh giá

```powershell
.\.venv\Scripts\python.exe -m datamate merge --model qwen35-9b --base models/qwen35-9b --adapter checkpoints/qwen35-9b-lora --output models/qwen35-9b-datamate
```

Merge dùng base không nén và adapter cùng revision, xuất safetensors/tokenizer/license/training report/manifest. Cần đủ RAM/VRAM/disk. Count phải khớp inventory base; không coi adapter chưa merge là miễn tham số. Đổi `model_dir` sang bản đã merge, giữ `model_key` để truy nguyên base.

So base/fine-tuned trên cùng `validation_ids.json`. PA là chỉ số mục tiêu. Nếu fine-tuned thấp hơn thì giữ baseline. Nếu dùng Gemma assistant sau fine-tune, đo lại với target đã đổi. Xuất lockfile runtime thật và chạy Docker offline trước khi nộp.
