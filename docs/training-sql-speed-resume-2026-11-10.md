# Qwen3.5 SQL SFT v1.3 — tăng tốc và tiếp tục training

Nguồn: yêu cầu chủ dự án và notebook v1.2 cập nhật ngày 10/11/2026. Baseline A100-SXM4-40GB đã hoàn tất pilot 5 optimizer steps / 40 sequences trong 2.601,4365 giây, loss 0,5220301. Log smoke xác nhận thiếu FLA và causal-conv1d; Gated DeltaNet chạy bằng PyTorch tham chiếu. Ngoại suy khoảng 43 giờ/epoch là ước tính theo pilot, chưa phải lượt full đã đo.

## Thay đổi v1.3

- Cài FLA 0.6.0, Transformers 5.22.1, PEFT 0.21.2, Accelerate 1.15.0; khóa Torch và Triton theo Colab đang có. FLA wheel đã đối chiếu metadata và SHA từ PyPI. Không cài CUDA stack vào máy local.
- Kiểm tra implementation mà Transformers thật sự chọn, không chỉ kiểm package tồn tại. Nếu native causal-conv1d chưa có, dùng causal conv Triton chính thức của FLA qua adapter layout `[B,D,T] ↔ [B,T,D]` trong process training. Không cần biên dịch thêm extension CUDA. Gated DeltaNet vẫn dùng FLA mặc định của Transformers.
- Probe forward/backward trên tensor nhỏ, so với reference trong sai số BF16 trước khi load model 9B. Nếu probe không đạt hoặc còn DeltaNet fallback, dừng thay vì âm thầm chạy một lượt chậm.
- Model base được chép từ Drive sang SSD `/content`, xác minh hash đầy đủ rồi dùng cho training; checkpoint và adapter luôn lưu trên Drive.
- Giữ nguyên các cải tiến bộ nhớ của v1.2; data, context 16K, rank 16, batch 1, accumulation 8, LR 5e-5, seed 42 và mục tiêu assistant-only giữ nguyên. Mọi byte train/validation và token report giống v1.2.
- Ghi thời gian từng optimizer step vào `adapter/step_timing.jsonl`, log đầy đủ và environment riêng cho từng lần kết nối.

Tham khảo: [Qwen3.5 kernels](https://huggingface.co/docs/transformers/model_doc/qwen3_5#usage-tips-and-notes), [FLA installation](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md). **Chưa có số đo speedup GPU mới ở thời điểm viết hướng dẫn.** Probe operator không thay cho đo pilot model đầy đủ.

## Bắt đầu và so tốc độ

Chép hai ZIP v1.3 vào `MyDrive/DataMateSQLSFT/`, mở notebook `DataMate_Colab_Qwen35_9B_SQL_SFT_v1_3.ipynb` trên kernel Colab mới. Base model cũ trên Drive được tái sử dụng. Giữ A100, `METHOD="qlora"`, `STAGE="pilot"`, RUN_TAG v1.3 mới. Giữ bản pilot v1.2 đã hoàn tất để đối chiếu.

Notebook kiểm package/kernel, chạy smoke dài nhất rồi pilot 5 steps như cũ. So `train_runtime`, `step_timing.jsonl`, VRAM và loss; tách thời gian cài đặt/copy/JIT lần đầu khỏi thời gian training, đồng thời báo tổng chi phí. v1.3 lưu checkpoint thường xuyên hơn v1.2 nên runtime bao gồm thêm chi phí bảo vệ tiến độ. Không dùng loss pilot để kết luận PA tăng. Sau pilot mới ước tính lại epoch; sau full đánh giá agent trên cùng 630 IDs.

Nếu kernel không cài/import/chạy được, gửi log setup và `kernel-check-*.json` nếu có; không giảm context hoặc gỡ guard để tiếp tục bằng fallback. Nếu Torch/Triton không tương thích, cần sửa môi trường GPU cụ thể, không tự tải bản Torch khác trong notebook.

## Resume sau mất mạng hoặc runtime

Đầu ra ở `MyDrive/DataMateSQLSFT/runs/<RUN_TAG>/adapter/`. Trainer lưu cả trọng số, optimizer, scheduler, RNG, trainer_state và training arguments. Checkpoint sau bước 1, sau mỗi 5 optimizer steps và khi quá ngưỡng 10 phút tại ranh giới bước; giữ 3 checkpoint gần nhất. Evaluation độc lập mỗi 50 bước, tránh đánh giá 618 mẫu mỗi lần lưu.

`checkpoint_complete.json` được ghi cuối cùng với global_step, run identity và SHA của mọi file. Chỉ checkpoint có marker và checksum đầy đủ mới được dùng. Nếu bản mới nhất ghi dở/hỏng, `RESUME="auto"` chọn bản hợp lệ trước đó và chuyển bản hỏng sang tên `incomplete-checkpoint-...` trong cùng thư mục để bảo toàn và tránh ghi đè nó khi train lại đến bước đó.

Khi mạng trở lại và runtime cũ còn chạy: kết nối đúng server để theo dõi, không chạy thêm một trainer. Khóa OS ngăn hai process cùng output trên cùng runtime. Không dùng hai server Colab đồng thời cho cùng RUN_TAG.

Khi runtime đã mất: mở kernel mới, chạy lại notebook từ đầu, **giữ nguyên RUN_TAG, STAGE và cấu hình**, để `RESUME="auto"`. Source/dataset/model được dựng lại từ các gói và Drive, rồi trainer chọn checkpoint mới nhất đã kiểm. Nếu run đã hoàn tất, notebook kiểm checksum adapter và hiển thị báo cáo, không train lại.

Resume khác model, dataset, cấu hình (kể cả epochs/max_steps), precision hoặc thiếu state bị từ chối. Đừng đổi pilot thành full dưới cùng RUN_TAG; full dùng tag mới và bắt đầu từ base. Không nhập checkpoint cũ v1.2 vào v1.3 bằng cách đổi hash/identity. Nếu chưa có checkpoint đầy đủ, chưa có tiến độ bền vững để resume; giữ thư mục cũ để xem log và dùng tag mới nếu quyết định chạy lại từ đầu.

Không thể bảo đảm giữ phần tính toán sau checkpoint gần nhất, lỗi ghi/xóa dữ liệu Drive, hoặc khôi phục bằng môi trường không tương thích. Checkpoint định kỳ giảm công việc phải tính lại, không loại bỏ mọi khả năng mất tiến độ.

## Kiểm chứng

Metadata run/config và marker được ghi qua file tạm rồi thay thế. Metadata root ghi dở được phục hồi sau khi checkpoint được xác minh; cấu hình hợp lệ nhưng khác vẫn bị từ chối. Nếu ngắt ngay sau checkpoint cuối trước khi xuất adapter/báo cáo, lần chạy lại chỉ hoàn tất xuất kết quả, không train dư một bước.

Ngày 10/11/2026: 31 kiểm thử training/notebook liên quan đạt trên CPU với Transformers 5.22.1, PEFT 0.21.2, Accelerate 1.15.0. Kiểm thử toàn dự án: 272 đạt, 1 lỗi WinError 5 khi thay ZIP ở phần Team200; chạy riêng lại bài này đạt (không sửa code Team200). Probe GPU và pilot v1.3 chưa chạy. Theo yêu cầu tiết kiệm credit của chủ dự án, dừng sau đóng gói và kiểm checksum; không khởi chạy GPU/full training.

Các test CPU kiểm marker/checksum, fallback checkpoint ghi dở, identity đổi, duplicate process lock, completed-run reuse; một Trainer nhỏ và một adapter GPT2+PEFT đã mô phỏng ngắt ở step 1, resume đến step 3 và khớp trọng số cuối với lượt liên tục, gồm RNG/dropout. Không load full Qwen trên máy cá nhân. Xem log/verification trong `outputs/runs/review/sql-sft-qwen35-9b-20261110-speed/` để biết kết quả CPU/GPU thực sự đã chạy.
